import os
import json
import logging
from collections import defaultdict, OrderedDict
from functools import partial
from itertools import chain
import time
import sys
import re
import ipdb

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd

import utils

logger = logging.getLogger(__name__)

BERT_BASE_DIM = 768
MAX_DOCS = 1000

lookup_times = utils.Timer()  # stores measured times
sample_fetching_times = utils.Timer()
collation_times = utils.Timer()
prep_docids_times = utils.Timer()
retrieve_candidates_times = utils.Timer()


# only used by RepBERT
class CollectionDataset:
    """Document / passage collection"""

    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32', )
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32', )
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap",
                                   dtype='int32', shape=(self.collection_size, 512))

    def __len__(self):
        return self.collection_size

    def __getitem__(self, ind):
        """
        :param ind: integer index of passage/doc in collection
        :return: list (unpadded) of token IDs corresponding to ind
        """
        assert self.pids[ind] == ind  # TODO: pids happen to be stored as 0, 1, 2, ... in collection.tsv.
        return self.token_ids[ind, :self.lengths[ind]].tolist()


class EmbeddedCollection:
    """Document / passage collection in the form of document embedding memmap"""

    def __init__(self, embedding_memmap_dir, sorted_nat_ids=False, load_to_memory=False):
        """
        :param embedding_memmap_dir: directory containing memmap file of precomputed document embeddings, and the
            corresponding passage/doc IDs in another memmpap file
        :param sorted_nat_ids: whether passage/doc IDs in the collection happen to exactly be 0, 1, ..., num_docs-1, and
            doc embeddings are stored exactly in that order (is True for MSMARCO passage collection). A bit more efficient.
        :param load_to_memory: If true, will load entire embedding array as np.array to memory, instead of memmap!
            Needs ~26GB for MSMARCO (~50GB project total), but is faster.
        REMOVED emb_dim: dimensionality of document vectors. If None, `embedding_memmap_dir` will be used to infer it,
            and in case this is not possible, the default value BERT_BASE_DIM will be used
        """

        # if emb_dim is None:
        #     try:
        #         # assumes that name of embedding directory contains dimensionality like: "path/embedding_768"
        #         emb_dim = int(os.path.dirname(embedding_memmap_dir).split('_')[-1])
        #         logger.warning("Inferred doc embedding dimensionality {} from directory name: {}".format(emb_dim,
        #                                                                                                  embedding_memmap_dir))
        #     except ValueError:
        #         logger.warning(
        #             "Could not infer doc embedding dimensionality from directory name: {}".format(embedding_memmap_dir))
        #         emb_dim = BERT_BASE_DIM
        #         logger.warning("Using default document embedding dimensionality: {}".format(emb_dim))

        pids = np.memmap(os.path.join(embedding_memmap_dir, "pids.memmap"), mode='r', dtype='int32')

        self.sorted_nat_ids = sorted_nat_ids  # whether passage/doc IDs in the collection happen to exactly be 0, 1, ..., num_docs-1
        self.pid2ind = None  # no mapping dictionary in case sorted == True
        self.map_pid_to_ind = self.get_pid_to_ind_mapping(pids)  # pID-to-matrix_index mapping function

        self.num_docs = len(pids)
        # shape is necessary input argument; this information is not stored and a 1D array is loaded by default
        self.embedding_vectors = np.memmap(os.path.join(embedding_memmap_dir, "embedding.memmap"), mode='r',
                                           dtype='float32')
        self.embedding_vectors = self.embedding_vectors.reshape((self.num_docs, -1))
        if load_to_memory:
            logger.warning("Loading all collection document embeddings to memory!")
            self.embedding_vectors = np.array(self.embedding_vectors)
        logger.info("Collection document embeddings array shape: {}, "
                    "type: {}".format(self.embedding_vectors.shape, type(self.embedding_vectors)))
        return

    def get_pid_to_ind_mapping(self, pids):
        """Returns function used to map a list of passage IDs to the corresponding integer indices of the embedding matrix"""
        if self.sorted_nat_ids:
            assert np.array_equal(pids, np.array(range(len(pids)), dtype=int))  # TODO: REMOVE as soon as verified
            return lambda x: x
        else:
            self.pid2ind = OrderedDict((pid, i) for i, pid in enumerate(pids))
            return lambda x: [self.pid2ind[pid] for pid in x]

    def __len__(self):
        return self.num_docs

    def __getitem__(self, ids):
        """
        :param ids: iterable of some IDs of passages/docs in collection
        :return: (num_ids, emb_dim) slice of numpy.memmap embedding vectors corresponding to `ids`
        """
        inds = self.map_pid_to_ind(ids)
        return self.embedding_vectors[inds, :]


class CandidatesDataset:
    """The result of a retrieval stage: rank-sorted document / passages per query, in the form of doc/passage IDs
    memmap array"""

    def __init__(self, memmap_dir, max_docs=None, load_to_memory=False):
        """
        :param memmap_dir: directory containing memmap file of passage/doc IDs per query, and the
            corresponding query IDs, and the number of candidates per query in two more memmpap files
        :param max_docs: maximum number of candidates per query. Used to restrict the maximum number as it exists in the
            memmap file on disk.
        :param load_to_memory: If true, will load entire candidates array as np.array to memory, instead of memmap!
            Needs ~2GB for MSMARCO training set (212MB for dev set) with 1000 candidates per query
            (~XGB project total), but is faster.
        """
        # if max_docs is None:
        #     try:
        #         # assumes that name of memmap directory contains dimensionality like: "path/top500"
        #         max_docs = int(re.search(r"top([0-9]+)", memmap_dir).group(1))
        #         logger.warning("Inferred max. num. candidates {} from directory name: {}".format(max_docs, memmap_dir))
        #     except ValueError:
        #         logger.warning(
        #             "Could not infer max. num. candidates from directory name: {}".format(memmap_dir))
        #         max_docs = MAX_DOCS
        #         logger.warning("Using default maximum number of candidates: {}".format(max_docs))

        self.qids = np.memmap(os.path.join(memmap_dir, "qids.memmap"), mode='r', dtype='int32')  # unique query IDs
        self.lengths = np.memmap(os.path.join(memmap_dir, "lengths.memmap"), mode='r', dtype='int32')  # num. candidates per query

        self.qid2ind = None  # OrderedDict maps query IDs to the corresponding integer indices of the candidates matrix
        self.map_qid_to_ind = self.get_qid_to_ind_mapping(self.qids)  # qID-to-matrix_index mapping function

        self.num_queries = len(self.qids)
        # shape is necessary input argument; this information is not stored and a 1D array is loaded by default
        self.doc_ids = np.memmap(os.path.join(memmap_dir, "candidates.memmap"), mode='r', dtype='int32')
        self.doc_ids = self.doc_ids.reshape((self.num_queries, -1))
        if max_docs is not None:
            self.doc_ids = self.doc_ids[:, :max_docs]
        if load_to_memory:
            logger.warning("Loading all candidate document IDs to memory!")
            self.doc_ids = np.array(self.doc_ids)
        logger.info("Candidate document IDs array shape: {}, "
                    "type: {}".format(self.doc_ids.shape, type(self.doc_ids)))
        return

    def get_qid_to_ind_mapping(self, qids):
        """Returns function used to map a list of query IDs to the corresponding integer indices of the candidates matrix"""

        self.qid2ind = OrderedDict((qid, i) for i, qid in enumerate(qids))
        return lambda x: [self.qid2ind[qid] for qid in x]

    def __len__(self):
        return self.num_queries

    def __getitem__(self, qid):
        """
        :param qid: query ID
        :return: (num_candidates,) slice of numpy.memmap candidate passage IDs corresponding to `qid`
        """
        ind = self.qid2ind[qid]
        return self.doc_ids[ind, :self.lengths[ind]]


def load_query_tokenids(query_tokenids_path):
    """
    :param query_tokenids_path: path to JSON file of {int qid: tokenized and numerized query} pairs
    :return: queries: dict {int qid: list of query token IDs (unpadded, no special tokens)}
    """
    queries = dict()
    with open(query_tokenids_path) as f:
        for line in tqdm(f, desc="Queries"):
            data = json.loads(line)
            queries[int(data['id'])] = data['ids']  # already tokenized and numerized!
    return queries


# only used by RepBERT
def load_querydoc_pairs(msmarco_dir, mode):
    """
    Has 2 separate modes with very different behavior
    :param msmarco_dir: dir containing qidpidtriples.train.small.tsv, qrels.train.tsv, top1000.{mode}
    :param mode:
        if "train", reads train triples file, then qrels.train file (TODO: why here?),
        and returns a qrels dict: {qID : set of relevant pIDs}, and aligned lists of qids, pids, labels (0 or 1).
        Within these 3 lists, each triple is broken into 2  successive elements:
        qids = [..., qid1, qid1, ...], pids = [..., pid1_pos, pid1_neg, ...], labels = [..., 1, 0, ...]
        if "dev", returns aligned qids, pids lists of (randomly ranked) MSMARCO top1000.dev BM25 results read from top1000.dev file
    """
    qrels = defaultdict(set)  # dict: {qID : set of relevant pIDs}
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open(f"{msmarco_dir}/qidpidtriples.train.small.tsv"),
                         # TODO: this file does not exist. RepBERT team made it
                         desc="load train triples"):
            qid, pos_pid, neg_pid = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid), int(neg_pid)
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)
        for line in open(f"{msmarco_dir}/qrels.train.tsv"):
            qid, _, pid, _ = line.split()  # the _ account for uninformative placeholders in TREC format
            qrels[int(qid)].add(int(pid))
    else:
        for line in open(f"{msmarco_dir}/top1000.{mode}"):
            qid, pid, _, _ = line.split("\t")  # the _ account for query and passage text (not used)
            qids.append(int(qid))
            pids.append(int(pid))
    qrels = dict(qrels)  # TODO: why need to convert to simple dict?! especially here and not within previous `if` !?
    if not mode == "train":
        labels, qrels = None, None  # qrels and labels NOT loaded for either "dev" or "eval"
    # each query ID is contained 2 consecutive times in qids, once corresponding to the positive and once to the negative pid
    return qids, pids, labels, qrels


def load_qrels(filepath):
    """Load ground truth relevant passages from file. Can handle several levels of relevance.
    Assumes that if a passage is not listed for a query, it is non-relevant.
    :param filepath: path to file of ground truth relevant passages in the following format:
        "qID1 \t Q0 \t pID1 \t 2\n qID1 \t Q0 \t pID2 \t 0\n qID1 \t Q0 \t pID3 \t 1\n..."
    :return:
        qid2relevance (dict): dictionary mapping from query_id (int) to relevant passages (dict {passageid : relevance})
    """
    qid2relevance = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            try:
                qid, _, pid, relevance = line.strip().split()
                if relevance:  # only if label is not zero
                    qid2relevance[int(qid)][int(pid)] = float(relevance)
            except Exception as x:
                print(x)
                raise IOError("'{}' is not valid format".format(line))
    return qid2relevance


# Not used because of REALLY SLOW access by pandas multi-row indexing when the number of queries is large
# def load_candidates_pandas(path_to_candidates):
#     """
#     Load candidate (retrieved) documents/passages from a file.
#     Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
#     columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).
#     :param path_to_candidates: path to file of candidate (retrieved) documents/passages per query
#     :return: pandas dataframe of candidate passage IDs indexed by qID. Multiple rows correspond to the same qID, each row is a pID
#     """
#     # internally uses C for parsing, and multiple chunks
#     candidates_df = pd.read_csv(path_to_candidates, delimiter='\t', header=None, index_col=0, memory_map=True, dtype=np.int32)
#
#     return candidates_df.iloc[:, 0]  # select only 1st column (pIDs), ignoring ranking and scores


class MYMARCO_Dataset(Dataset):
    """
    Used for passages (the terms passage/document used interchangeably in the documentation).
    Only considers queries existing in file of retrieved candidate passages for each query!
    Requires:
        1. memmap array of doc embeddings and an accompanying memmap array of doc/passage IDs, precomputed by
            precompute.py *on the entire collection of passages*.
        2. JSON file of {int qid: tokenized and numerized query} pairs, produced by convert_text_to_tokenized.py
        3. memmap array of candidate doc IDs and an accompanying memmap array of query IDs, produced by create_memmaps.py.
            holds candidate (retrieved) documents/passages per query. This can be produced by e.g. Anserini
        4. qrels file of ground truth relevant passages (if used for training or validation)
    """
    def __init__(self, mode,
                 embedding_memmap_dir, queries_tokenids_path, candidates_path, qrels_path=None,
                 tokenizer=None, max_query_length=64, num_candidates=None, candidate_sampling=None,
                 limit_size=None, emb_collection=None, load_collection_to_memory=False, inject_ground_truth=False,
                 collection_neutrality_path=None):
        """
        :param mode: 'train', 'dev' or 'eval'
        :param embedding_memmap_dir: directory containing (num_docs_in_collection, doc_emb_dim) memmap array of doc
            embeddings and an accompanying (num_docs_in_collection,) memmap array of corresponding doc/passage IDs
        :param queries_tokenids_path: path to dir or JSON file of {qid: list of token IDs} pairs
        :param candidates_path: directory containing memmap file of passage/doc IDs per query, and the
            corresponding query IDs, and the number of candidates per query in two more memmpap files
        :param qrels_path: dir or path to file of ground truth relevant passages in the following format:
            "qID1 \t Q0 \t pID1 \t 1\n qID1 \t Q0 \t pID2 \t 1\n ..."
        :param tokenizer: HuggingFace Tokenizer object. Must be the same as the one used for pre-tokenizing queries.
        :param max_query_length: max. number of query tokens, excluding special tokens
        :param num_candidates: number of document IDs to sample from all document IDs corresponding to a query and found
            in `candidates_path` file. If None, all found document IDs will be used
        :param candidate_sampling: method to use for sampling candidates. If None, the top `num_candidates` will be used
        :param limit_size: If set, limit dataset size to a smaller subset, e.g. for debugging. If in [0,1], it will
            be interpreted as a proportion of the dataset, otherwise as an integer absolute number of samples.
        :param load_collection_to_memory: If true, will load entire embedding array as np.array to memory, instead of memmap!
            Needs ~26GB for MSMARCO (~50GB project total), but is faster.
        :param emb_collection: (optional) already initialized EmbeddedCollection object. Will be used instead of
            reading matrix from `embedding_memmap_dir`
        :param inject_ground_truth: if True, during evaluation the ground truth relevant documents will be always included in the
            candidate documents, even if they weren't part of the original candidates. (Helps isolate effect of first-stage recall)
        """
        self.mode = mode  # "train", "dev", "eval"
        self.documents_neutrality = None
        if collection_neutrality_path is not None:
            logger.info("Loading document neutrality scores ...")
            self.documents_neutrality = {}
            for l in open(collection_neutrality_path):
                vals = l.strip().split('\t')
                self.documents_neutrality[int(vals[0])] = float(vals[1])    
        self.inject_ground_truth = inject_ground_truth  # if True, during evaluation the ground truth relevant documents will be always part of the candidate documents
        if self.inject_ground_truth:
            if self.mode == 'dev':
                logger.warning("Will include ground truth document(s) in candidates for each query during evaluation!")
            elif self.mode == 'eval':
                raise ValueError("It is not possible to use args.inject_ground_truth in 'eval' mode")
        if emb_collection is not None:
            self.emb_collection = emb_collection
        else:
            logger.info("Opening collection document embeddings memmap in '{}' ...".format(embedding_memmap_dir))
            self.emb_collection = EmbeddedCollection(embedding_memmap_dir, sorted_nat_ids=True,
                                                     load_to_memory=load_collection_to_memory)
        logger.info("Size of emb. collection:"
                    " {} MB".format(round(sys.getsizeof(self.emb_collection.embedding_vectors) / 1024**2)))
        if os.path.isdir(queries_tokenids_path):
            queries_tokenids_path = os.path.join(queries_tokenids_path, "queries.{}.json".format(mode))
        start_time = time.time()
        logger.info("Loading tokenized queries in '{}' ...".format(queries_tokenids_path))
        self.queries = load_query_tokenids(queries_tokenids_path)  # dict: {qID : list of token IDs}
        logger.info("Done in {:.3f} sec".format(time.time() - start_time))

        start_time = time.time()
        logger.info("Opening retrieved candidates for queries memmap in '{}' ...".format(candidates_path))
        self.candidates = CandidatesDataset(candidates_path, load_to_memory=True)
        logger.info("Done in {:.3f} sec".format(time.time() - start_time))
        logger.info("Size of candidates: {} MB".format(round(sys.getsizeof(self.candidates.doc_ids)/1024**2)))
        self.qids = self.candidates.qids  # memmap array of qIDs as found in retrieved candidates file
        self.limit_dataset_size(limit_size)  # Potentially changes candidates, qids

        if mode != 'eval':
            if os.path.isdir(qrels_path):
                qrels_path = os.path.join(qrels_path, "qrels.{}.tsv".format(mode))
            logger.info("Loading ground truth documents (labels) in '{}' ...".format(qrels_path))
            self.qrels = load_qrels(qrels_path)  # dict{qID: dict{pID: relevance}}
        else:
            self.qrels = None

        if tokenizer is None:  # TODO: for backwards compatibility. Consider removing
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.max_query_length = max_query_length

        self.num_candidates = num_candidates
        self.candidate_sampling = candidate_sampling

    def limit_dataset_size(self, limit_size):
        """Changes dataset to a smaller subset, e.g. for debugging"""
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.qids))
            if limit_size < len(self.qids):
                self.qids = self.qids[:limit_size]
                self.candidates.doc_ids = self.candidates.doc_ids[:limit_size, :]
                # self.candidates.lengths = self.candidates.lengths[:limit_size]
                # self.candidates.get_qid_to_ind_mapping(self.qids)
        return

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, ind):
        """
        For a given integer index corresponding to a single sample (query), returns a tuple of model input data
        :param ind: integer index of sample (query) in dataset
        :return:
            qid: (int) query ID
            query_token_ids: list of token IDs corresponding to query (unpadded, includes start/stop tokens)
            doc_ids: iterable of candidate document/passage IDs in order of relevance
            doc_embeddings: (len(doc_ids), emb_dim) slice of numpy.memmap embedding vectors corresponding to `doc_ids`
            rel_docs: set of ground truth relevant passage IDs corresponding to query ID; None if mode == 'eval'
        """
        global retrieve_candidates_times, prep_docids_times, lookup_times, sample_fetching_times

        sample_fetching_start = time.perf_counter()
        qid = self.qids[ind]
        query_token_ids = self.queries[qid]
        query_token_ids = query_token_ids[:self.max_query_length]
        query_token_ids = [self.cls_id] + query_token_ids + [self.sep_id]

        tic = time.perf_counter()
        doc_ids = self.candidates[qid]  # iterable of candidate document/passage IDs in order of relevance
        retrieve_candidates_times.update(time.perf_counter() - tic)
        doc_ids = self.sample_candidates(doc_ids)  # sampled subset of candidate doc_ids

        if self.mode == "eval":
            rel_docs = None
        else:
            tic = time.perf_counter()
            rel_docs = self.qrels[qid].keys()
            if self.mode == "train" or self.inject_ground_truth:
                # prepend relevant documents at the beginning of doc_ids, whether pre-existing in doc_ids or not,
                # while ensuring that they are only included once
                num_candidates = len(doc_ids)
                new_doc_ids = (list(rel_docs) + [docid for docid in doc_ids if docid not in rel_docs])[:num_candidates]
                doc_ids = new_doc_ids  # direct assignment wouldn't work in line above
            prep_docids_times.update(time.perf_counter() - tic)


        tic = time.perf_counter()
        doc_embeddings = torch.tensor(self.emb_collection[doc_ids])  # (num_doc_ids, emb_dim) tensor of doc. embeddings
        lookup_times.update(time.perf_counter() - tic)
        sample_fetching_times.update(time.perf_counter() - sample_fetching_start)
        return qid, query_token_ids, doc_ids, doc_embeddings, rel_docs

    def sample_candidates(self, candidates):
        """
        Sample `self.num_candidates` from all retrieved candidate document IDs corersponding to a query,
        according to method `self.candidate_sampling`
        :param candidates: iterable of candidate document/passage IDs (corresponding to a query) in order of relevance
        :return: list of len(self.num_candidates) subset of `candidates`
        """
        if self.num_candidates:
            return candidates[:self.num_candidates]
        # TODO: implement more sampling methods (e.g. for multi-tiered relevance loss)
        else:
            return candidates

    def get_collate_func(self, num_inbatch_neg=0, max_candidates=MAX_DOCS, n_gpu=1):
        """
        :param num_inbatch_neg: number of negatives to randomly sample from other queries in the batch.
            Can only be > 0 if mode == 'train'
        :param max_candidates: maximum number of candidates per query to be scored by the model (capacity of model)
            `num_candidates` will be reduced accordingly, if necessary.
        :return: function with a single argument, which corresponds to a list of individual sample data. Used by DataLoader
        """
        if self.mode != 'train':
            num_inbatch_neg = 0

        return partial(collate_function, mode=self.mode, pad_token_id=self.pad_id, num_inbatch_neg=num_inbatch_neg,
                       max_candidates=max_candidates, n_gpu=n_gpu, inject_ground_truth=self.inject_ground_truth,
                       documents_neutrality=self.documents_neutrality)


# TODO: can I pass the np.memmap instead of a list of doc_emb arrays (inside batch)? This would replace `assembled_emb_mat` and `docID_to_localind`
def collate_function(batch_samples, mode, pad_token_id, num_inbatch_neg=0, max_candidates=MAX_DOCS, n_gpu=1,
                     inject_ground_truth=False, documents_neutrality=None):
    """
    :param batch_samples: (batch_size) list of tuples (qids, query_token_ids, doc_ids, doc_embeddings, rel_docs)
    :param mode: 'train', 'dev', or 'eval'
    :param pad_token_id: ID of token used for padding queries
    :param num_inbatch_neg: number of negatives to randomly sample from other queries in the batch.
        Can only be > 0 if mode == 'train'
    :param max_candidates: maximum number of candidates per query to be scored by the model (capacity of model)
    :return:
        qids: (batch_size) list of query IDs
        doc_ids: (batch_size) list of lists of candidate (retrieved) document IDs corresponding to a query in `qids`.
            If num_inbatch_neg > 0, it includes the docIDs of randomly sampled in-batch negatives
        data: dict to serve as input to the model. Contains:
            query_token_ids: (batch_size, max_query_len) int tensor of query token IDs
            query_mask: (batch_size, max_query_len) bool tensor of padding mask corresponding to query; 1 use, 0 ignore
            doc_padding_mask: (batch_size, max_docs_per_query) boolean tensor indicating padding (0) or valid (1) elements
                in `doc_emb` or `docinds`
            If num_inbatch_neg > 0, additionally contains:
            local_emb_mat: (num_unique_docIDs, emb_dim) tensor of local doc embedding matrix containing emb. vectors
                of all unique documents in the batch.  Used to lookup document vectors in nn.Embedding on the GPU
                This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth.
            docinds: (batch_size, max_docs_per_query) local indices of documents corresponding to `local_emb_mat`
            else:
            doc_emb: (batch_size, max_docs_per_query, emb_dim) float tensor of document embeddings corresponding
                to the pool of candidates for each query

            If mode != 'eval', additionally contains:
            labels: (batch_size, max_docs_per_query) int tensor which for each query (row) contains the indices of the
                relevant documents within its corresponding pool of candidates, `doc_ids`. Padded with -1.
    """
    start_collation_time = time.perf_counter()
    batch_size = len(batch_samples)

    qids, query_token_ids, doc_ids, doc_embeddings, rel_docs = zip(*batch_samples)
    query_lengths = [len(seq) for seq in query_token_ids]
    max_query_length = max(query_lengths)
    query_masks = [[1] * ql for ql in query_lengths]  # 1 use, 0 ignore

    # The 2D tensors are padded NOT to a standard length (transformer input length), but to the query max_len in the batch
    data = {"query_token_ids": pack_tensor_2D(query_token_ids, default=pad_token_id, length=max_query_length, dtype=torch.int32),
            "query_mask": pack_tensor_2D(query_masks, default=0, length=max_query_length, dtype=torch.bool)}

    if num_inbatch_neg:  # only in 'train' mode
        # In this case, the doc embeddings are not packed here, but inside the model (on the GPU), in order
        # to avoid transferring replicas of document embeddings corresponding to in-batch negatives and thus spare GPU mem. bandwidth.
        # For this purpose, we pass a smaller local doc embedding matrix containing emb. vectors of all documents
        # in the batch and a dictionary mapping docID to local indices corresponding to this matrix

        assembled_emb_mat = torch.cat(doc_embeddings, dim=0)  # (batch_size*num_docs_per_query, emb_dim)

        # maps unique docIDs to rows of local embedding matrix
        docID_to_localind = OrderedDict((ID, i) for i, ID in enumerate(chain.from_iterable(doc_ids)))
        # get all unique document IDs in the batch. This is an object akin to a set, but keeping the order
        unique_candidates = docID_to_localind.keys()  # all unique document IDs in the batch

        # augment doc_ids with randomly sampled document IDs from candidates retrieved for other qIDs in the batch
        doc_ids = [(cands + list(
            np.random.choice(list(unique_candidates - set(cands)), size=num_inbatch_neg, replace=False)))[:max_candidates]
                   for cands in doc_ids]

        # local doc embedding matrix. It is only slightly (if at all) smaller than `assembled_emb_mat`, because the chance of
        # duplicate docIDs in the original (unaugmented) `doc_ids` is very small #TODO: so its creation can be omitted
        local_emb_mat = torch.zeros(len(unique_candidates) + 1,  # we make 1 extra row at the end, for padding!
                                    doc_embeddings[0].shape[-1])  # (num_unique_docs, emb_dim)
        for i, docid in enumerate(unique_candidates):
            local_emb_mat[i, :] = assembled_emb_mat[docID_to_localind[docid], :]
            docID_to_localind[docid] = i  # re-assign mapping to smaller local_emb_mat (from assembled_emb_mat)

        data['docinds'], data['doc_padding_mask'] = prepare_docinds_and_mask(doc_ids, docID_to_localind,
                                                                             padding_idx=local_emb_mat.shape[0]-1)
        data['local_emb_mat'] = local_emb_mat.repeat(n_gpu, 1)  # replicas passed to each split of multi-gpu

    max_docs_per_query = min(max_candidates, max(len(cands) for cands in doc_ids))  # length used for padding

    if num_inbatch_neg == 0:
        # Pack 3D tensors: doc embeddings and corresponding padding mask
        doc_emb_tensor = torch.zeros(batch_size, max_docs_per_query, doc_embeddings[0].shape[-1])  # (batch_size, padded_length, emb_dim)
        doc_emb_mask = torch.zeros(doc_emb_tensor.shape[:2], dtype=torch.bool)  # (batch_size, padded_length)
        for i, doc_embs in enumerate(doc_embeddings):
            end = min(doc_embs.shape[0], max_docs_per_query)
            doc_emb_tensor[i, :end, :] = doc_embs[:end, :]
            doc_emb_mask[i, :end] = 1
        data['doc_emb'] = doc_emb_tensor
        data['doc_padding_mask'] = doc_emb_mask

    if documents_neutrality is not None:
        doc_neutscores = []
        for doc_ids_batch in doc_ids:
            doc_neutscores_batch = []
            for doc_id in doc_ids_batch:
                if doc_id in documents_neutrality:
                    doc_neutscores_batch.append(documents_neutrality[doc_id])    
                else:
                    logger.debug("Document neutrality score of ID %d is not found (set to 1)" % doc_id)
                    doc_neutscores_batch.append(1.0)
            doc_neutscores.append(doc_neutscores_batch)

        doc_neutscores_tensor = torch.zeros(batch_size, max_docs_per_query)  # (batch_size, padded_length)
        for i, doc_neutscores_batch in enumerate(doc_neutscores):
            end = min(len(doc_neutscores_batch), max_docs_per_query)
            doc_neutscores_tensor[i, :end] = torch.tensor(doc_neutscores_batch)[:end]
        data['doc_neutscore'] = doc_neutscores_tensor
        
    if mode != 'eval':
        if mode == "train" or inject_ground_truth:
            # `labels` holds for each query the indices of the relevant doc IDs within its corresponding pool of candidates, `doc_ids`
            # In this case, it is guaranteed by the construction of doc_ids in MYMARCO_Dataset.__get_item__
            # (and the fact that we are only randomly sampling negatives not already in the list of candidates for a given qID)
            # that 1 or more positives (rel. docs) will be at the beginning of the list (and only there)
            labels = [list(range(len(rd))) for rd in rel_docs]
        else:  # this is when 'dev', but not inject_ground_truth
            labels = []
            for i in range(batch_size):
                query_labels = []
                for ind in range(len(doc_ids[i])):
                    if doc_ids[i][ind] in rel_docs[i]:
                        query_labels.append(ind)
                labels.append(query_labels)

        # must be padded with -1 to have same dimensions as the transformer decoder output: (batch_size, max_docs_per_query)
        data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int16, length=max_docs_per_query)  # no more than a couple of rel. documents per query exist, so even int8 could be used



    global collation_times
    collation_times.update(time.perf_counter() - start_collation_time)

    return data, qids, doc_ids


def prepare_docinds_and_mask(doc_ids, docID_to_localind, padding_idx, length=None):
    """
    :param doc_ids: (batch_size) list of: (num_docs) list of document IDs corresponding to some query
    :param docID_to_localind: dict mapping from docIDs to local indices corresponding to loc. emb. matrix
    :param padding_idx: special padding index: the last row of loc. emb. matrix. Used to lookup padding in nn.Embedding
    :param length: fixed length to be used for batch
    :return:
        docinds: (batch_size, padded_num_docs) tensor of local indices of documents corresponding to local emb. matrix
            used to lookup document vectors in nn.Embedding on the GPU
        docinds_mask: (batch_size, padded_num_docs) bool mask tensor corresponding to `docinds`; 1 valid, 0 ignore
    """
    batch_size = len(doc_ids)
    length = length if length is not None else max(len(docids) for docids in doc_ids)
    docinds = torch.full((batch_size, length), padding_idx, dtype=torch.int32)  # although docinds are local indices, int32 is used because the number of documents in the batch can grow large: cand_per_query*batch_size
    docinds_mask = torch.zeros_like(docinds, dtype=torch.bool)
    for i, docids in enumerate(doc_ids):
        local_docinds = [docID_to_localind[docid] for docid in docids]
        docinds[i, :len(docids)] = torch.tensor(local_docinds, dtype=torch.int32)  # casting to tensor required for assignment
        docinds_mask[i, :len(docids)] = True
    return docinds, docinds_mask


# Not used! We use nn.Embedding instead! ~10 times faster!
# def assemble_3D_tensors(doc_ids, local_emb_mat, docID_to_localind):
#     """Pack 3D tensors for doc embeddings and corresponding padding mask
#     :param doc_ids: (batch_size) list of candidate (retrieved) document IDs (int list) corresponding to a query in `qids`
#         Includes the docIDs of randomly sampled in-batch negatives
#     :param local_emb_mat: (num_unique_docIDs, emb_dim) tensor of local doc embedding matrix containing emb. vectors
#             of all unique documents in the batch
#     :param docID_to_localind: OrderedDict mapping unique docIDs to rows of `local_emb_mat`
#     :return:
#         doc_emb_tensor: (max_docs_per_query, batch_size, emb_dim) document embeddings tensor, input to "decoder"
#         doc_emb_mask: (batch_size, max_docs_per_query) boolean tensor indicating padding (0) or valid (1) elements
#     """
#
#     batch_size = len(doc_ids)
#     max_docs_per_query = max(len(cands) for cands in doc_ids)  # length used for padding
#
#     doc_emb_tensor = torch.zeros(max_docs_per_query, batch_size, local_emb_mat.shape[1])  # (padded_length, batch_size, emb_dim)
#     doc_emb_mask = torch.zeros((batch_size, max_docs_per_query), dtype=torch.bool)  # (batch_size, padded_length)
#     for i in range(batch_size):
#         num_docs = len(doc_ids[i])
#         doc_emb_mask[i, :num_docs] = 1
#         for j in range(num_docs):
#             doc_emb_tensor[j, i, :] = local_emb_mat[docID_to_localind[j], :]
#     return doc_emb_tensor, doc_emb_mask


# TODO: only used by RepBERT! (consider removing)
class MSMARCODataset(Dataset):
    def __init__(self, mode, msmarco_dir,
                 collection_memmap_dir, queries_tokenids_path,
                 max_query_length=20, max_doc_length=256, limit_size=None):
        """
        :param limit_size: If set, limit dataset size to a smaller subset, e.g. for debugging. If in [0,1], it will
            be interpreted as a proportion of the dataset, otherwise as an integer absolute number of samples.
        """
        self.collection = CollectionDataset(collection_memmap_dir)
        if os.path.isdir(queries_tokenids_path):
            queries_tokenids_path = os.path.join(queries_tokenids_path, "queries.{}.json".format(mode))
        self.queries = load_query_tokenids(queries_tokenids_path)  # dict: {qID : list of token IDs}
        # qids, pids, labels: corresponding lists of query IDs, passage IDs, 1 / 0 relevance labels
        # each query ID is contained 2 consecutive times in qids, once corresponding to the positive and once to the negative pid
        # qrels is the ground truth dict: {qID : set of relevant pIDs}
        self.qids, self.pids, self.labels, self.qrels = load_querydoc_pairs(msmarco_dir, mode)
        self.mode = mode  # "train", "dev", "eval"
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.limit_dataset_size(limit_size)

    def limit_dataset_size(self, limit_size):
        """Changes dataset to a smaller subset, e.g. for debugging"""
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.qids))
            if limit_size < len(self.qids):
                self.qids = self.qids[:limit_size]
                self.pids = self.pids[:limit_size]
                self.labels = self.labels[:limit_size]
                # self.candidates.lengths = self.candidates.lengths[:limit_size]
                # self.candidates.get_qid_to_ind_mapping(self.qids)
        return

    def get_collate_function(self, n_gpu=1):  # TODO n_gpu is not handled in the function
        def collate_function(batch):
            input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
            token_type_ids_lst = [[0] * len(x["query_input_ids"]) + [1] * len(x["doc_input_ids"]) for x in batch]
            valid_mask_lst = [[1] * len(input_ids) for input_ids in input_ids_lst]
            position_ids_lst = [list(range(len(x["query_input_ids"]))) +
                                list(range(len(x["doc_input_ids"]))) for x in batch]

            # The 2D tensors are padded NOT to a standard length (transformer seq. length), but to the max_len in the batch!
            data = {
                "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
                # int64 is required by torch nn.Embedding :(
                "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
                "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
                "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
            }
            qid_lst = [x['qid'] for x in batch]  # TODO: this can be avoided by returning qid, docid as part of a tuple in __getitem__
            docid_lst = [x['docid'] for x in batch]
            if self.mode == "train":
                # labels holds the in-batch indices of the relevant doc IDs for each query
                labels = [[j for j in range(len(docid_lst)) if docid_lst[j] in x['rel_docs']] for x in batch]
                data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int64, length=len(batch))
            return data, qid_lst, docid_lst

        return collate_function

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = [self.cls_id] + doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid": pid
        }
        if self.mode == "train":
            ret_val["rel_docs"] = self.qrels[qid]
        return ret_val


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = torch.full((batch_size, length), default, dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)  # casting to tensor required for assignment
    return tensor
