import os
import json
import logging
from collections import defaultdict, OrderedDict
from functools import partial
from itertools import chain
from operator import itemgetter
import time
import sys
import pickle
import ipdb

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

import utils

logger = logging.getLogger(__name__)

BERT_BASE_DIM = 768
# NOTE: MAX_DOCS should depend on the GPU memory, but maybe it's better to crash
# rather than silently scoring fewer documents than the user specified
MAX_DOCS = int(1e12)  # 1000

lookup_times = utils.Timer()  # stores measured times
sample_fetching_times = utils.Timer()
collation_times = utils.Timer()
prep_docids_times = utils.Timer()
retrieve_candidates_times = utils.Timer()


def nanstd(x, dim=-1):
    """Returns std of Tensor x along specified dimension, ignoring NaN values""" 
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=dim).unsqueeze(dim), 2), dim=dim))


def normalize_batch(relevances, norm_type):
    """
    Normalize relevances for each query in a batch, ignoring padding (assumed to be -inf).
    Also works for a (num_candidates,) vector of relevances with respect to a single query.
    Normalization is supposed to help dealing with the variety of relevance score ranges resulting from different hyperparameter combinations.
    Ultimately, a KL divergence between the model-predicted score distribution (possibly tempered by temperature)
    and the target relevance distribution will be computed, which will be obtained by applying a softmax on relevances computed here.

    :param relevances: (batch_size, num_candidates) tensor of target relevances. Also supports (num_candidates,)
    :param norm_type: str. One of 'max', 'maxmin', 'std'
    :return: (batch_size, num_candidates) tensor of normalized target relevances
    """
    
    if norm_type == 'max':  # in [0, 1]. May result in fairly flat "distribution" close to 1.
        # max is not affected by -inf values
        return relevances / torch.max(relevances, dim=-1).values.unsqueeze(-1)
    
    not_valid = (relevances == float('-inf'))  # -inf is used as padding
    if norm_type.startswith('maxmin'):  # increases inter-document dynamic range for "uniform" initial scores, while still in [0, 1]
        # max is not affected by -inf values
        max_relev = torch.max(relevances, dim=-1).values
        # to ignore -inf values, we first make a copy of new_relevances where we replace non-valid values with +inf, and then we take the min along each row
        min_relev = torch.min(torch.where(not_valid, torch.full_like(relevances, float('inf')), relevances), dim=-1).values
        relevances = (relevances - min_relev.unsqueeze(-1)) / (max_relev - min_relev).unsqueeze(-1)
        if norm_type == 'maxminmax':  # even wider dynamic range for "uniform" initial scores, but in [0, mean(row)]
            relevances = max_relev.unsqueeze(-1) * relevances
        elif norm_type == 'maxminbatchmean':  # wide dynamic range for "uniform" initial scores, while reducing variance across queries; in [0, mean(batch)]
            relevances = torch.nanmean(torch.where(not_valid, torch.full_like(relevances, float('nan')), relevances)) * relevances
    elif norm_type == 'std': # even wider dynamic range for "uniform" initial scores, but in [0, f], with f > 1
        # to ignore -inf values for std, we follow the same approach as with min, but use NaN. There is a special torch function ignoring NaN
        min_relev = torch.min(torch.where(not_valid, torch.full_like(relevances, float('inf')), relevances), dim=-1).values
        relev_std = nanstd(torch.where(not_valid, torch.full_like(relevances, float('nan')), relevances), dim=-1).unsqueeze(-1)
        relevances = (relevances - min_relev.unsqueeze(-1)) / relev_std
    else:
        raise ValueError(f"Unknown relevance score normalization option '{norm_type}'")
    return relevances


# only used by RepBERT and `precompute.py`
class CollectionDataset:
    """Document / passage collection in form of integer token IDs"""

    def __init__(self, collection_memmap_dir, max_doc_length=None):
        """
        :param collection_memmap_dir: directory containing memmap files of token IDs and lengths for each document ID
        :param max_doc_length: if provided, it will force reading the memmap with specified dimensions, rather than
            inferring size. This can be useful in the unfortunate case a memmap has been created and overwritten with different sizes
        """

        self.pids = np.memmap(os.path.join(collection_memmap_dir, 'pids.memmap'), mode='r', dtype='int32')
        self.lengths = np.memmap(os.path.join(collection_memmap_dir, "lengths.memmap"), mode='r', dtype='int32')
        self.collection_size = len(self.pids)
        if max_doc_length:  # force certain array dimensions
            self.token_ids = np.memmap(os.path.join(collection_memmap_dir, 'token_ids.memmap'),
                                       mode='r', dtype='int32', shape=(self.collection_size, max_doc_length))
        else:
            self.token_ids = np.memmap(os.path.join(collection_memmap_dir, 'token_ids.memmap'), mode='r', dtype='int32')
            self.token_ids = self.token_ids.reshape((self.collection_size, -1))
        logger.info("Tokenized doc collection memmap array shape: {}, type: {}".format(self.token_ids.shape, type(self.token_ids)))

        self.sorted_nat_ids = np.array_equal(self.pids, np.array(range(self.collection_size), dtype=int))  # check whether IDs are the sequence 0, 1, ...
        if self.sorted_nat_ids:  # this is True for MSMARCO
            self.id2ind = None
            self.map_id_to_ind = lambda x: x
        else:
            self.id2ind = OrderedDict((ID, i) for i, ID in enumerate(self.pids))
            self.map_id_to_ind = self.id2ind.get  # ID-to-matrix_index mapping function

    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):
        """
        :param item: integer index of passage/doc in collection
        :return: list (unpadded) of token IDs corresponding to ind
        """
        ind = self.map_id_to_ind(item)  # convert ID/index to ordinal index
        return self.token_ids[ind, :self.lengths[ind]].tolist()


class EmbeddedSequences:
    """Document / passage collection or queries in the form of embeddings memmap"""

    def __init__(self, embedding_memmap_dir, load_to_memory=False, seq_type='document'):
        """
        :param embedding_memmap_dir: directory containing memmap file of precomputed document/query embeddings, and the
            corresponding passage/doc/queries IDs in another memmpap file
        :param load_to_memory: If true, will load entire embedding array as np.array to memory, instead of memmap!
            Needs ~26GB for MSMARCO document collection (~50GB project total), but is faster.
        :param seq_type: 'document' or 'query'. What type of sequences are contained in specified memmap
        REMOVED emb_dim: dimensionality of document/query vectors. If None, `embedding_memmap_dir` will be used to infer it,
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

        try:
            self.ids = np.memmap(os.path.join(embedding_memmap_dir, 'ids.memmap'), mode='r', dtype='int32')
        except FileNotFoundError:
            # due to legacy reasons, sometimes document memmaps have the name 'pids.memmap'
            self.ids = np.memmap(os.path.join(embedding_memmap_dir, 'pids.memmap'), mode='r', dtype='int32')
        # Check whether passage/doc IDs in the collection happen to exactly be 0, 1, ..., num_docs-1, and
        # doc embeddings are stored exactly in that order (is True for MSMARCO passage collection, but not for queries).
        # If so, overall a bit more efficient, but a lot more efficient when sampling (non-in-batch) random negative documents
        self.sorted_nat_ids = np.array_equal(self.ids, np.array(range(len(self.ids)), dtype=int))  # sorted natural integers as IDs
        self.map_id_to_ind = self.get_id_to_ind_mapping(self.ids)  # ID-to-matrix_index mapping function

        self.num_sequences = len(self.ids)  # number of rows of embedding matrix (i.e. number of sequences)
        # shape is necessary input argument; this information is not stored and a 1D array is loaded by default
        self.embedding_vectors = np.memmap(os.path.join(embedding_memmap_dir, "embedding.memmap"), mode='r', dtype='float32')
        self.embedding_vectors = self.embedding_vectors.reshape((self.num_sequences, -1))
        if load_to_memory:
            logger.warning("Loading all {} embeddings to memory!".format(seq_type))
            self.embedding_vectors = np.array(self.embedding_vectors)
        logger.info("{} embeddings array shape: {}, "
                    "type: {}".format(seq_type, self.embedding_vectors.shape, type(self.embedding_vectors)))
        return

    def get_id_to_ind_mapping(self, ids):
        """Returns function used to map a list of IDs to the corresponding integer indices of the embedding matrix"""
        if self.sorted_nat_ids:
            self.id2ind = None  # no mapping dictionary in case of sorted natural integers as IDs
            return lambda x: x
        else:
            self.id2ind = OrderedDict((ID, i) for i, ID in enumerate(ids))
            return lambda x: [self.id2ind[ID] for ID in x]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ids):
        """
        :param ids: iterable of some IDs of passages/docs in collection
        :return: (num_ids, emb_dim) slice of numpy.memmap embedding vectors corresponding to `ids`
        """
        inds = self.map_id_to_ind(ids)
        return self.embedding_vectors[inds, :]


class CandidatesDataset:
    """The result of a retrieval stage: rank-sorted document / passages per query, in the form of doc/passage IDs
    memmap array of shape (num_queries, num_candidates)"""

    def __init__(self, memmap_dir, max_docs=None, load_to_memory=False):
        """
        :param memmap_dir: directory containing memmap file of passage/doc IDs per query, and the
            corresponding query IDs, and the number of candidates per query in two more memmpap files
        :param max_docs: maximum number of candidates per query. Used to restrict the maximum number as it exists in the
            memmap file on disk.
        :param load_to_memory: If true, will load entire candidates array as np.array to memory, instead of memmap!
            Needs ~2GB for MSMARCO training set (212MB for dev set) with 1000 candidates per query
            (~XGB program total), but is faster.
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

# only used in Inspect mode
def load_original_sequences(seq_path):
    """
    For a specified collection or queries file, loads original sequence IDs and respective text into a dictionary. Used for 'inspect' mode.
    :param seq_path: path of the raw queries/collection file (each line: seqID \t text)
    :return: {seqID: sequence_text} dictionary
    """
    total_size = sum(1 for _ in open(seq_path))  # simply to get number of lines
    sequence_dict = {}
    with open(seq_path, 'r') as qFile:
        for line in tqdm(qFile, total=total_size, desc="Sequences: "):
            query_id, text = line.split("\t")
            sequence_dict[int(query_id)] = text
    return sequence_dict


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
# def load_querydoc_pairs(msmarco_dir, mode):
#     """
#     Has 2 separate modes with very different behavior
#     :param msmarco_dir: dir containing qidpidtriples.train.small.tsv, qrels.train.tsv, top1000.{mode}
#     :param mode:
#         if "train", reads train triples file, then qrels.train file (GEO: why here?),
#         and returns a qrels dict: {qID : set of relevant pIDs}, and aligned lists of qids, pids, labels (0 or 1).
#         Within these 3 lists, each triple is broken into 2  successive elements:
#         qids = [..., qid1, qid1, ...], pids = [..., pid1_pos, pid1_neg, ...], labels = [..., 1, 0, ...]
#         if "dev", returns aligned qids, pids lists of (randomly ranked) MSMARCO top1000.dev BM25 results read from top1000.dev file
#     """
#     qrels = defaultdict(set)  # dict: {qID : set of relevant pIDs}
#     qids, pids, labels = [], [], []
#     if mode == "train":  # GEO: this triples file does not exist. RepBERT team made it
#         for line in tqdm(open(f"{msmarco_dir}/qidpidtriples.train.small.tsv"), desc="load train triples"):
#             qid, pos_pid, neg_pid = line.split("\t")
#             qid, pos_pid, neg_pid = int(qid), int(pos_pid), int(neg_pid)
#             qids.append(qid)
#             pids.append(pos_pid)
#             labels.append(1)
#             qids.append(qid)
#             pids.append(neg_pid)
#             labels.append(0)
#         for line in open(f"{msmarco_dir}/qrels.train.tsv"):
#             qid, _, pid, _ = line.split()  # the _ account for uninformative placeholders in TREC format
#             qrels[int(qid)].add(int(pid))
#     else:
#         for line in open(f"{msmarco_dir}/top1000.{mode}"):
#             qid, pid, _, _ = line.split("\t")  # the _ account for query and passage text (not used)
#             qids.append(int(qid))
#             pids.append(int(pid))
#     qrels = dict(qrels)  # GEO: why need to convert to simple dict?! especially here and not within previous `if` !?
#     if not mode == "train":
#         labels, qrels = None, None  # qrels and labels NOT loaded for either "dev" or "eval"
#     # each query ID is contained 2 consecutive times in qids, once corresponding to the positive and once to the negative pid
#     return qids, pids, labels, qrels


def load_qrels(filepath, relevance_thr=1, score_mapping=None):
    """Load ground truth relevant passages from file. Can handle several levels of relevance.
    Assumes that if a passage is not listed for a query, it is non-relevant.
    :param filepath: path to file of ground truth relevant passages in the following format:
        "qID1 \t Q0 \t pID1 \t 2\n
         qID1 \t Q0 \t pID2 \t 0\n
         qID1 \t Q0 \t pID3 \t 1\n..."
    :param relevance_thr: only include candidates which have at least the specified relevance threshold score
        (after potential mapping)
    :param score_mapping: dictionary mapping relevance scores in qrels file to a different value (e.g. 1 -> 0.03)
    :return:
        qid2relevance (dict): dictionary mapping from int query_id to int passage IDs with g.t. relevance judgement (dict {passageid : relevance})
    """
    qid2relevance = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            try:
                qid, _, pid, relevance = line.strip().split()
                relevance = float(relevance)
                if (score_mapping is not None) and (relevance in score_mapping):
                    relevance = score_mapping[relevance]  # map score to new value
                if relevance >= relevance_thr:  # include only if score >= specified relevance threshold level
                    qid2relevance[int(qid)][int(pid)] = relevance
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


def load_query_ids(filepath):
    """Loads query IDs from a file which contains 1 ID per line (and possibly more fields on its right, separated by whitespace)"""
    query_ids = []
    with open(filepath, 'r') as f:
        for line in f:
            qid, *_ = line.strip().split()  # read first field, ignore the rest, if any exist
            query_ids.append(int(qid))
    return query_ids


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
                 embedding_memmap_dir, queries_tokenids_path, candidates_path=None, 
                 qrels_path=None, target_scores_path=None, query_ids_path=None,
                 tokenizer=None, max_query_length=64,
                 num_candidates=None, candidate_sampling=None, dynamic_candidates=None,
                 limit_size=None, emb_collection=None, load_collection_to_memory=False, inject_ground_truth=False,
                 relevance_labels_mapping=None, include_at_level=1, relevant_at_level=1, max_inj_relevant=1000,
                 label_normalization=None,
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
        :param target_scores_path: Optional. Path to a file containing relevance scores which will be used as training labels
            (instead of qrels, which will be used only for evaluation).
        :param query_ids_path: Optional. Path to a file containing a subset of externally provided query IDs to be used in dataset.
        :param tokenizer: HuggingFace Tokenizer object. Must be the same as the one used for pre-tokenizing queries.
        :param max_query_length: max. number of query tokens, excluding special tokens
        :param num_candidates: number of document IDs to sample from all document IDs corresponding to a query and found
            in `candidates_path` file. If None, all found document IDs will be used.
        :param candidate_sampling: method to use for sampling candidates. If None, the top `num_candidates` will be used
        :param limit_size: If set, limit dataset size to a smaller subset, e.g. for debugging. If in [0,1], it will
            be interpreted as a proportion of the dataset, otherwise as an integer absolute number of samples.
        :param load_collection_to_memory: If true, will load entire embedding array as np.array to memory, instead of memmap!
            Needs ~26GB for MSMARCO (~50GB project total), but is faster.
        :param emb_collection: (optional) already initialized EmbeddedSequences object. Will be used instead of
            reading matrix from `embedding_memmap_dir`
        :param inject_ground_truth: if True, during evaluation the ground truth relevant documents will be always included in the
            candidate documents, even if they weren't part of the original candidates. (Helps isolate effect of first-stage recall)
        :param relevance_labels_mapping: dict mapping from relevance scores inside `qrels_path` to new (overriding) values
        :param include_at_level: candidate with a score >= this level will be included in qrels and injected candidates
        :param relevant_at_level: candidate with a score >= this level will be included in injected candidates and
            keep their respective scores as target scores (if `label_format` is `scores`).
        :param max_inj_relevant: maximum number of 'relevant' candidates to inject per query when training
            (whether they come from qrels or target scores)
        :param collection_neutrality_path: path to file containing neutrality scores, in the format: docID \t score\n
        :param query_ids_path: Path to a file containing the query IDs to be used for this dataset, 1 in each line.
            If not provided, the IDs in the memmap inside `candidates_path` will be used.
            If that is also not provided, the IDs inside `queries_tokenids_path` will be used.
        """
        self.mode = mode  # "train", "dev", "eval"
        
        logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
        logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

        if target_scores_path is not None:
            if self.mode != 'train':
                raise ValueError("Continuous target scores are not supposed to used for evaluation, only for training")
            logger.info("Loading target scores (training labels) for documents from '{}' ...".format(target_scores_path))
            # target_scores: dict{qID: dict{pID: relevance}} continuous (float) label scores used for training
            if target_scores_path.endswith('.hdf5'):
                self.target_scores = utils.load_qrels_from_hdf5(target_scores_path, overwrite_queries=True) # string query/doc IDs
            elif target_scores_path.endswith('.pickle'):
                self.target_scores = utils.load_qrels_from_pickles(target_scores_path, overwrite_queries=True)
            else:
                self.target_scores = load_qrels(target_scores_path, relevance_thr=include_at_level)
            logger.info("Size of target scores: {} MB".format(int(round(sys.getsizeof(self.target_scores)/1024**2))))
            logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
            logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))
        
        else:
            self.target_scores = None

        self.label_normalization = label_normalization  # if not None, will normalize *training* labels (usually, self.target_scores) 

        self.inject_ground_truth = inject_ground_truth  # if True, during evaluation the ground truth relevant documents will be always part of the candidate documents
        if self.inject_ground_truth:
            if self.mode == 'dev':
                logger.warning("Will include ground truth document(s) in candidates for each query during evaluation!")
            elif self.mode == 'eval':
                raise ValueError("It is not possible to use args.inject_ground_truth in 'eval' mode (no labels exist)")
        if emb_collection is not None:
            self.emb_collection = emb_collection
        else:
            logger.info("Opening collection document embeddings memmap in '{}' ...".format(embedding_memmap_dir))
            self.emb_collection = EmbeddedSequences(embedding_memmap_dir, load_to_memory=load_collection_to_memory)
        logger.info("Size of emb. collection:"
                    " {} MB".format(round(sys.getsizeof(self.emb_collection.embedding_vectors) / 1024**2)))
        if os.path.isdir(queries_tokenids_path):
            queries_tokenids_path = os.path.join(queries_tokenids_path, "queries.{}.json".format(mode))
        start_time = time.time()
        logger.info("Loading tokenized queries in '{}' ...".format(queries_tokenids_path))
        self.queries = load_query_tokenids(queries_tokenids_path)  # dict: {qID : list of token IDs}
        logger.info("Done in {:.3f} sec".format(time.time() - start_time))

        if query_ids_path is None:  # typical case
            self.qids = None  # query IDs used for this dataset
        else:
            self.qids = load_query_ids(query_ids_path)  # query IDs used for this dataset
            logger.info("Loaded {} query IDs from: '{}'".format(len(self.qids), query_ids_path))

        self.dynamic_candidates = dynamic_candidates
        if candidates_path is not None:  # this is the typical case
            start_time = time.time()
            logger.info("Opening retrieved candidates for queries memmap in '{}' ...".format(candidates_path))
            self.candidates = CandidatesDataset(candidates_path, load_to_memory=True)
            logger.info("Done in {:.3f} sec".format(time.time() - start_time))
            logger.info("Size of candidates: {} MB".format(int(round(sys.getsizeof(self.candidates.doc_ids)/1024**2))))

            if self.qids is None:  # this is the typical case
                self.qids = self.candidates.qids  # assign memmap array of qIDs as found in retrieved candidates file
            else:
                # NOTE: the following is done for efficiency, but is probably unnecessary
                self.update_candidates(self.qids)

        else:
            # if no candidate documents are provided, they will be either dynamically retrieved
            # or randomly sampled from the collections
            self.candidates = None
            if self.qids is None:  # Since no candidates file is provided, unless query IDs are explicitly given,
                self.qids = list(self.queries.keys())  # the IDs from the tokenized queries file will be used
            logger.warning("No static candidates will be used for '{}' dataset".format(self.mode))
            if not self.dynamic_candidates and self.mode != 'train':
                raise ValueError("Evaluation set was initialized with random candidates!")

        logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
        logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

        self.limit_dataset_size(limit_size)  # Potentially trims self.qids, self.candidates

        if mode != 'eval':
            if os.path.isdir(qrels_path):
                qrels_path = os.path.join(qrels_path, "qrels.{}.tsv".format(mode))
            logger.info("Loading ground truth documents (labels) in '{}' ...".format(qrels_path))
            # relevance level for qrels can be different from the one used for injection (and metrics evaluation)
            self.qrels = load_qrels(qrels_path, relevance_thr=include_at_level, score_mapping=relevance_labels_mapping)  # dict{int qID: dict{int pID: float relevance}}
            logger.info("Size of g.t. labels (qrels): {} MB".format(int(round(sys.getsizeof(self.qrels)/1024**2))))
           
            logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
            logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))
           
            # pytrec_eval requires conversion to *str IDs* and *int relevance* for qrels
            logger.info('Making a reformatted copy of qrels ...')
            self.reformatted_qrels = {str(qid): {str(pid): int(score) for pid, score in pdict.items()} for qid, pdict in self.qrels.items()}
            logger.info("Size of reformatted qrels: {} MB".format(int(round(sys.getsizeof(self.reformatted_qrels)/1024**2))))
            
            logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
            logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))
            
            extra_qids = set(self.qids) - set(self.qrels.keys())
            if len(extra_qids) > 0:  # fail early if there are missing labels
                err_str = "{} query IDs in the specified '{}' dataset do not exist in '{}'!".format(len(extra_qids), mode, qrels_path)
                logger.critical(err_str)
                logger.info("Extra query IDs: {}".format(extra_qids))
                raise ValueError(err_str)
        else:
            self.qrels = None

        logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
        logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

        self.include_at_level = include_at_level
        self.relevant_at_level = relevant_at_level
        self.max_inj_relevant = max_inj_relevant  # limits the number of documents considered "relevant" that will be injected (e.g. to leave space for dynamic negatives)


        self.tokenizer = tokenizer
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.max_query_length = max_query_length

        self.num_candidates = num_candidates
        self.candidate_sampling = candidate_sampling

        # Bias mitigation
        self.documents_neutrality = None
        if collection_neutrality_path is not None:
            logger.info("Loading document neutrality scores ...")
            # TODO: wrap this in separate function
            self.documents_neutrality = {}
            for line in open(collection_neutrality_path):
                vals = line.strip().split('\t')
                self.documents_neutrality[int(vals[0])] = float(vals[1])

    def limit_dataset_size(self, limit_size):
        """Changes dataset to a smaller subset, e.g. for debugging"""
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.qids))
            if limit_size < len(self.qids):
                self.qids = self.qids[:limit_size]
                # NOTE: the following is done for efficiency, but is probably unnecessary
                if self.candidates is not None:
                    self.update_candidates(self.qids)
        return

    def update_candidates(self, qids):
        """
        NOTE: This may require less memory, but is probably unnecessary and definitely completely optional.
        Redefines the query IDs to be the specified iterable, instead of the original memmap array.
        When qids change, it trims the self.candidates memmap accordingly.  To accommodate this change, it also
        updates the qid2ind dictionary and corresponding map_qid_to_ind function inside self.candidates
        """
        self.candidates.qids = qids
        query_inds = self.candidates.map_qid_to_ind(qids)  # current indices of qIDs corresponding to memmap arrays
        self.candidates.doc_ids = self.candidates.doc_ids[query_inds, :]  # trims the candidates memmap
        self.candidates.lengths = self.candidates.lengths[query_inds]
        self.candidates.get_qid_to_ind_mapping(qids)  # updates candidates.qid2ind dict and map_qid_to_ind
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
            doc_ids: iterable of candidate document/passage IDs in order of relevance,
                of variable length (depends on number of relevant and static candidates)  # GEO: consider fixing
            rel_scores: list of ground truth relevance scores (labels) for each element in doc_ids[:len(rel_scores)]
                Can have shorter length than doc_ids when injecting positives. None if mode == 'eval'
        """
        global retrieve_candidates_times, prep_docids_times, lookup_times, sample_fetching_times

        sample_fetching_start = time.perf_counter()
        qid = self.qids[ind]
        query_token_ids = self.queries[qid]
        query_token_ids = query_token_ids[:self.max_query_length]
        query_token_ids = [self.cls_id] + query_token_ids + [self.sep_id]

        if not self.dynamic_candidates:
            tic = time.perf_counter()
            if self.candidates is not None:  # typical case
                doc_ids = self.candidates[qid]  # iterable of candidate document/passage IDs in order of estimated relevance
                doc_ids = self.sample_candidates(doc_ids)  # sampled subset of candidate doc_ids. Enforces maximum self.num_candidates
            else:  # if no candidates are provided, randomly samples from entire collection
                # Sampling from 10M docs can take up to 1 sec, when sampling from IDs!
                # Sampling time does not depend on the number of sampled values; only on the size of the population set!
                num_safety_docs = 100  # to account for non-unique randomly sampled documents
                # For MSMARCO, self.emb_collection.ids can be replaced with len(self.emb_collection) -- when replace=True, x10000 faster!???
                if self.emb_collection.id2ind is None:  # if document IDs are ordinal numbers:
                    doc_ids = list(set(np.random.choice(len(self.emb_collection), size=(self.num_candidates+num_safety_docs), replace=True)))
                else:  # x2 faster than if replace=False
                    doc_ids = list(set(np.random.choice(self.emb_collection.ids, size=(self.num_candidates+num_safety_docs), replace=True)))
            retrieve_candidates_times.update(time.perf_counter() - tic)
        else:  # will be retrieved per-batch for efficiency (in collate_fn)
            doc_ids = []

        if self.mode == "train" or self.inject_ground_truth:
            if self.target_scores is not None:
                label_source = self.target_scores
            else:
                label_source = self.qrels
            # Prepend relevant documents at the beginning of doc_ids, whether pre-existing in doc_ids or not,
            # while ensuring that they are only included once.
            # We can limit the number of injected documents from the label source with `max_inj_relevant`, to make space for negatives (e.g. dynamic)
            # Because of sorting, g.t. positives may be excluded when target_scores are provided, 
            # if their score is not high enough to receive a rank < max_inj_relevant
            # NOTE: modified to convert to int IDs - remove int() if entire code is modified to use str IDs!
            rel_cands, rel_scores = zip(*sorted(((int(docid), score) for docid, score in label_source[qid].items() if score >= self.include_at_level), key=itemgetter(1), reverse=True)[:self.max_inj_relevant])
            rel_cands = list(rel_cands)
            if self.label_normalization is not None:
                rel_scores = np.asarray(normalize_batch(torch.tensor(rel_scores, dtype=torch.float16), self.label_normalization), dtype=np.float16) # (len(rel_scores),) tensor
            else:
                rel_scores = np.asarray(rel_scores, dtype=np.float16)

            # also covers case of dynamic candidates, in which initially doc_ids = [] and then doc_ids = rel_cands
            new_cand_ids = (rel_cands + [candid for candid in doc_ids if candid not in set(rel_cands)])
            doc_ids = new_cand_ids  # direct assignment wouldn't work in line above
        elif self.mode == 'dev':
            rel_scores = [self.qrels[qid].get(candid, float('-inf')) for candid in doc_ids]
        else:  # when mode == 'eval'
            rel_scores = None

        sample_fetching_times.update(time.perf_counter() - sample_fetching_start)
        return qid, query_token_ids, doc_ids, rel_scores

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

    def get_collate_func(self, num_random_neg=0, max_candidates=MAX_DOCS, n_gpu=1,
                         label_format='scores', relevant_at_level=None, label_normalization=None):
        """
        :param num_random_neg: number of negatives to randomly sample from other queries in the batch.
            Can only be > 0 if mode == 'train'
        :param max_candidates: maximum number of candidates per query to be scored by the model (capacity of model)
            `num_candidates` will be reduced accordingly, if necessary.
        :param n_gpu: number of GPUs used for data parallelism
        :param label_format: 'indices' or 'scores'.
            If 'scores', `labels` have the same formatting as `predictions`:
                each position is a relevance score, -Inf for non-relevant and padding, e.g. [2.0 1.0 1.0 -Inf ... -Inf]
            If 'indices', labels are num_relevant integer indices of relevant documents and padded with -1,
                e.g. [0, 1, 2, -1, ..., -1]. Used with MaxMargin loss
        :param relevant_at_level: relevance level to be considered relevant in labels (will use init default if None).
            When `label_format` is 'indices', a candidate with a qrels score >= this level is considered relevant,
            which means that all candidates with lower or unspecified g.t. score will be required to be scored lower than this candidate.
            When `label_format` is `scores`, a score < this level will be converted to -Inf (non-relevant), while a score
            >= this level will be considered a target score. For example, a score of 0 will receive some prob. weight.
        :return: function with a single argument, which corresponds to a list of individual sample data. Used by DataLoader
        """
        if self.mode != 'train':
            num_random_neg = 0

        if relevant_at_level is None:
            relevant_at_level = self.relevant_at_level  # set to the level defined in `__init__`

        qrels = None
        if (label_format == 'indices') and \
                ((relevant_at_level != self.include_at_level) or
                 (self.mode == 'dev' and not self.inject_ground_truth)):
            qrels = self.qrels  # in this case the shortcut cannot be used, and qrels is required

        return partial(collate_function, emb_collection=self.emb_collection, mode=self.mode, pad_token_id=self.pad_id,
                       num_random_neg=num_random_neg, max_candidates=max_candidates, n_gpu=n_gpu,
                       label_format=label_format, relevant_at_level=relevant_at_level, label_normalization=label_normalization,
                       qrels=qrels,
                       documents_neutrality=self.documents_neutrality)


def get_docID_to_localind_mapping(doc_ids):
    """Maps unique docIDs to rows of local embedding matrix
    :param doc_ids: (batch_size) list of lists of candidate (retrieved) document IDs corresponding to a query in `qids`.
    """
    docID_to_localind = OrderedDict()
    localind = 0
    for ID in chain.from_iterable(doc_ids):
        if ID not in docID_to_localind:
            docID_to_localind[ID] = localind
            localind += 1
    return docID_to_localind


def pack_tensor_2D(sequences, default, dtype, length=None):
    """
    Padds and/or truncates each sequence in the input list of sequences to the specified `length` or the max. sequence
    length in the list. Padded values are filled with `default`.
    :param sequences: list of sequences to be packed into a tensor (batch)
    :param default: constant value to be used as padding
    :param dtype: type of tensor
    :param length: constant output length of all sequences in the batch;
        if None, the max. sequence length in `lstlst` will be used
    :return: (len(sequences), length) tensor of type dtype
    """
    batch_size = len(sequences)
    length = length if length is not None else max(len(seq) for seq in sequences)
    tensor = torch.full((batch_size, length), default, dtype=dtype)
    for i, seq in enumerate(sequences):
        end = min(len(seq), length)
        tensor[i, :end] = torch.tensor(seq[:end], dtype=dtype)  # casting to tensor required for assignment
    return tensor


def collate_function(batch_samples, mode, emb_collection, pad_token_id, num_random_neg=0, max_candidates=MAX_DOCS,
                     n_gpu=1, label_format='scores', relevant_at_level=1, label_normalization=None, qrels=None,
                     documents_neutrality=None):
    """
    :param batch_samples: (batch_size) list of tuples (qids, query_token_ids, doc_ids, rel_scores)
    :param emb_collection: (collection_size, emb_dim) numpy.memmap of document collection embedding vectors.
                Used to look up embedding vectors by doc_id
    :param mode: 'train', 'dev', or 'eval'
    :param pad_token_id: ID of token used for padding queries
    :param num_random_neg: number of negatives to randomly sample from candidates of other queries in the batch.
                Can only be > 0 if mode == 'train'
    :param max_candidates: maximum number of candidates per query to be scored by the model (capacity of model)
    :param n_gpu: number of GPUs used for data parallelism
    :param label_format: 'indices' or 'scores'.
              If 'scores', returned rows of `labels` have the same formatting as `predictions`:
                  each position is a relevance score, -Inf for non-relevant and padding, e.g. [2.0 1.0 1.0 -Inf ... -Inf]
              If 'indices', each row of `labels` is num_relevant integer indices of relevant documents and is padded with -1,
                  e.g. [0, 1, 2, -1, ..., -1]. Used with MaxMargin loss
    :param relevant_at_level: relevance level to be considered relevant in labels.
        When `label_format` is 'indices', a candidate with a qrels score >= this level is considered relevant,
        which means that all other candidates will be pushed to be scored lower than this level.
        When `label_format` is `scores`, a score < this level will be converted to -Inf (non-relevant), while a score
        >= this level will be considered a target score. For example, a score of 0 will receive some prob. weight.
    :param qrels: (optional) {query ID: {candidate ID: relevance score}} mapping,
        to be used when `label_format` == 'indices' and mode == 'dev' when 'inject_ground_truth' is False
    :return:
        qids: (batch_size) list of query IDs
        doc_ids: (batch_size) list of lists of candidate (retrieved) document IDs corresponding to a query in `qids`.
            If num_random_neg > 0, it includes the docIDs of randomly sampled in-batch negatives
        data: dict to serve as input to the model. Contains:
            query_token_ids: (batch_size, max_query_len) int tensor of query token IDs
            query_mask: (batch_size, max_query_len) bool tensor of padding mask corresponding to query; 1 use, 0 ignore
            doc_padding_mask: (batch_size, max_cands_per_query) boolean tensor indicating padding (0) or valid (1) elements
                in `doc_emb` or `docinds`
            If num_random_neg > 0, additionally contains:
            local_emb_mat: (num_unique_docIDs, emb_dim) tensor of local doc embedding matrix containing emb. vectors
                of all unique documents in the batch.  Used to lookup document vectors in nn.Embedding on the GPU
                This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth.
            docinds: (batch_size, max_cands_per_query) local indices of documents corresponding to `local_emb_mat`
            else:
            doc_emb: (batch_size, max_cands_per_query, emb_dim) float tensor of document embeddings corresponding
                to the pool of candidates for each query

            If mode != 'eval', additionally contains:
            labels: (batch_size, max_cands_per_query) tensor of labels, the format of which depends on `label_format`
    """
    start_collation_time = time.perf_counter()
    batch_size = len(batch_samples)

    qids, query_token_ids, doc_ids, rel_scores = zip(*batch_samples)
    query_lengths = [len(seq) for seq in query_token_ids]
    max_query_length = max(query_lengths)
    query_masks = [[1] * ql for ql in query_lengths]  # 1 use, 0 ignore

    # The 2D tensors are padded NOT to a standard length (transformer input length), but to the query max_len in the batch
    data = {"query_token_ids": pack_tensor_2D(query_token_ids, default=pad_token_id, length=max_query_length, dtype=torch.int32),
            "query_mask": pack_tensor_2D(query_masks, default=0, length=max_query_length, dtype=torch.bool)}

    if num_random_neg:  # only in 'train' mode. Will include *random* (i.e. not retrieved) negatives
        # In this case, the doc embeddings are not packed here, but inside the model (on the GPU), in order
        # to avoid transferring replicas of document embeddings corresponding to in-batch negatives and thus spare GPU mem. bandwidth.
        # For this purpose, we pass a smaller local doc embedding matrix containing emb. vectors of all documents
        # in the batch and a dictionary mapping docID to local indices corresponding to this matrix

        docID_to_localind = get_docID_to_localind_mapping(doc_ids)  # maps unique docIDs to rows of local embedding matrix
        unique_candidates = docID_to_localind.keys()  # all unique document IDs in the batch (akin to a set, but keeping the order)

        try:
            # augment doc_ids with randomly sampled document IDs from candidates retrieved for other qIDs in the batch
            doc_ids = [(cands + list(
                np.random.choice(list(unique_candidates - set(cands)), size=num_random_neg, replace=False)))[:max_candidates]
                       for cands in doc_ids]
        except ValueError:  # this happens when not enough in-batch documents exist to satisfy num_random_neg per query
            # Sample additional random documents (on top of in-batch)
            # Sampling from 10M docs can take up to 1 sec, when sampling from IDs!
            # Sampling time does not depend on the number of sampled values; only on the size of the population set!
            # However, we still want to use as few unique documents in the batch as possible (because it takes GPU memory, bandwidth)
            exp_shared = 3  # safety factor. Number of docs/query expected to be non-unique (shared with other queries) or missing
            num_cand = len(doc_ids[0])  # number of retrieved candidates per query. exp_shared covers queries with fewer candidates
            num_remaining = batch_size * (num_random_neg - len(unique_candidates) + num_cand + exp_shared)

            # For MSMARCO, doc IDs can be replaced with ordinal numbers -- when replace=True, x10000 faster!??
            if emb_collection.id2ind is None:  # if document IDs are ordinal numbers:
                extra_docs = np.random.choice(len(emb_collection), size=num_remaining)
            else:  # much slower
                extra_docs = np.random.choice(emb_collection.ids, size=num_remaining)
            negatives_pool = set(unique_candidates) | set(extra_docs)  # in-batch and randomly sampled documents

            doc_ids = [(cands + list(
                np.random.choice(list(negatives_pool - set(cands)), size=num_random_neg, replace=False)))[:max_candidates]
                       for cands in doc_ids]  # replace=False doesn't hurt here, because the population is small
            docID_to_localind = get_docID_to_localind_mapping(doc_ids)  # maps unique docIDs to rows of local embedding matrix
            unique_candidates = docID_to_localind.keys()  # all unique document IDs in the batch (akin to a set, but keeping the order)

        # Assemble (num_unique_docs + 1, emb_dim) local doc embedding matrix: to add an emb. corresponding to padding,
        # we concatenate 1 extra row of 0s at the end of the (num_unique_docs, emb_dim) tensor of unique doc. embeddings
        tic = time.perf_counter()
        local_emb_mat = torch.cat((torch.tensor(emb_collection[list(unique_candidates)]),
                                   torch.zeros((1, emb_collection.embedding_vectors.shape[-1]))), dim=0)
        lookup_times.update(time.perf_counter() - tic)

        data['docinds'], data['doc_padding_mask'] = prepare_docinds_and_mask(doc_ids, docID_to_localind,
                                                                             padding_idx=local_emb_mat.shape[0]-1)
        data['local_emb_mat'] = local_emb_mat.repeat(n_gpu, 1)  # replicas passed to each split of multi-gpu

    max_cands_per_query = min(max_candidates, max(len(cands) for cands in doc_ids))  # length used for padding

    if num_random_neg == 0: # Will only use retrieved candidates (either as negatives or for evaluation)
        # Pack 3D tensors: candidate embeddings and corresponding padding mask
        # Tensors are padded NOT to a standard length (transformer input length), but to the max_len in the batch
        data['doc_emb'], data['doc_padding_mask'] = pack_candidate_embeddings(doc_ids, emb_collection, batch_size, max_cands_per_query)
        
    if mode != 'eval':
        # Prepare labels
        if label_format == 'scores':
            # must be padded to have same dimensions as the transformer "decoder" output: (batch_size, max_cands_per_query)
            # Because softmax(rel_scores) is used to derive the distribution, -Inf must be used such that prob. for all non-relevant is 0
            # -Inf is also used for padding, and should also be used at the respective masked positions of predicted scores
            labels = pack_tensor_2D(rel_scores, default=float('-inf'), dtype=torch.float32, length=max_cands_per_query)
            if label_normalization is not None:
                labels = normalize_batch(labels, label_normalization)
            labels[labels < relevant_at_level] = float('-inf')  # necessary to enforce relevance threshold for loss computation
            data['labels'] = labels
        else:  # 'indices'
            if qrels is None:
                # faster variant (shortcut)
                # valid when: (mode == "train" or inject_ground_truth) AND (relevant_at_level == include_at_level)
                # `labels` holds for each query the indices of the relevant doc IDs within its corresponding pool of candidates, `doc_ids`
                # In this case, it is guaranteed by the construction of doc_ids in MYMARCO_Dataset.__get_item__
                # (and the fact that we are only randomly sampling negatives not already in the list of candidates for a given qID)
                # that 1 or more positives (rel. docs) will be at the beginning of the list (and only there)
                labels = [list(range(len(rd))) for rd in rel_scores]  # rel_scores contains only scores >= dataset.include_at_level
            else:  # this is when 'dev', but not inject_ground_truth
                labels = prepare_ind_labels(qids, doc_ids, qrels, relevance_level=relevant_at_level)
            # must be padded with -1 to have same dimensions as the transformer "decoder" output: (batch_size, max_cands_per_query)
            data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int16, length=max_cands_per_query)  # supports up to 32767 rel. documents per query

    # Bias mitigation
    if documents_neutrality is not None:
        data['doc_neutscore'] = prepare_neutrality_labels(doc_ids, documents_neutrality, max_cands_per_query)

    global collation_times
    collation_times.update(time.perf_counter() - start_collation_time)

    return data, qids, doc_ids


def prepare_neutrality_labels(doc_ids, documents_neutrality, max_cands_per_query):
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

    # TODO: use pack_tensor_2D?
    doc_neutscores_tensor = torch.zeros(len(doc_ids), max_cands_per_query)  # (batch_size, padded_length)
    for i, doc_neutscores_batch in enumerate(doc_neutscores):
        end = min(len(doc_neutscores_batch), max_cands_per_query)
        doc_neutscores_tensor[i, :end] = torch.tensor(doc_neutscores_batch)[:end]
    return doc_neutscores_tensor


def prepare_ind_labels(qids, doc_ids, qrels, relevance_level=1):
    """
    Prepare relevance labels for `label_format` == 'indices' (used for max margin losses).
    This is necessary when mode == 'dev' and not `inject_ground_truth` - otherwise:
        labels = [list(range(len(rd))) for rd in rel_scores]
    :return: list, each item is a list of index locations (not necessarily sorted!) within doc_ids corresponding to g.t. relevant candidates.
    """
    labels = []
    for i in range(len(qids)):
        query_labels = []
        for ind in range(len(doc_ids[i])):
            score = qrels[qids[i]].get(doc_ids[i][ind], -1)
            if score >= relevance_level:
                query_labels.append(ind)
        labels.append(query_labels)
    return labels


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


def pack_candidate_embeddings(cand_ids, emb_collection, batch_size, max_len):
    """Pack 3D tensors: candidate embeddings and corresponding padding mask"""

    global lookup_times

    cand_emb_tensor = torch.zeros(batch_size, max_len,
                                  emb_collection.embedding_vectors.shape[-1])  # (batch_size, padded_length, emb_dim)
    cand_emb_mask = torch.zeros(cand_emb_tensor.shape[:2], dtype=torch.bool)  # (batch_size, padded_length)
    for i, candids in enumerate(cand_ids):
        end = min(len(candids), max_len)
        tic = time.perf_counter()
        cand_emb_tensor[i, :end, :] = torch.tensor(emb_collection[candids[:end]])
        lookup_times.update(time.perf_counter() - tic)
        cand_emb_mask[i, :end] = 1
    return cand_emb_tensor, cand_emb_mask


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


# NOTE: only used by RepBERT! (consider removing)
# class MSMARCODataset(Dataset):
#     def __init__(self, mode, msmarco_dir,
#                  collection_memmap_dir, queries_tokenids_path,
#                  max_query_length=20, max_doc_length=256, limit_size=None):
#         """
#         :param limit_size: If set, limit dataset size to a smaller subset, e.g. for debugging. If in [0,1], it will
#             be interpreted as a proportion of the dataset, otherwise as an integer absolute number of samples.
#         """
#         self.collection = CollectionDataset(collection_memmap_dir)
#         if os.path.isdir(queries_tokenids_path):
#             queries_tokenids_path = os.path.join(queries_tokenids_path, "queries.{}.json".format(mode))
#         self.queries = load_query_tokenids(queries_tokenids_path)  # dict: {qID : list of token IDs}
#         # qids, pids, labels: corresponding lists of query IDs, passage IDs, 1 / 0 relevance labels
#         # each query ID is contained 2 consecutive times in qids, once corresponding to the positive and once to the negative pid
#         # qrels is the ground truth dict: {qID : set of relevant pIDs}
#         self.qids, self.pids, self.labels, self.qrels = load_querydoc_pairs(msmarco_dir, mode)
#         self.mode = mode  # "train", "dev", "eval"
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         self.cls_id = tokenizer.cls_token_id
#         self.sep_id = tokenizer.sep_token_id
#         self.max_query_length = max_query_length
#         self.max_doc_length = max_doc_length
#         self.limit_dataset_size(limit_size)

#     def limit_dataset_size(self, limit_size):
#         """Changes dataset to a smaller subset, e.g. for debugging"""
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.qids))
#             if limit_size < len(self.qids):
#                 self.qids = self.qids[:limit_size]
#                 self.pids = self.pids[:limit_size]
#                 self.labels = self.labels[:limit_size]
#                 # self.candidates.lengths = self.candidates.lengths[:limit_size]
#                 # self.candidates.get_qid_to_ind_mapping(self.qids)
#         return

#     def get_collate_function(self, n_gpu=1):  # WARNING: n_gpu is not handled in the function! (must be updated for n_gpu>1)
#         def collate_function(batch):
#             input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
#             token_type_ids_lst = [[0] * len(x["query_input_ids"]) + [1] * len(x["doc_input_ids"]) for x in batch]
#             valid_mask_lst = [[1] * len(input_ids) for input_ids in input_ids_lst]
#             position_ids_lst = [list(range(len(x["query_input_ids"]))) +
#                                 list(range(len(x["doc_input_ids"]))) for x in batch]

#             # The 2D tensors are padded NOT to a standard length (transformer seq. length), but to the max_len in the batch!
#             data = {
#                 "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
#                 # int64 is required by torch nn.Embedding :(
#                 "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
#                 "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
#                 "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
#             }
#             qid_lst = [x['qid'] for x in batch]  # GEO: this can be avoided by returning qid, docid as part of a tuple in __getitem__
#             docid_lst = [x['docid'] for x in batch]
#             if self.mode == "train":
#                 # labels holds the in-batch indices of the relevant doc IDs for each query
#                 labels = [[j for j in range(len(docid_lst)) if docid_lst[j] in x['rel_docs']] for x in batch]
#                 data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int64, length=len(batch))
#             return data, qid_lst, docid_lst

#         return collate_function

#     def __len__(self):
#         return len(self.qids)

#     def __getitem__(self, item):
#         qid, pid = self.qids[item], self.pids[item]
#         query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
#         query_input_ids = query_input_ids[:self.max_query_length]
#         query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
#         doc_input_ids = doc_input_ids[:self.max_doc_length]
#         doc_input_ids = [self.cls_id] + doc_input_ids + [self.sep_id]

#         ret_val = {
#             "query_input_ids": query_input_ids,
#             "doc_input_ids": doc_input_ids,
#             "qid": qid,
#             "docid": pid
#         }
#         if self.mode == "train":
#             ret_val["rel_docs"] = self.qrels[qid]
#         return ret_val
