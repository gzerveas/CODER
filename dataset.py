import os
import json
import logging
from collections import defaultdict, OrderedDict
from functools import partial
from itertools import chain

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd

logger = logging.getLogger(__name__)

BERT_BASE_DIM = 768


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

    def __init__(self, embedding_memmap_dir, emb_dim=None, sorted=False):
        """
        :param embedding_memmap_dir: directory containing memmap file of precomputed document embeddings, and the
            corresponding passage/doc IDs in another memmpap file
        :param emb_dim: dimensionality of document vectors. If None, `embedding_memmap_dir` will be used to infer it,
            and in case this is not possible, the default value BERT_BASE_DIM will be used
        :param sorted: whether passage/doc IDs in the collection happen to exactly be 0, 1, ..., num_docs-1, and
            doc embeddings are stored exactly in that order (is True  for MSMARCO passage collection)
        """

        if emb_dim is None:
            try:
                # assumes that name of embedding directory contains dimensionality like: "path/embedding_768"
                emb_dim = int(os.path.dirname(embedding_memmap_dir).split('_')[-1])
                logger.warning("Inferred doc embedding dimensionality {} from directory name: {}".format(emb_dim,
                                                                                                         embedding_memmap_dir))
            except ValueError:
                logger.warning(
                    "Could not infer doc embedding dimensionality from directory name: {}".format(embedding_memmap_dir))
                emb_dim = BERT_BASE_DIM
                logger.warning("Using default document embedding dimensionality: {}".format(emb_dim))

        pids = np.memmap(os.path.join(embedding_memmap_dir, "pids.memmap"), mode='r', dtype='int32')

        self.sorted = sorted  # whether passage/doc IDs in the collection happen to exactly be 0, 1, ..., num_docs-1
        self.pid2ind = None  # no mapping dictionary in case sorted == True
        self.map_pid_to_ind = self.get_pid_to_ind_mapping(pids)  # pID-to-matrix_index mapping function

        self.num_docs = len(pids)
        # shape is necessary input; this information is not stored and a 1D array is loaded by default
        self.embedding_vectors = np.memmap(os.path.join(embedding_memmap_dir, "embedding.memmap"), mode='r',
                                           dtype='float32', shape=(self.num_docs, emb_dim))

    def get_pid_to_ind_mapping(self, pids):
        """Returns function used to map a list of passage IDs to the corresponding integer indices of the embedding matrix"""
        if self.sorted:
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


def load_query_tokenids(query_tokenids_path):
    """
    :param query_tokenids_path: path to JSON file of {int qid: tokenized and numerized query} pairs
    :return: queries: dict {int qid: list of query token IDs (unpadded, no special tokens)}
    """
    queries = dict()
    with open(query_tokenids_path) as f:
        for line in tqdm(f, desc="queries"):
            data = json.loads(line)
            queries[int(data['id'])] = data['ids']  # already tokenized and numerized!
    return queries


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
                         # TODO: this file does not exist. they made it themselves
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
    """Load ground truth relevant passages from file. Assumes a *single* level of relevance (1), an assumption that holds
    for MSMARCO qrels.{train,dev}.tsv
    :param filepath: path to file of ground truth relevant passages in the following format:
        "qID1 \t Q0 \t pID1 \t 1\n qID1 \t Q0 \t pID2 \t 1\n ..."
    :return:
        qrels: dict: {int qID : set of ground truth relevant int pIDs}
    """
    qrels = defaultdict(set)
    with open(filepath, 'r') as f:
        for line in f:
            qid, _, pid, _ = line.split()  # the _ account for uninformative placeholders in TREC format
            qrels[int(qid)].add(int(pid))
    return qrels


# Not used currently
def load_candidates(path_to_candidates):
    """
    Load candidate (retrieved) documents/passages from a file.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).
    :param path_to_candidates: path to file of candidate (retrieved) documents/passages per query
    :return:
        qid_to_candidate_passages: dict: {int qID : list of retrieved int pIDs in order of relevance}
    """

    qid_to_candidate_passages = defaultdict(list)  # dict: {qID : list of retrieved pIDs in order of relevance}

    with open(path_to_candidates, 'r') as f:
        for line in f:
            try:
                fields = line.strip().split('\t')
                qid = int(fields[0])
                pid = int(fields[1])

                qid_to_candidate_passages[qid].append(pid)
            except Exception as x:
                print(x)
                logger.warning("Line \"{}\" is not in valid format and resulted in: {}".format(line, fields))
    return qid_to_candidate_passages


def load_candidates_pandas(path_to_candidates):
    """
    Load candidate (retrieved) documents/passages from a file.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).
    :param path_to_candidates: path to file of candidate (retrieved) documents/passages per query
    :return: pandas dataframe of candidate passage IDs indexed by qID. Multiple rows correspond to the same qID, each row is a pID
    """
    # internally uses C for parsing, and multiple chunks
    candidates_df = pd.read_csv(path_to_candidates, delimiter='\t', index_col=0, memory_map=True, dtype=np.int32)

    return candidates_df


# TODO: from top1000 results file, 1a) make a precomputed (num_queries, topN) mmap of qid -> topN pids
# TODO: or 1b) read qid -> topN pids dict into memory (probably better, but parallelize for speed?)
# TODO: 2) precompute (num_docs, doc_emb_dim) memmap "doc_emb"
# TODO: 3) use "doc_emb" instead of CollectionDataset in the below dataset
class MYMARCO_Dataset(Dataset):
    def __init__(self, mode,
                 embedding_memmap_dir, queries_tokenids_path, candidates_path, qrels_path=None,
                 tokenizer=None, max_query_length=64, num_candidates=None, candidate_sampling=None):
        """
        :param mode:
        :param embedding_memmap_dir:
        :param queries_tokenids_path:
        :param candidates_path:
        :param qrels_path:
        :param tokenizer:
        :param max_query_length:
        :param num_candidates: number of document IDs to sample from all document IDs corresponding to a query and found
            in `candidates_path` file. If None, all found document IDs will be used
        :param candidate_sampling: method to use for sampling candidates. If None, the top `num_candidates` will be used
        """

        self.emb_collection = EmbeddedCollection(embedding_memmap_dir, emb_dim=None, sorted=True)
        self.queries = load_query_tokenids(queries_tokenids_path)  # dict: {qID : list of token IDs}
        self.candidates_df = load_candidates_pandas(
            candidates_path)  # pandas dataframe of candidate pIDs indexed by qID
        self.qids = self.candidates_df.index.unique()  # Series of qIDs as found in retrieved candidates file

        self.mode = mode  # "train", "dev", "eval"
        if mode == 'train':
            self.qrels = load_qrels(qrels_path)  # dict: {qID : set of ground truth relevant pIDs}
        else:
            self.qrels = None

        if tokenizer is None:  # TODO: for backwards compatibility, consider removing
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        self.max_query_length = max_query_length

        self.num_candidates = num_candidates
        self.candidate_sampling = candidate_sampling

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, ind):
        """
        For a given integer index corresponding to a sample (query), returns a tuple of model input data
        Args:
            ind: integer index of sample (query) in dataset
        Returns:
            qid: (int) query ID
            query_token_ids: list of token IDs corresponding to query (unpadded, includes start/stop tokens)
            doc_ids: iterable of candidate document/passage IDs in order of relevance
            doc_embeddings: (len(doc_ids), emb_dim) slice of numpy.memmap embedding vectors corresponding to `doc_ids`
        if self.mode == 'train':
            rel_docs: set of ground truth relevant passage IDs corresponding to query ID
        """

        qid = self.qids[ind]
        query_token_ids = self.queries[qid]
        query_token_ids = query_token_ids[:self.max_query_length]
        query_token_ids = [self.cls_id] + query_token_ids + [self.sep_id]

        doc_ids = self.candidates_df.loc[qid]  # iterable of candidate document/passage IDs in order of relevance
        doc_ids = self.sample_candidates(doc_ids)  # sampled subset of candidate doc_ids

        if self.mode == "train":
            rel_docs = self.qrels[qid]
            # prepend relevant documents at the beginning of doc_ids, whether pre-existing in doc_ids or not,
            # while ensuring that they are only included once
            num_candidates = len(doc_ids)
            new_doc_ids = (list(rel_docs) + [docid for docid in doc_ids if docid not in rel_docs])[:num_candidates]
            doc_ids = new_doc_ids  # direct assignment wouldn't work in line above

            doc_embeddings = self.emb_collection[doc_ids]  # (num_doc_ids, emb_dim) slice of numpy.memmap embeddings
            return qid, query_token_ids, doc_ids, doc_embeddings, rel_docs
        else:
            doc_embeddings = self.emb_collection[doc_ids]  # (num_doc_ids, emb_dim) slice of numpy.memmap embeddings
            return qid, query_token_ids, doc_ids, doc_embeddings

    def sample_candidates(self, candidates):
        """
        Sample `self.num_candidates` from all retrieved candiadate document IDs corersponding to a query,
        according to method `self.candidate_sampling`
        :param candidates: iterable of candidate document/passage IDs (corresponding to a query) in order of relevance
        :return: list of len(self.num_candidates) subset of `candidates`
        """
        if self.num_candidates:
            return candidates[:self.num_candidates]
        # TODO: implement more sampling methods (e.g. for multi-tiered relevance loss)
        else:
            return candidates

    def get_collate_func(self, num_inbatch_neg=0, max_candidates=1000):

        if self.mode != 'train':
            num_inbatch_neg = 0
        return partial(collate_function, mode=self.mode, pad_token_id=self.pad_id, num_inbatch_neg=num_inbatch_neg,
                       max_candidates=max_candidates)


def collate_function(batch, mode, pad_token_id, num_inbatch_neg=0, max_candidates=1000):
    """
    :param batch: (batch_size) list of tuples (qids, query_token_ids, doc_ids, doc_embeddings, <rel_docs?>)
    :param mode: 'train' or 'eval'
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

            If num_inbatch_neg > 0, additionally contains:
            local_emb_mat: (num_unique_docIDs, emb_dim) tensor of local doc embedding matrix containing emb. vectors
                of all documents in the batch
            docID_to_localind: OrderedDict mapping unique docIDs to rows of `local_emb_mat`. Used to assemble tensors
                on the GPU. This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth
            else:
            doc_emb_tensor: (batch_size, max_docs_per_query, emb_dim) float tensor of document embeddings corresponding
                to the pool of candidates for each query
            doc_emb_mask: (batch_size, max_docs_per_query) boolean tensor indicating padding (0) or valid (1) elements
                in `doc_emb_tensor`.

            If 'train', additionally contains:
            labels: (batch_size, max_docs_per_query) int tensor which for each query (row) contains the indices of the
                relevant doc IDs within its corresponding pool of candidates, `doc_ids`
    """

    batch_size = len(batch)

    qids, query_token_ids, doc_ids, doc_embeddings, rel_docs = zip(*batch)
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

        docID_to_localind = OrderedDict((ID, i) for i, ID in enumerate(chain.from_iterable(doc_ids)))
        # get all unique document IDs in the batch. This is an object akin to a set, but keeping the order
        unique_candidates = docID_to_localind.keys()  # all unique document IDs in the batch

        # augment doc_ids with randomly sampled document IDs from candidates retrieved for other qIDs in the batch
        doc_ids = [(cands + list(
            np.random.choice(list(unique_candidates - set(cands)), size=num_inbatch_neg, replace=False)))[
                   :max_candidates]
                   for cands in doc_ids]

        # local doc embedding matrix. It is only slightly smaller than `assembled_emb_mat`, because the chance of
        # duplicate docIDs in the original (unaugmented) `doc_ids` is very small (so its creation can be omitted)
        local_emb_mat = torch.zeros(len(unique_candidates),
                                    doc_embeddings[0].shape[-1])  # (num_unique_docs, emb_dim)
        for i, docid in enumerate(unique_candidates):
            local_emb_mat[i, :] = assembled_emb_mat[docID_to_localind[docid], :]
            docID_to_localind[docid] = i  # re-assign mapping to smaller local_emb_mat (from assembled_emb_mat)

        data['local_emb_mat'] = local_emb_mat
        data['docID_to_localind'] = docID_to_localind  # also contains unique_candidates

    max_docs_per_query = min(max_candidates, max(len(cands) for cands in doc_ids))  # length used for padding

    if num_inbatch_neg == 0:
        # Pack 3D tensors: doc embeddings and corresponding padding mask
        doc_emb_tensor = torch.zeros(batch_size, max_docs_per_query, doc_embeddings[0].shape[-1])  # (batch_size, padded_length, emb_dim)
        doc_emb_mask = torch.zeros(doc_emb_tensor, dtype=torch.bool)  # (batch_size, padded_length)
        for i, doc_embs in enumerate(doc_embeddings):
            end = min(doc_embs.shape[0], max_docs_per_query)
            doc_emb_tensor[i, :end, :] = doc_embs[:end, :]
            doc_emb_mask[i, :end] = 1
        data['doc_emb_tensor'] = doc_emb_tensor
        data['doc_emb_mask'] = doc_emb_mask

    if mode == "train":
        # `labels` holds for each query the indices of the relevant doc IDs within its corresponding pool of candidates, `doc_ids`
        # It is guaranteed by the construction of doc_ids in MYMARCO_Dataset.__get_item__
        # (and the fact that we are only randomly sampling negatives not already in the list of candidates for a given qID)
        # that 1 or more positives (rel. docs) will be at the beginning of the list (and only there)
        labels = [list(range(len(rd))) for rd in rel_docs]
        # must be padded with -1 to have same dimensions as the transformer decoder output: (batch_size, max_docs_per_query)
        data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int16, length=max_docs_per_query)

    return data, qids, doc_ids


class MSMARCODataset(Dataset):
    def __init__(self, mode, msmarco_dir,
                 collection_memmap_dir, tokenize_dir,
                 max_query_length=20, max_doc_length=256):
        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)  # dict: {qID : list of token IDs}
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


def get_collate_function(mode):
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
        token_type_ids_lst = [[0] * len(x["query_input_ids"]) + [1] * len(x["doc_input_ids"]) for x in batch]
        valid_mask_lst = [[1] * len(input_ids) for input_ids in input_ids_lst]
        position_ids_lst = [list(range(len(x["query_input_ids"]))) +
                            list(range(len(x["doc_input_ids"]))) for x in batch]

        # The 2D tensors are padded NOT to a standard length (transformer seq. length), but to the max_len in the batch!
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in
                   batch]  # TODO: this can be avoided by returning qid, docid as part of a tuple in __getitem__
        docid_lst = [x['docid'] for x in batch]
        if mode == "train":
            # labels holds the in-batch indices of the relevant doc IDs for each query
            labels = [[j for j in range(len(docid_lst)) if docid_lst[j] in x['rel_docs']] for x in batch]
            data['labels'] = pack_tensor_2D(labels, default=-1, dtype=torch.int64, length=len(batch))
        return data, qid_lst, docid_lst

    return collate_function


def _test_dataset():
    dataset = MSMARCODataset(mode="train")
    for data in dataset:
        tokens = dataset.tokenizer.convert_ids_to_tokens(data["query_input_ids"])
        print(tokens)
        tokens = dataset.tokenizer.convert_ids_to_tokens(data["doc_input_ids"])
        print(tokens)
        print(data['qid'], data['docid'], data['rel_docs'])
        print()
        k = input()
        if k == "q":
            break


def _test_collate_func():
    from torch.utils.data import DataLoader, SequentialSampler
    eval_dataset = MSMARCODataset(mode="train")
    train_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_function(mode="train")
    dataloader = DataLoader(eval_dataset, batch_size=26,
                            num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
    tokenizer = eval_dataset.tokenizer
    for batch, qidlst, pidlst in tqdm(dataloader):
        pass
        '''
        print(batch['input_ids'])
        print(batch['token_type_ids'])
        print(batch['valid_mask'])
        print(batch['position_ids'])
        print(batch['labels'])
        k = input()
        if k == "q":
            break
        '''


if __name__ == "__main__":
    _test_collate_func()
