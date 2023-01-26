import sys

import os
import math
import random
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from collections import namedtuple, defaultdict, OrderedDict
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from dataset import CollectionDataset, pack_tensor_2D, MSMARCODataset
from utils import generate_rank, eval_results
import logging
import time

logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()


def get_embed_memmap(memmap_dir, dim):
    embedding_path = os.path.join(memmap_dir, "embedding.memmap")
    id_path = os.path.join(memmap_dir, "ids.memmap")
    # Tensor doesn't support non-writeable numpy array
    # Thus we use copy-on-write mode 
    try:
        id_memmap = np.memmap(id_path, dtype='int32', mode="c")
    except FileNotFoundError:
        id_path = os.path.join(memmap_dir, "pids.memmap")
        id_memmap = np.memmap(id_path, dtype='int32', mode="c")
    embedding_memmap = np.memmap(embedding_path, dtype='float32', mode="c", shape=(len(id_memmap), dim))
    return embedding_memmap, id_memmap


def print_memory_info(memmap, docs_per_chunk, device):
    doc_chunk_size = sys.getsizeof(np.array(memmap[:docs_per_chunk]))/1024**2
    logger.info("{} chunks of {} documents (total of {}), each of approx. size {} MB, "
                "will be loaded to the following device:".format(math.ceil(memmap.shape[0]/docs_per_chunk), docs_per_chunk,
                                                                 memmap.shape[0], math.ceil(doc_chunk_size)))

    if device.type == 'cuda':
        logger.info("Device: {}".format(torch.cuda.get_device_name(0)))
        total_mem = torch.cuda.get_device_properties(0).total_memory/1024**2
        logger.info("Total memory: {} MB".format(math.ceil(total_mem)))
        reserved_mem = torch.cuda.memory_reserved(0)/1024**2
        logger.info("Reserved memory: {} MB".format(math.ceil(reserved_mem)))
        allocated_mem = torch.cuda.memory_allocated(0)/1024**2
        logger.info("Allocated memory: {} MB".format(math.ceil(allocated_mem)))
        free_mem = total_mem - allocated_mem
        logger.info("Free memory: {} MB".format(math.ceil(free_mem)))
        logger.warning("This device could potentially support "
                       "`per_gpu_doc_num` up to {}".format(math.floor(args.per_gpu_doc_num*free_mem/doc_chunk_size)))
    else:
        logger.info("CPU")


def allrank(args):
    """Retrieves from the entire collection as found in document embeddings memmap"""
    logger.info("Loading document embeddings memmap ...")
    doc_embedding_memmap, doc_id_memmap = get_embed_memmap(args.doc_embedding_dir, args.embedding_dim)
    # assert np.all(doc_id_memmap == list(range(len(doc_id_memmap))))  # NOTE: valid only for MSMARCO

    print_memory_info(doc_embedding_memmap, args.per_gpu_doc_num, args.device)

    logger.info("Loading query embeddings memmap ...")
    query_embedding_memmap, query_id_memmap = get_embed_memmap(args.query_embedding_dir, args.embedding_dim)
    qid2pos = {identity: i for i, identity in enumerate(query_id_memmap)}

    if args.query_ids is None:
        query_ids = query_id_memmap
    else:  # read subset of (integer) query IDs from file
        logger.info("Will use queries inside: {}".format(args.query_ids))
        with open(args.query_ids, 'r') as f:
            query_ids = OrderedDict.fromkeys(int(line.split()[0]) for line in f)  # acts as an ordered set

    logger.info("{} queries found".format(len(query_ids)))

    # PriorityQueue has a O(nlogn) insertion, and keeps elements sorted (in ascending order)
    results_dict = {qid: PriorityQueue(maxsize=args.hit) for qid in query_ids}

    logger.info("Retrieving documents ...")
    for doc_begin_index in tqdm(range(0, len(doc_id_memmap), args.per_gpu_doc_num), desc="doc"):
        doc_end_index = doc_begin_index+args.per_gpu_doc_num
        doc_ids = doc_id_memmap[doc_begin_index:doc_end_index]
        doc_embeddings = doc_embedding_memmap[doc_begin_index:doc_end_index]
        doc_embeddings = torch.from_numpy(doc_embeddings).to(args.device)
        for qid in tqdm(query_ids, desc="query"):
            query_embedding = query_embedding_memmap[qid2pos[qid]]
            query_embedding = torch.from_numpy(query_embedding)
            query_embedding = query_embedding.to(args.device)
        
            all_scores = torch.sum(query_embedding * doc_embeddings, dim=-1)
            
            k = min(args.hit, len(doc_embeddings))
            top_scores, top_indices = torch.topk(all_scores, k, largest=True, sorted=True)
            top_scores, top_indices = top_scores.cpu(), top_indices.cpu()
            top_doc_ids = doc_ids[top_indices.numpy()]
            cur_q_queue = results_dict[qid]
            for score, docid in zip(top_scores, top_doc_ids):
                score, docid = score.item(), docid.item()
                if cur_q_queue.full():
                    lowest_score, lowest_docid = cur_q_queue.get_nowait()
                    if lowest_score >= score:
                        cur_q_queue.put_nowait((lowest_score, lowest_docid))
                        break
                    else:
                        cur_q_queue.put_nowait((score, docid))
                else:
                    cur_q_queue.put_nowait((score, docid))

    logger.info("Writing scores and ranks to file: {} ...".format(args.output_path))
    with open(args.output_path, 'w') as outFile:
        for qid, docqueue in tqdm(results_dict.items(), desc="Queries: "):
            candidates_lst = []
            while not docqueue.empty():
                score, docid = docqueue.get_nowait()
                candidates_lst.append((score, docid))
            random.shuffle(candidates_lst)
            candidates_lst = sorted(candidates_lst, key=lambda x: x[0], reverse=True)
            for rank_idx, (score, doc_id) in enumerate(candidates_lst):
                outFile.write("{}\t{}\t{}\t{}\n".format(qid, doc_id, rank_idx + 1, score))


# def rerank(args):
#     logger.info("Loading document embeddings memmap ...")
#     doc_embedding_memmap, doc_id_memmap = get_embed_memmap(args.doc_embedding_dir, args.embedding_dim)
#     assert np.all(doc_id_memmap == list(range(len(doc_id_memmap))))
#     # did2pos = {identity: i for i, identity in enumerate(doc_id_memmap)}  # unnecessary, according to previous line
#
#     logger.info("Loading query embeddings memmap ...")
#     query_embedding_memmap, query_id_memmap = get_embed_memmap(args.query_embedding_dir, args.embedding_dim)
#     qid2pos = {identity: i for i, identity in enumerate(query_id_memmap)}
#
#     logger.info("Loading candidate documents ...")
#     qid_to_candidate_passages = load_candidates(args.candidates_path)
#
#     if args.query_ids is None:
#         query_ids = qid_to_candidate_passages.keys()
#     else:  # read subset of (integer) query IDs from file
#         logger.info("Will use queries inside: {}".format(args.query_ids))
#         with open(args.query_ids, 'r') as f:
#             query_ids = {int(line.rstrip()) for line in f}
#         query_ids = qid_to_candidate_passages.keys() & query_ids
#
#     results_dict = {qid: PriorityQueue(maxsize=args.hit) for qid in query_ids}
#
#     logger.info("Retrieving documents ...")
#
#     for qid in tqdm(query_ids, desc="query"):
#         query_embedding = query_embedding_memmap[qid2pos[qid]]
#         query_embedding = torch.from_numpy(query_embedding)
#         query_embedding = query_embedding.to(args.device)
#
#         doc_ids = np.array(qid_to_candidate_passages[qid], dtype=int)
#         doc_embeddings = doc_embedding_memmap[doc_ids]  # doc_ids are at the same time integer positions for doc_embedding_memmap
#         doc_embeddings = torch.from_numpy(doc_embeddings).to(args.device)
#
#         all_scores = torch.sum(query_embedding * doc_embeddings, dim=-1)
#
#         k = min(args.hit, len(doc_embeddings))
#         top_scores, top_indices = torch.topk(all_scores, k, largest=True, sorted=True)
#         top_scores, top_indices = top_scores.cpu(), top_indices.cpu()
#         top_doc_ids = doc_ids[top_indices.numpy()]
#         cur_q_queue = results_dict[qid]
#         for score, docid in zip(top_scores, top_doc_ids):
#             score, docid = score.item(), docid.item()
#             if cur_q_queue.full():
#                 lowest_score, lowest_docid = cur_q_queue.get_nowait()
#                 if lowest_score >= score:
#                     cur_q_queue.put_nowait((lowest_score, lowest_docid))
#                     break
#                 else:
#                     cur_q_queue.put_nowait((score, docid))
#             else:
#                 cur_q_queue.put_nowait((score, docid))
#
#     score_path = f"{args.output_path}.score"
#     with open(score_path, 'w') as outputfile:
#         for qid, docqueue in results_dict.items():
#             while not docqueue.empty():
#                 score, docid = docqueue.get_nowait()
#                 outputfile.write(f"{qid}\t{docid}\t{score}\n")
#     generate_rank(score_path, args.output_path)


def rerank3(args):
    """Reranks existing candidates per query in a qID -> ranked cand. list memmap"""
    logger.info("Loading document embeddings memmap ...")
    doc_embedding_memmap, doc_id_memmap = get_embed_memmap(args.doc_embedding_dir, args.embedding_dim)
    # assert np.all(doc_id_memmap == list(range(len(doc_id_memmap))))  # only for MSMARCO
    did2pos = {identity: i for i, identity in enumerate(doc_id_memmap)}  # un/necessary, depending on previous line
    doc_embedding_memmap = np.array(doc_embedding_memmap)

    logger.info("Loading query embeddings memmap ...")
    query_embedding_memmap, query_id_memmap = get_embed_memmap(args.query_embedding_dir, args.embedding_dim)
    qid2pos = {identity: i for i, identity in enumerate(query_id_memmap)}
    query_embedding_memmap = np.array(query_embedding_memmap)

    logger.info("Loading candidate documents ...")
    qid_to_candidate_passages = load_candidates(args.candidates_path)

    if args.query_ids is None:
        query_ids = qid_to_candidate_passages.keys()
    else:  # read subset of (integer) query IDs from file
        logger.info("Will use queries inside: {}".format(args.query_ids))
        with open(args.query_ids, 'r') as f:
            query_ids = {int(line.split()[0]) for line in f}
        query_ids = qid_to_candidate_passages.keys() & query_ids

    logger.info("{} queries found".format(len(query_ids)))

    if args.qrels_path:
        logger.info("Loading ground truth documents (labels) in '{}' ...".format(args.qrels_path))
        qrels = load_qrels(args.qrels_path)  # dict: {qID : set of ground truth relevant pIDs}

    logger.info("Retrieving documents ...")

    with open(args.output_path, 'w') as outputfile:
        for qid in tqdm(query_ids, desc="query"):
            start_time = time.time()
            query_embedding = query_embedding_memmap[qid2pos[qid]]
            query_embedding = torch.from_numpy(query_embedding)

            doc_ids = qid_to_candidate_passages[qid]

            if args.qrels_path:
                rel_docs = qrels[qid]
                # prepend relevant documents at the beginning of doc_ids, whether pre-existing in doc_ids or not,
                # while ensuring that they are only included once
                num_candidates = len(doc_ids)
                new_doc_ids = (list(rel_docs) + [docid for docid in doc_ids if docid not in rel_docs])[:num_candidates]
                doc_ids = new_doc_ids  # direct assignment wouldn't work in line above

            doc_ids = np.array(doc_ids, dtype=int)

            doc_embeddings = doc_embedding_memmap[[did2pos[docid] for docid in doc_ids]]
            doc_embeddings = torch.from_numpy(doc_embeddings)  # (num_cands, emb_dim)
            logger.debug("get embeddings: {:.3f} s".format(time.time() - start_time))

            start_time = time.time()
            # all_scores = query_embedding.dot(doc_embeddings)
            # all_scores = torch.from_numpy(all_scores)
            all_scores = torch.sum(query_embedding * doc_embeddings, dim=-1)
            logger.debug("all_scores: {:.3f} s".format(time.time() - start_time))

            start_time = time.time()
            k = min(args.hit, len(doc_ids))
            top_scores, top_indices = torch.topk(all_scores, k, largest=True, sorted=True)
            top_doc_ids = doc_ids[top_indices.numpy()]
            logger.debug("get top k: {:.3f} s".format(time.time() - start_time))

            start_time = time.time()
            for i, docid in enumerate(top_doc_ids):
                outputfile.write(f"{qid}\t{docid}\t{i+1}\t{top_scores[i]}\n")
            logger.debug("write candidates to file: {:.3f} s".format(time.time() - start_time))


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
        for line in tqdm(f, desc="Query"):
            try:
                fields = line.strip().split('\t')
                qid = int(fields[0])
                pid = int(fields[1])

                qid_to_candidate_passages[qid].append(pid)
            except Exception as x:
                print(x)
                logger.warning("Line \"{}\" is not in valid format and resulted in: {}".format(line, fields))
    return qid_to_candidate_passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Retrieval (for 1 GPU) based on precomputed query and document embeddings.")

    ## Required parameters
    parser.add_argument("--per_gpu_doc_num", default=4000000, type=int,
                        help="Number of documents to be loaded on the single GPU. Set to 4e6 for ~12GB GPU memory. "
                             "Reduce number in case of insufficient GPU memory.")
    parser.add_argument("--hit", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--output_path", type=str,
                        help="File path where to write the predictions/ranked candidates.")
    parser.add_argument("--doc_embedding_dir", type=str,
                        help="Directory containing the memmap files corresponding to document embeddings.")
    parser.add_argument("--query_embedding_dir", type=str,
                        help="Directory containing the memmap files corresponding to query embeddings.")
    parser.add_argument("--query_ids", type=str, default=None,
                        help="A text file containing query IDs (and possibly other fields, separated by whitespace), "
                             "one per line. If provided, will limit retrieval to this subset.")
    parser.add_argument("--candidates_path", type=str, default=None,
                        help="""If specified, will rerank candidate (retrieved) documents/passages given in a text a file. 
                        Assumes that retrieved documents per query are given one per line, in the order of rank 
                        (most relevant first) in the first 2 columns (ignores rest columns) as 
                        "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID)""")
    parser.add_argument("--qrels_path", type=str, default=None,
                        help="""If specified, will inject the ground truth relevant documents into the set of candidate 
                        documents, even if they weren't part of the original candidates.rerank candidate (retrieved) documents/passages given in a text a file. 
                        Assumes that retrieved documents per query are given one per line, in the order of rank 
                        (most relevant first) in the first 2 columns (ignores rest columns) as 
                        "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID)""")
    args = parser.parse_args()

    print(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        logger.warning("Found {} GPUs, but only a single GPU will be used by this program.".format(n_gpu))

    args.device = device

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    if os.path.isdir(args.output_path):
        raise IOError("Option `output_path` should be a file path, not directory path")

    with torch.no_grad():
        if args.candidates_path:
            rerank3(args)
        else:
            allrank(args)

    logger.info("Done!")
