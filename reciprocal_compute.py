import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

logger.info("Loading packages ...")

import sys
import os
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time

from beir.retrieval.evaluation import EvaluateRetrieval

import utils
from reciprocal_neighbors import pairwise_similarities, compute_rNN_matrix, compute_jaccard_sim, combine_similarities, local_query_expansion

embed_load_times = utils.Timer()
pwise_times = utils.Timer()
topk_times = utils.Timer()
pwise_times = utils.Timer()
recipNN_times = utils.Timer()
query_exp_times = utils.Timer()
jaccard_sim_times = utils.Timer()
top_results_times = utils.Timer()
write_results_times = utils.Timer()
total_times = utils.Timer()


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
    # doc_chunk_size = sys.getsizeof(np.array(memmap[:docs_per_chunk]))/1024**2
    # logger.info("{} chunks of {} documents (total of {}), each of approx. size {} MB, "
    #             "will be loaded to the following device:".format(math.ceil(memmap.shape[0]/docs_per_chunk), docs_per_chunk,
    #                                                              memmap.shape[0], math.ceil(doc_chunk_size)))

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
        # logger.warning("This device could potentially support "
        #                "`per_gpu_doc_num` up to {}".format(math.floor(args.per_gpu_doc_num*free_mem/doc_chunk_size)))
    else:
        logger.info("CPU")


def rerank_reciprocal_neighbors(args):
    """Reranks existing candidates per query in a qID -> ranked cand. list memmap"""
    logger.info("Loading document embeddings memmap ...")
    doc_embedding_memmap, doc_id_memmap = get_embed_memmap(args.doc_embedding_dir, args.embedding_dim)
    did2pos = {str(identity): i for i, identity in enumerate(doc_id_memmap)}
    doc_embedding_memmap = np.array(doc_embedding_memmap)

    logger.info("Loading query embeddings memmap ...")
    query_embedding_memmap, query_id_memmap = get_embed_memmap(args.query_embedding_dir, args.embedding_dim)
    qid2pos = {str(identity): i for i, identity in enumerate(query_id_memmap)}
    query_embedding_memmap = np.array(query_embedding_memmap)

    logger.info("Loading candidate documents per query from: '{}'".format(args.candidates_path))
    qid_to_candidate_passages = load_candidates(args.candidates_path)

    if args.query_ids is None:
        query_ids = qid_to_candidate_passages.keys()
    else:  # read subset of (integer) query IDs from file
        logger.info("Will use queries inside: {}".format(args.query_ids))
        with open(args.query_ids, 'r') as f:
            query_ids = {line.split()[0] for line in f}
        logger.info("{} queries found".format(len(query_ids)))
        query_ids = qid_to_candidate_passages.keys() & query_ids

    logger.info("Total queries to evaluate: {}".format(len(query_ids)))

    if args.qrels_path:
        logger.info("Loading ground truth documents (labels) in '{}' ...".format(args.qrels_path))
        qrels = load_qrels(args.qrels_path, relevance_level=args.relevance_thr, score_mapping=None)  # dict: {qID: {passageid: g.t. relevance}}

    logger.info("Reranking candidates ...")

    results = {}  # final predictions dict: {qID: {passageid: score}}. Used for metrics calculation through pytrec_eval
    out_rankfile =  open(args.output_path, 'w')
    
    for qid in tqdm(query_ids, desc="query "):
        start_total = time.perf_counter()
        start_time = start_total
        doc_ids = qid_to_candidate_passages[qid]  # list of string IDs

        if args.qrels_path and args.inject_ground_truth:
            rel_docs = qrels[qid].keys()
            # prepend relevant documents at the beginning of doc_ids, whether pre-existing in doc_ids or not,
            # while ensuring that they are only included once
            num_candidates = len(doc_ids)
            new_doc_ids = (list(rel_docs) + [docid for docid in doc_ids if docid not in rel_docs])[:num_candidates]
            doc_ids = new_doc_ids  # direct assignment wouldn't work in line above

        doc_ids = np.array(doc_ids)  # string IDs

        doc_embeddings = doc_embedding_memmap[[did2pos[docid] for docid in doc_ids]]
        doc_embeddings = torch.from_numpy(doc_embeddings).float().to(args.device)  # (num_cands, emb_dim)
        
        query_embedding = query_embedding_memmap[qid2pos[qid]]
        query_embedding = torch.from_numpy(query_embedding).float().to(args.device)
        
        global embed_load_times
        embed_load_times.update(time.perf_counter() - start_time)

        start_time = time.perf_counter()
        pwise_sims = pairwise_similarities(torch.cat((query_embedding.unsqueeze(0), doc_embeddings), dim=0), normalize='max') # (num_cands+1, num_cands+1)
        global pwise_times
        pwise_times.update(time.perf_counter() - start_time)

        final_sims = recompute_similarities(pwise_sims, k=args.k, trust_factor=args.trust_factor, k_exp=args.k_exp, orig_coef=args.sim_mixing_coef, device=args.device)[1:]  # (num_cands,)

        # Final selection of top candidates
        start_time = time.perf_counter()
        num_retrieve = min(args.hit, len(doc_ids))
        top_scores, top_indices = torch.topk(final_sims, num_retrieve, largest=True, sorted=True)
        top_doc_ids = doc_ids[top_indices.cpu()]
        top_scores = top_scores.cpu().numpy()
        global top_results_times
        top_results_times.update(time.perf_counter() - start_time)
        
        results[qid] = {str(docid): float(top_scores[i]) for i, docid in enumerate(top_doc_ids)}
        
        # Write new ranks to file
        start_time = time.perf_counter()
        for i, docid in enumerate(top_doc_ids):
            out_rankfile.write(f"{qid}\t{docid}\t{i+1}\t{top_scores[i]}\n")
        global write_results_times
        write_results_times.update(time.perf_counter() - start_time)
        
        global total_times
        total_times.update(time.perf_counter() - start_total)
    
    out_rankfile.close()

    if args.qrels_path:
        if len(query_ids) < len(qrels):
            logger.warning(f"File '{args.qrels_path}' contains rel. labels for {len(qrels)} queries, while only {len(query_ids)} "
                        "queries were evaluated. Performance metrics will only consider the intersection.")
            qrels = {qid: qrels[qid] for qid in query_ids}  # trim qrels dict
        
        logger.info("Computing metrics ...")
        start_time = time.perf_counter()
        # Evaluate using BEIR (which relies on pytrec_eval) for comparison with BEIR benchmarks
        # Returns dictionaries with metrics for each cut-off value, e.g. ndcg["NDCG@{k}".format(cutoff_values[0])] == 0.3
        cutoff_values = [1, 3, 5, 10, 100, 1000]
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, cutoff_values)
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, cutoff_values, 'MRR')
        metrics_time = time.perf_counter() - start_time
        logger.info("Time to calculate performance metrics: {:.3f}".format(metrics_time))
        
        print(mrr)
        print(ndcg)
        print(_map)
        print(recall)
        print(precision)
        
    return


def smoothen_relevance_labels():
    pass


def recompute_similarities(pwise_sims, k=20, trust_factor=0.5, k_exp=6, orig_coef=0.3, device=None):
    """Compute new similarities with respect to a probe (query) based on its reciprocal nearest neighbors Jaccard similarity 
    with the geometric Nearest Neibors, as well as geometric similarities. Assumes similarities, not distances.

    :param pwise_sims: (m, m) tensor, where m = N + 1. Geometric similarities between all m*m pairs of emb. vectors in the set {query, doc1, ..., doc_N}
    :param k: number of Nearest Neighbors, defaults to 20
    :param k_exp: k used for query expansion. No expansion takes place with k<=1. Defaults to 6
    :param orig_coef: float in [0, 1]. If > 0, this will be the coefficient of the original geometric similarities (in `pwise_sims`)
                        when computing the final similarities. 
    :return: (m,) tensor, updated similarities. Includes self-similarity at index 0
    """

    global topk_times, recipNN_times, query_exp_times, jaccard_sim_times

    start_time = time.perf_counter()
    initial_rank = torch.topk(pwise_sims, k+1, dim=1, largest=True, sorted=True).indices  # (m, k+1) tensor of indices corresponding to largest values in each row of pwise_sims
    topk_times.update(time.perf_counter() - start_time)

    # Compute sparse reciprocal neighbor vectors for each item (i.e. reciprocal adjecency matrix)
    start_time = time.perf_counter()
    V = compute_rNN_matrix(pwise_sims, initial_rank, k=k, trust_factor=trust_factor, overlap_factor=2/3, weight_func='exp', param=2.4, device=device) # (m, m) float tensor, (sparse) adjacency matrix
    recipNN_times.update(time.perf_counter() - start_time)

    if k_exp > 1:
        start_time = time.perf_counter()
        V = local_query_expansion(V, initial_rank, k_exp, device=device)  # (m, m)
        query_exp_times.update(time.perf_counter() - start_time)
    
    start_time = time.perf_counter()
    jaccard_sims = compute_jaccard_sim(V)  # (m,)
    jaccard_sim_times.update(time.perf_counter() - start_time)

    final_sims = combine_similarities(pwise_sims[0, :], jaccard_sims, orig_coef=orig_coef)  # (m,) includes self-similarity at index 0

    return final_sims


def load_qrels(filepath, relevance_level=1, score_mapping=None):
    """Load ground truth relevant passages from file. Can handle several levels of relevance.
    Assumes that if a passage is not listed for a query, it is non-relevant.
    :param filepath: path to file of ground truth relevant passages in the following format:
        "qID1 \t Q0 \t pID1 \t 2\n
         qID1 \t Q0 \t pID2 \t 0\n
         qID1 \t Q0 \t pID3 \t 1\n..."
    :param relevance_level: only include candidates which have at least the specified relevance score
        (after potential mapping)
    :param score_mapping: dictionary mapping relevance scores in qrels file to a different value (e.g. 2 -> 0.3)
    :return:
        qid2relevance (dict): dictionary mapping from query_id to relevant passages (dict {passageid : relevance})
    """
    qid2relevance = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            try:
                qid, _, pid, relevance = line.strip().split()
                relevance = int(relevance)
                if (score_mapping is not None) and (relevance in score_mapping):
                    relevance = score_mapping[relevance]  # map score to new value
                if relevance >= relevance_level:  # include only if score >= specified relevance level
                    qid2relevance[qid][pid] = relevance
            except Exception as x:
                print(x)
                raise IOError("'{}' is not valid format".format(line))
    return qid2relevance


def load_candidates(path_to_candidates):
    """
    Load candidate (retrieved) documents/passages from a file.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns (ignores rest columns) as "qID1 \t pID1\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).
    :param path_to_candidates: path to file of candidate (retrieved) documents/passages per query
    :return:
        qid_to_candidate_passages: dict: {qID : list of retrieved pIDs in order of relevance}
    """

    qid_to_candidate_passages = defaultdict(list)  # dict: {qID : list of retrieved pIDs in order of relevance}

    with open(path_to_candidates, 'r') as f:
        for line in tqdm(f, desc="Query"):
            try:
                fields = line.strip().split('\t')
                qid = fields[0]
                pid = fields[1]

                qid_to_candidate_passages[qid].append(pid)
            except Exception as x:
                print(x)
                logger.warning("Line \"{}\" is not in valid format and resulted in: {}".format(line, fields))
    return qid_to_candidate_passages


def run_parse_args():
    parser = argparse.ArgumentParser("Retrieval (for 1 GPU) based on precomputed query and document embeddings.")

    ## Required parameters
    parser.add_argument("--task", choices=['rerank', 'label_smoothing'], default='rerank')
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda')
    # parser.add_argument("--per_gpu_doc_num", default=4000000, type=int,
    #                     help="Number of documents to be loaded on the single GPU. Set to 4e6 for ~12GB GPU memory. "
    #                          "Reduce number in case of insufficient GPU memory.")
    parser.add_argument("--hit", type=int, default=1000, 
                        help="Number of retrieval results (ranked candidates) for each query.")
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
    parser.add_argument("--inject_ground_truth", action='store_true',
                        help="If true, the ground truth document(s) will be injected into the set of documents "
                             "to be reranked, even if they weren't part of the original candidates.")
    parser.add_argument("--qrels_path", type=str, default=None,
                        help="Path to file of ground truth relevant passages in the following format: 'qID1 \t Q0 \t pID1 \t 1\n qID1 \t Q0 \t pID2 \t 1\n ...)'")
    parser.add_argument('--relevance_thr', type=float, default=1.0, 
                        help="Score threshold in qrels (g.t. relevance judgements) to consider a document relevant.")
    parser.add_argument('--k', type=int, default=20, 
                        help="Number of Nearest Neighbors in terms of similarity. Used in finding Reciprocal Nearest Neighbors.")
    parser.add_argument('--trust_factor', type=float, default=0.5,
                        help="If > 0, will build an extended set of reciprocal neighbors, by considering neighbors of neighbors. "
                        "The number of nearest reciprocal neighbors to consider for each k-reciprocal neighbor is trust_factor*k")
    parser.add_argument('--k_exp', type=int, default=6, 
                        help="Number of Nearest Neighbors to consider when performing 'local query expansion' of "
                        "Reciprocal NN sparse vectors.")
    parser.add_argument('--sim_mixing_coef', type=float, default=0.3,
                        help="Coefficient of geometric similarity when linearly mixing it with Jaccard similarity based on Recip. NN")
    args = parser.parse_args()

    # Setup CUDA, GPU 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        logger.warning("Found {} GPUs, but only a single GPU will be used by this program.".format(n_gpu))

    args.device = device

    # Log current hardware setup
    logger.info("Device: %s, n_gpu: %s", args.device, n_gpu)
    if args.device.type == 'cuda':
        logger.info("Device: {}".format(torch.cuda.get_device_name(0)))
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        logger.info("Total memory: {} MB".format(int(total_mem)))

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    if os.path.isdir(args.output_path):
        raise IOError("Option `output_path` should be a file path, not directory path")
    
    return args
    

if __name__ == "__main__":
    args = run_parse_args()

    with torch.no_grad():
        if args.task == 'rerank':
            rerank_reciprocal_neighbors(args)
        else:
            raise NotImplementedError(f'Task {args.task} not implemented!')

    logger.info("AVG. TOTAL TIME per query: {} sec".format(total_times.get_average()))
    logger.info("Avg. time to get embeddings related to a query: {} sec".format(embed_load_times.get_average()))
    logger.info("Avg. pairwise sim. comp. time per query: {} sec".format(pwise_times.get_average()))
    logger.info("Avg. top-k time per query: {} sec".format(topk_times.get_average()))
    logger.info("Avg. recip. NN time per query: {} sec".format(recipNN_times.get_average()))
    logger.info("Avg. query expansion time per query: {} sec".format(query_exp_times.get_average()))
    logger.info("Avg. Jaccard sim. time per query: {} sec".format(jaccard_sim_times.get_average()))
    logger.info("Avg. time to get top results per query: {} sec".format(top_results_times.get_average()))
    logger.info("Avg. time to write results to file per query: {} sec".format(write_results_times.get_average()))
    logger.info("Done!")
