import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.info("Loading packages ...")
import random
import string
import os
import math
from collections import OrderedDict, defaultdict
import time
from datetime import datetime
import argparse

import torch
import numpy as np
from tqdm import tqdm

import utils
from reciprocal_neighbors import pairwise_similarities, normalize, combine_similarities, compute_jaccard_similarities, topk_times, recipNN_times, query_exp_times, jaccard_sim_times

embed_load_times = utils.Timer()
pwise_times = utils.Timer()
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


def print_memory_info(device):
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


def load_ranked_candidates(path_to_candidates):
    """
    Load ranked/sorted candidate (retrieved) documents/passages from a file.
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

class ReciprocalNearestNeighbors(object):
    
    def __init__(self, query_embedding_dir, doc_embedding_dir, embedding_dim, candidates_path, 
                 qrels_path=None, query_ids_path=None, 
                 compute_only_for_qrels=True, inject_ground_truth=False, relevance_thr=1.0,
                 device='cpu', save_memory=False) -> None:
        """
        :param query_embedding_dir: _description_
        :param doc_embedding_dir: _description_
        :param embedding_dim: _description_
        :param candidates_path: _description_
        :param qrels_path: _description_, defaults to None
        :param query_ids_path: _description_, defaults to None
        :param compute_only_for_qrels: _description_, defaults to True
        :param inject_ground_truth: _description_, defaults to False
        :param relevance_thr: _description_, defaults to 1.0
        :param device: PyTorch device to run the computation. By default CPU.
        :param save_memory: _description_, defaults to False
        """
        # System
        self.device = device
        self.save_memory = save_memory
        # IO
        self.query_embedding_dir = query_embedding_dir
        self.doc_embedding_dir = doc_embedding_dir
        self.candidates_path = candidates_path
        self.qrels_path = qrels_path
        self.query_ids_path = query_ids_path
        # Setting
        self.compute_only_for_qrels = compute_only_for_qrels
        self.embedding_dim = embedding_dim
        self.inject_ground_truth = inject_ground_truth
        self.relevance_thr = relevance_thr
        
        self.load_from_files()
            
    def set(self, **kwargs):
        for key in kwargs:
            if key in self.__dict__:
                setattr(self, key, kwargs[key])
            else:
                raise KeyError(f"Key '{key}' not a memmber of '{self}'")
            
    def load_from_files(self):
        """Load document embeddings, query embeddings, candidate documents and qrels from files."""
        
        logger.info("Loading document embeddings memmap ...")
        self.doc_embedding_memmap, doc_id_memmap = get_embed_memmap(self.doc_embedding_dir, self.embedding_dim)
        self.did2pos = {str(identity): i for i, identity in enumerate(doc_id_memmap)}
        if not self.save_memory:
            self.doc_embedding_memmap = np.array(self.doc_embedding_memmap)

        logger.info("Loading query embeddings memmap ...")
        self.query_embedding_memmap, query_id_memmap = get_embed_memmap(self.query_embedding_dir, self.embedding_dim)
        self.qid2pos = {str(identity): i for i, identity in enumerate(query_id_memmap)}
        if not self.save_memory:
            self.query_embedding_memmap = np.array(self.query_embedding_memmap)

        logger.info("Loading candidate documents per query from: '{}'".format(self.candidates_path))
        self.qid_to_candidate_passages = load_ranked_candidates(self.candidates_path)
        self.query_ids = self.qid_to_candidate_passages.keys()

        if self.qrels_path:
            logger.info("Loading ground truth documents (labels) in '{}' ...".format(self.qrels_path))
            self.qrels = utils.load_qrels(self.qrels_path, relevance_level=self.relevance_thr, score_mapping=None)  # dict: {qID: {passageid: g.t. relevance}}
            
            if self.compute_only_for_qrels:
                intersection = self.query_ids & self.qrels.keys()
                if len(intersection) < len(self.query_ids):
                    logger.warning(f"Only {len(intersection)} of queries in {self.query_embedding_dir} are contained in qrels file, "
                                f"which contains {len(self.qrels)} queries."
                                "Computation will be limited to this smaller intersection set.")
                    self.query_ids = intersection
        else:
            self.qrels = None

        if self.query_ids_path is not None:  # read subset of (integer) query IDs from file
            logger.info("Will only use queries inside: {}".format(self.query_ids_path))
            with open(self.query_ids_path, 'r') as f:
                ext_query_ids = {line.split()[0] for line in f}
            logger.info("{} queries found".format(len(ext_query_ids)))
            self.query_ids = self.query_ids & ext_query_ids
        
        logger.info("Total queries to evaluate: {}".format(len(self.query_ids)))
        
        return

    def rerank(self, query_ids, hit=1000, 
               normalization='None', k=20, trust_factor=0.5, k_exp=6, weight_func='exp', weight_func_param=2.4, orig_coef=0.3):
        """Reranks existing candidates per query in a qID -> ranked cand. list .tsv file

        :param query_ids: iterable of query IDs for which to rerank candidates
        :param hit: int, number of (top) candidates to consider for each query
        :param normalization: str, how to normalize the original geometric similarity scores, defaults to 'None'
        :param k: int, number of Nearest Neighbors, defaults to 20
        :param trust_factor: If > 0, will build an extended set of reciprocal neighbors, by considering neighbors of neighbors.
                The number of reciprocal neighbors to consider for each k-reciprocal neighbor is trust_factor*k. Defaults to 1/2
        :param k_exp: int, k used for query expansion, i.e. how many Nearest Neighbors should be linearly combined to result in an expanded sparse vector (row). 
                        No expansion takes place with k<=1.
        :param weight_func: str, function mapping similarities to weights, defaults to 'exp'.
                            When not 'exp', uses similarities themselves as weights (proportional weighting) 
                            If None, returns binary adjacency matrix, without weighting based on geometric similarity.
        :param weight_func_param: parameter of the weight function. Only used when `weight_func` is 'exp'.
        :param orig_coef: float in [0, 1]. If > 0, this will be the coefficient of the original geometric similarities (in `pwise_sims`)
                            when computing the final similarities.
        :return reranked_scores: final/reranked predictions dict: {qID: OrderedDict{passageid: score}}
        :return orig_scores: original predictions dict: {qID: OrderedDict{passageid: score}}
        """        
        
        orig_scores = {}  # original predictions dict: {qID: OrderedDict{passageid: score}}
        reranked_scores = {}  # final/reranked predictions dict: {qID: OrderedDict{passageid: score}}. Used for metrics calculation through pytrec_eval
        
        for qid in tqdm(query_ids, desc="query "):
            start_total = time.perf_counter()
            start_time = start_total
            
            doc_ids = self.qid_to_candidate_passages[qid]  # list of string IDs
            max_candidates = min(hit, len(doc_ids))
            doc_ids = doc_ids[:max_candidates]

            if self.qrels and self.inject_ground_truth:
                rel_docs = self.qrels[qid].keys()
                # prepend relevant documents at the beginning of doc_ids, whether pre-existing in doc_ids or not,
                # while ensuring that they are only included once
                new_doc_ids = (list(rel_docs) + [docid for docid in doc_ids if docid not in rel_docs])[:max_candidates]
                doc_ids = new_doc_ids  # direct assignment wouldn't work in line above

            doc_ids = np.array(doc_ids)  # string IDs

            doc_embeddings = self.doc_embedding_memmap[[self.did2pos[docid] for docid in doc_ids]]
            doc_embeddings = torch.from_numpy(doc_embeddings).float().to(self.device)  # (num_cands, emb_dim)
            
            query_embedding = self.query_embedding_memmap[self.qid2pos[qid]]
            query_embedding = torch.from_numpy(query_embedding).float().to(self.device)
            
            global embed_load_times
            embed_load_times.update(time.perf_counter() - start_time)

            start_time = time.perf_counter()
            pwise_sims = pairwise_similarities(torch.cat((query_embedding.unsqueeze(0), doc_embeddings), dim=0)) # (num_cands+1, num_cands+1)
            pwise_sims = normalize(pwise_sims, normalization) # (num_cands+1, num_cands+1)
            global pwise_times
            pwise_times.update(time.perf_counter() - start_time)
            
            orig_scores[qid] = OrderedDict((docid, float(pwise_sims[0, 1 + i])) for i, docid in enumerate(doc_ids))

            jaccard_sims = compute_jaccard_similarities(pwise_sims, k=k, trust_factor=trust_factor, k_exp=k_exp,
                                                        weight_func=weight_func, weight_func_param=weight_func_param, 
                                                        device=self.device)  # (num_cands+1,) includes self-similarity at index 0
            
            # top_scores, top_indices = torch.topk(jaccard_sims[1:], max_candidates, largest=True, sorted=True)
            # top_doc_ids = doc_ids[top_indices.cpu()]
            # top_scores = top_scores.cpu().numpy()
            
            final_sims = combine_similarities(pwise_sims[0, :], jaccard_sims, orig_coef=orig_coef)[1:]  # (num_cands,) 

            # Final selection of top candidates
            start_time = time.perf_counter()
            top_scores, top_indices = torch.topk(final_sims, max_candidates, largest=True, sorted=True)
            top_doc_ids = doc_ids[top_indices.cpu()]
            top_scores = top_scores.cpu().numpy()
            global top_results_times
            top_results_times.update(time.perf_counter() - start_time)
            
            reranked_scores[qid] = OrderedDict((docid, float(top_scores[i])) for i, docid in enumerate(top_doc_ids))
            
            global total_times
            total_times.update(time.perf_counter() - start_total)
        
        return reranked_scores, orig_scores


def recip_NN_rerank(args):
    """Reranks existing candidates per query in a qID -> ranked cand. list .tsv file.

    :param args: arguments object, as returned by argparse
    :return perf_metrics: dict {metric_name: metric_value}
    """
    
    rNN_reranker = ReciprocalNearestNeighbors(args.query_embedding_dir, args.doc_embedding_dir, args.embedding_dim, args.candidates_path,
                                              args.qrels_path, args.query_ids_path, args.compute_only_for_qrels, args.inject_ground_truth, args.relevance_thr,
                                              args.device, args.save_memory)
    
    logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
    logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))
    
    logger.info("Reranking candidates ...")
    reranked_scores, _ = rNN_reranker.rerank(rNN_reranker.query_ids, args.hit,
                                             args.normalize, args.k, args.trust_factor, args.k_exp, args.weight_func, args.weight_func_param, args.orig_coef)
    
    if args.write_scores_to_file:
        # Write new ranks to file
        with open(args.out_rankfilepath, 'w') as out_rankfile:
            start_time = time.perf_counter()
            for qid, doc2score in reranked_scores.items():
                for i, (docid, score) in enumerate(doc2score.items()):
                    out_rankfile.write(f"{qid}\t{docid}\t{i+1}\t{score}\n")
                global write_results_times
                write_results_times.update(time.perf_counter() - start_time)

    perf_metrics = None
    if args.qrels_path:
        perf_metrics = utils.get_retrieval_metrics(reranked_scores, rNN_reranker.qrels)
        perf_metrics['time'] = total_times.get_average()
        # NOTE: Order of keys is guaranteed in Python 3.7+ (but informally also in 3.6)
        parameters = {'sim_mixing_coef': args.sim_mixing_coef,
                      'k': args.k,
                      'trust_factor': args.trust_factor,
                      'normalize': args.normalize,
                      'weight_func': args.weight_func,
                      'weight_func_param': args.weight_func_param}
        
        # Export record metrics to a file accumulating records from all experiments
        utils.register_record(args.records_file, args.formatted_timestamp, args.out_rankfilename, perf_metrics, parameters=parameters)
        
    return perf_metrics


def smoothen_relevance_labels():
    pass


def setup(args):
    
    # Create prefix and output file name
    initial_timestamp = datetime.now()
    args.formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    args.out_rankfilename = args.formatted_timestamp + "_" + rand_suffix
    if args.exp_name:
        args.out_rankfilename += "_{}".format(args.exp_name)
    args.out_rankfilename += "_rerank_" + os.path.basename(args.candidates_path)
    
    args.out_rankfilepath = os.path.join(args.output_dir, args.out_rankfilename)
    
    return args


def run_parse_args():
    parser = argparse.ArgumentParser("Retrieval (for 1 GPU) based on precomputed query and document embeddings.")

    ## Required parameters
    parser.add_argument("--task", choices=['rerank', 'label_smoothing'], default='rerank')
    parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--save_memory", action="store_true",
                        help="If set, embeddings will be loaded from memmaps and not entirely preloaded to memory. "
                        "Saves much memory (approx. 2GB vs 50GB) at the expense of approx. x2 time")
    parser.add_argument("--hit", type=int, default=1000, 
                        help="Number of top retrieval results (ranked candidates) to consider for each query when reranking.")
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--output_dir", type=str, default='.',
                        help="Directory path where to write the predictions/ranked candidates file.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Characteristic string describing the experiment. "
                        " This is going to be appended after the timestamp and random prefix, and before the original filename.")
    parser.add_argument('--records_file', default='./records.xls', 
                        help='Excel file keeping best records of all experiments')
    parser.add_argument("--write_scores_to_file", type=bool, default=True,
                        help="If set, predictions (scores per document for each query) will be written to a .tsv file")
    parser.add_argument("--doc_embedding_dir", type=str,
                        help="Directory containing the memmap files corresponding to document embeddings.")
    parser.add_argument("--query_embedding_dir", type=str,
                        help="Directory containing the memmap files corresponding to query embeddings. "
                        "By default, all queries found within these files will be used to compute candidate scores.")
    parser.add_argument("--query_ids", dest='query_ids_path', type=str, default=None,
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
    parser.add_argument("--compute_only_for_qrels", type=bool, default=True,
                        help="If true, and a `qrels_path` is provided, then scores will be computed only for queries which also exist "
                        "in the qrels file")
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
    parser.add_argument('--normalize', type=str, choices=['max', 'mean', 'None'], default='None',
                        help="How to normalize values. It is *extremely* important to consider what function is used to map similarities to weights, i.e. `weight_func`")
    parser.add_argument('--weight_func', type=str, choices=['linear', 'exp'], default='linear',
                        help="How to weight (potentially normalized) geometric similarities when building sparse neighbors vector.")
    parser.add_argument('--weight_func_param', type=float, default=1.0,
                        help="Parameter of weight function (used when function is exponential)")
    #args = parser.parse_args()

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(args.output_dir):
        raise IOError("Option `output_dir` should be a directory path, not a file path")
    
    return args
    

if __name__ == "__main__":
    args = run_parse_args()
    args = setup(args)

    total_start_time = time.time()
    with torch.no_grad():
        if args.task == 'rerank':
            perf_metrics = recip_NN_rerank(args)
            print(perf_metrics)
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
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))
    logger.info("Done!")
