"""
Adapted by gzerveas from:
https://gist.github.com/bwhite/3726239
"""

import numpy as np
from typing import List, Tuple, Dict
import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def mrr(qrels, results, k_values, relevance_level=1, verbose=False) -> Tuple[Dict[str, float]]:
    """Aggregate (mean) MRR calculation at multiple cut-offs given ground truth and prediction dictionaries.

    :param qrels: Dict[query_id, Dict[pasasge_id, relevance_score]] ground truth
    :param results: Dict[query_id, Dict[pasasge_id, relevance_score]] predictions
    :param k_values: iterable of integer cut-off thresholds
    :param relevance_level: relevance score in qrels which a doc should at least have in order to be considered relevant
    :return: Dict[str, float] value for each metric (determined by `k_values`) 
    """
    
    MRR = {}
    
    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] >= relevance_level])    
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break
    
    if verbose: print('\n')
    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"]/len(qrels), 5)
        if verbose:
            logger.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def reciprocal_rank(ground_truth, pred, max_docs=None):
    """
    Args:
        ground_truth: dict {pid:relevance}
        pred: list of candidate pids
    """
    if max_docs is None:
        max_docs = len(pred)
    reciprocal_rank = 0
    for i in range(max_docs):
        if pred[i] in ground_truth:
            reciprocal_rank = 1/(i + 1)
            break  # because we only care for 1st (highest) result
    return reciprocal_rank


def mean_reciprocal_rank(rs, k=None):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: iterable of ground truth relevance scores (which are list or numpy) per query in rank order

    Returns:
        Mean reciprocal rank
    """
    if k is None:
        # iterable of: Tuple(array of indices of non-zero elements for each query, )
        rs = (np.asarray(r).nonzero()[0] for r in rs)
    else:  # same, but considers only first k
        rs = (np.asarray(r)[:min(k, len(r))].nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])  # here r[0] selects index of first non-zero element


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def recall_at_k(rs, num_relevant, k=None, fixed_num_candidates=False):
    """
    Every level of relevance counts the same here, i.e. finding a doc with rel 1 contributes as much as a doc with 3.
    :param rs: (num_queries) list of (num_docs) list of ground truth relevances of ranked candidates for each query.
    :param num_relevant: (num_queries) list of int number of ground truth relevant documents
    :param k: consider only top k candidates
    :param fixed_num_candidates: whether each query has the same number of candidates
    :return: mean recall at k for rs
    """
    if fixed_num_candidates:
        relevances = np.asarray(rs)
        if k is not None:
            relevances = relevances[:, :k]
        num_found = (relevances > 0).sum(axis=1)
        return np.mean(num_found / np.array(num_relevant))
    else:
        if k is None:
            return np.mean([np.sum(np.array(rs[i]) > 0) / num_relevant[i] for i in range(len(rs))])
        else:
            return np.mean([np.sum(np.array(rs[i][:k]) > 0) / num_relevant[i] for i in range(len(rs))])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (approx. area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / len(r)


def mean_average_precision(rs, k=None):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    if k is None:
        return np.mean([average_precision(r) for r in rs])
    else:
        return np.mean([average_precision(r[:min(k, len(r))]) for r in rs])


def mean_average_precision_at_r(rs, num_relevant):
    """
    IMPORTANT: This metric will be highly underestimated if len(rs) (i.e. number of scored candidates) < num_relevant.
    This can happen in certain datasets with many ground truth relevant samples for each query (e.g. SOP superclass).
    Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 0]]
    >>> mean_average_precision(rs, [2])
    1.0
    >>> rs = [[1, 1, 0, 0]]
    >>> mean_average_precision(rs, [3])
    0.67
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
        num_relevant: list of total positive samples for each query in the entire dataset (class_counts['class_name']-1)
    Returns:
        Mean average precision at R
    """
    return np.mean([average_precision(r[:min(k, len(r))]) for (r, k) in zip(rs, num_relevant)])


def dcg_at_k(relevance, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        relevance: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    relevance = np.asfarray(relevance)[:k]
    if relevance.size:
        if method == 0:
            return relevance[0] + np.sum(relevance[1:] / np.log2(np.arange(2, relevance.size + 1)))
        elif method == 1:
            return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=None, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    if k is None:
        k = len(r)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
