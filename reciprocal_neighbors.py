"""
Algorithm loosely based on code from Zhong et al., 2017: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np
import torch

# class ReciprocalNeighbors(object):
#     def __init__(self, pwise_sims=None) -> None:
#         pass

def pairwise_similarities(vectors, type='dot_product', normalize=None):
    """Computes similarities between all pairs of vectors in a tensor `vectors`.

    :param vectors: (m, d) float tensor, a sequence of m vectors
    :param type: str, function to compute similarity, defaults to 'dot_product'
    :param normalize: str, how to normalize values. It is *extremely* important to consider what function is used in `assign_neighbor_weights`
        to map similarities to weights (e.g. an exponential function when using unnormalized values may lead to NaN)
    :return: (m, m) float tensor, similarity matrix between all m*m pairs of elements in `vectors`
    """

    if type == 'dot_product':
        similarities = torch.mm(vectors, vectors.T)
    elif type == 'cosine':
        similarities = torch.nn.functional.cosine_similarity(vectors, vectors[:, None, :], dim=2)  # stable
    else:
        raise NotImplementedError("Similarity '{}' not implemented".format(type))
    if normalize == 'max':
        similarities = similarities / similarities.max(dim=1).values.unsqueeze(1) # divides by the largest sim. per row
    elif normalize == 'mean':
        similarities = similarities / (1e-8 + similarities.mean(dim=1).unsqueeze(1)) # divides by the mean sim. per row
    
    return similarities

def k_reciprocal_NN(initial_rank, ind, k=None):
    """Return tensor of indices of k-Reciprocal Nearest Neighbors of an item with index `ind` within a table `initial_rank`, 
    which ranks for each item all other items by decreasing proximity. Includes `ind` as a neighbor.

    :param initial_rank: (m, k+1) int tensor, which for each item (row) ranks all other items by decreasing proximity;
        it is obtained by top-(k+1) on a (m, m) similarity matrix, equivalent to the k+1 first columns of argsort.
        Since the item itself is contained in each row, it is expected that initial_rank[i, 0] = i 
        (i.e. the nearest item is the item itself).
    :param ind: the index of the item in `initial_rank` for which we wish to obtain the reciprocal nearest neighbors.
    :param k: number of reciprocal nearest neighbors *excluding* item itself. If None, k = initial_rank.shape[1].
    :return: (r,) int tensor of indices of reciprocal neighbors, r in [1, k+1] (because the item itself is always returned)
    """

    end = initial_rank.shape[1] if k is None else (k + 1)

    forward_kNN_inds = initial_rank[ind, :end]  # (k+1,) int tensor of indices corresponding to k+1 NN of `ind``
    backward_kNN_inds = initial_rank[forward_kNN_inds, :end]  # (k+1, k+1) tensor: rows of initial_rank corresponding to NN of `ind`
    valid_rows = torch.any(backward_kNN_inds==ind, dim=1)  # (k+1,) boolean tensor of rows of backward_kNN_inds containing `ind`
    # valid_rows = torch.nonzero(backward_kNN_inds==ind)[:, 0]  # indices of rows of backward_kNN_inds where ind exists (equivalent)
    return forward_kNN_inds[valid_rows]

def boolean_vectorize(indices, length):
    """Converts a 1D tensor of integer indices into a sparse boolean tensor of length `length` 
    with True only at the index locations in `indices`.

    :param indices: (n,) tensor of integer indices
    :param length: int, length of boolean vector
    :return: (length,) boolean tensor of length `length` with True only at the index locations in `indices`.
    """
    return torch.zeros(length, dtype=bool).index_fill_(0, indices, True)

def extended_k_recip_NN(recip_neighbors, initial_rank, trust_factor=1/2, overlap_factor=2/3, out_type='bool'):
    """Extends a given set of k reciprocal neighbors of a probe by considering for each neighbor 
    a smaller ("trusted") set of its own k reciprocal neighbors and including them in case this tursted set 
    has significant overlap with the existing set of reciprocal neighbors.

    :param recip_neighbors: (r1,) int tensor, indices of a set of k-reciprocal nearest neighbors (of a probe)
    :param initial_rank: (m, k+1) int tensor, which for each item (row) ranks all other items by decreasing proximity;
        it is obtained by top-(k+1) on a (m, m) similarity matrix, equivalent to the k+1 first columns of argsort.
        Since the item itself is contained in each row, it is expected that initial_rank[i, 0] = i 
        (i.e. the nearest item is the item itself).
    :param trust_factor: the number of reciprocal neighbors to consider including for each k-reciprocal neighbor 
        is trust_factor*k. Defaults to 1/2
    :param overlap_factor: float in [0, 1], how much overlap with the original set of recip. neighbors the
        set of recip. neighbors of each of its members should have in order to be included. Defaults to 2/3
    :return: extended_rneighbors:
        if `out_type == 'bool'`: (m,) bool tensor, True only at indices of the extended set of reciprocal nearest neighbors
        else: (r2,) int tensor, indices of the extended set of reciprocal nearest neighbors
    """
    m = initial_rank.shape[0]  # total number of elements (num_candidates + 1 query)
    # NOTE: original implementation uses k without +1, i.e.: round(tf*k) + 1
    num_secondary_rNN = round(trust_factor * initial_rank.shape[1]) + 1  # num. rNN of each rNN. Should be trust_factor*k, +1 because item itself is included in neighbors

    rNN_bool = boolean_vectorize(recip_neighbors, m)  # (m,) boolean indicating original recip. NNs as True
    extended_rneighbors = rNN_bool.clone()  # (m,) boolean indicating extended rNN as True
    for rneighbor in recip_neighbors:
        rneighbor_candidates = boolean_vectorize(k_reciprocal_NN(initial_rank, rneighbor, num_secondary_rNN), m) # (m,) boolean indicating rNNs of rneighbor as True
        if (rneighbor_candidates & rNN_bool).sum() > overlap_factor * rneighbor_candidates.sum():  # if intersection of candidates with *original* rNNs is significant 
            extended_rneighbors = extended_rneighbors | rneighbor_candidates  # union of candidates with current rNNs

    if out_type != 'bool':
        extended_rneighbors = torch.nonzero(extended_rneighbors).squeeze()  # convert to integer indices
    return extended_rneighbors

def assign_neighbor_weights(neighbors, similarities, weight_func='exp', param=2.4):
    """Defines and applies function assigning weights to reciprocal neighbors based on their similarity to a probe.
    These weights are used e.g. in computing the Jaccard similarity or performing local query vector expansion.
    The type of function and its parameters should be chosen depending on the similarity measure and its normalization.
    Weights are normalized to sum to 1. 

    :param neighbors: (m,) bool tensor, True only at indices of the extended set of reciprocal nearest neighbors;
        or: (num_neighbors,) int tensor, indices of the extended set of reciprocal nearest neighbors
    :param similarities: (m,) tensor of similarities of vectors with respect to a probe (query) vector; similarities[0] is the self-similarity  
    :param weight_func: function mapping similarities to weights, defaults to 'exp'
    :param param: parameter of the weight function, defaults to 2.4
    :return: (num_neighbors,) float tensor, weights of neighbors according to their similaririty
    """

    if weight_func == 'exp':
        # 0 at similarity 0
        # param=2.4 means that when similarities are max normalized, the nearest vector (the query itself, sim=1.0)
        # will be weighted ~4 times higher than a vector with half the similarity (sim=0.5)
        weights = torch.exp(param * similarities[neighbors]) - 1 # (num_neighbors,) weight per neighbor
    else:
        weights = similarities[neighbors]  # uses similarities themselves as weights
    weights = weights / torch.sum(weights)    

    return weights

def compute_rNN_matrix(pwise_sims, initial_rank, k=20, trust_factor=0.5, overlap_factor=2/3, weight_func='exp', param=2.4):
    """
    Compute the reciprocal adjecency matrix, i.e. sparse reciprocal neighbor vectors 
    for each item corresponding to a row in `pwise_sims`.

    :param pwise_sims: (m, m) tensor, where m = N + 1. Similarities between all m*m pairs of emb. vectors in the set {query, doc1, ..., doc_N}
    :param initial_rank: (m, k+1) int tensor, which for each item (row) ranks all other items by decreasing proximity;
        it is obtained by top-(k+1) on a (m, m) similarity matrix, equivalent to the k+1 first columns of argsort.
        Since the item itself is contained in each row, it is expected that initial_rank[i, 0] = i 
        (i.e. the nearest item is the item itself).
    :param k: number of Nearest Neighbors, defaults to 20
    :param trust_factor: the number of reciprocal neighbors to consider including for each k-reciprocal neighbor 
        is trust_factor*k. Defaults to 1/2
    :param overlap_factor: float in [0, 1], how much overlap with the original set of recip. neighbors the
        set of recip. neighbors of each of its members should have in order to be included. Defaults to 2/3
    :param weight_func: function mapping similarities to weights, defaults to 'exp'
    :param param: parameter of the weight function, defaults to 2.4
    :return: (m, m) float tensor, nonzero elements in each row correspond to reciprocal neighbors and their values to a function mapping similarity to weight
    """
    
    V = torch.zeros_like(pwise_sims, dtype=torch.float32)
    for i in range(pwise_sims.shape[0]):
        recip_neighbors = k_reciprocal_NN(initial_rank, i, k)  # (r,) int tensor of indices of reciprocal neighbors of i, r in [1, k+1]
        if trust_factor > 0:
            # extend set of reciprocal neighbors by considering neighbors of neighbors
            recip_neighbors = extended_k_recip_NN(recip_neighbors, initial_rank, trust_factor, overlap_factor) # (m,) bool tensor, True only at indices of the extended set of reciprocal nearest neighbors
        # assign weights to nonzero values of a sparse vector where the index of nonzero elements is the index of neighbors
        V[i, recip_neighbors] = assign_neighbor_weights(recip_neighbors, pwise_sims[i, :], weight_func, param) # (m,) float tensor
        
    return V
        
def local_query_expansion(V, initial_rank, k_exp):
    """Perform 'local query expansion', i.e. a linear combination of each sparse vector with its k_exp Nearest Neighbors.
    NOTE: Unlike in the Zhang et al. implementation (but not in the paper), where there are interactions between all queries and all documents,
    here only a single query vector participates in the expansion; document sparse vectors may also be expanded based on the query.

    :param V: (m, m) float tensor, non-zero elements in each row correspond to reciprocal neighbors and their values depend on their similarity (to the item corresponding to the row)
                Row 0 corresponds to the probe/query.
    :param initial_rank: _description_
    :param k_exp: _description_
    :return: _description_
    """
    V_qexp = np.zeros_like(V, dtype=np.float32)
    for i in range(V_qexp.shape[0]):
        V_qexp[i, :] = torch.mean(V[initial_rank[i, :k_exp], :], dim=0)
    return V_qexp

def compute_jaccard_sim(V):
    """Computes the Jaccard similarity between the probe (query) and each of the m (i.e. num_candidates + 1) candidates (including the query itself).
    The Jaccard similarity is the "intersection over union" of the reciprocal neighbors of the probe and each candidate.
    This implementation works also for non-binary (i.e. with weighted non-zero values) reciprocal neighbor vectors V[i, :].

    :param V: (m, m) float tensor, non-zero elements in each row correspond to reciprocal neighbors and their values depend on their similarity (to the item corresponding to the row).
                Row 0 corresponds to the probe/query.
    :return: (m,) float tensor, Jaccard similarity of the probe with each of the m candidates
    """
    # intersection = torch.min(V[0, :].unsqueeze(0), V) # (m, m)
    # union = torch.max(V[0, :].unsqueeze(0), V)  # (m, m)
    jaccard_sim = torch.sum(torch.min(V[0, :].unsqueeze(0), V), dim=1) / torch.sum(torch.max(V[0, :].unsqueeze(0), V), dim=1)  # (m,)
    
    return jaccard_sim

def combine_similarities(orig_sims, jaccard_sims, comb_type='linear', orig_coef=0.3):
    """Combines geometric similarities with Jaccard similarities into the final similarities used to rank documents.

    :param orig_sims: _description_
    :param jaccard_sims: _description_
    :param comb_type: _description_, defaults to 'linear'
    :param orig_coef: _description_, defaults to 0.3
    :raises NotImplementedError: _description_
    :return: _description_
    """

    if comb_type == 'linear':
        final_sims = orig_coef * orig_sims + (1 - orig_coef) * jaccard_sims
    else:
        raise NotImplementedError("Combination function '{}' not implemented".format(comb_type))
    return final_sims


def re_ranking(pwise_sims, k=20, k_exp=6, lambda_value=0.3):
    """Assumes similarities, not distances

    :param pwise_sims: (m, m) tensor, where m = N + 1. Similarities between all m*m pairs of emb. vectors in the set {query, doc1, ..., doc_N}
    :param k: number of Nearest Neighbors, defaults to 20
    :param k_exp: k used for query expansion. No expansion takes place with k<=1. Defaults to 6
    :param lambda_value: _description_, defaults to 0.3
    :return: _description_
    """

    num_total = pwise_sims.shape[0]  # number of candidate documents + 1 (query)
    initial_rank = torch.topk(pwise_sims, k+1, dim=1, largest=True, sorted=True).indices  # (m, k+1) tensor of indices corresponding to largest values in each row of pwise_sims

    # Compute sparse reciprocal neighbor vectors for each item (i.e. reciprocal adjecency matrix)
    V = compute_rNN_matrix(pwise_sims, initial_rank, k=k, trust_factor=0.5, overlap_factor=2/3, weight_func='exp', param=2.4) # (m, m) float tensor, (sparse) adjacency matrix

    if k_exp > 1:
        V = local_query_expansion(V, initial_rank, k_exp)  # (m, m)
    
    jaccard_sims = compute_jaccard_sim(V)  # (m,)

    final_sims = combine_similarities(pwise_sims[0, :], jaccard_sims, orig_coef=0.3)  # (m,) includes self-similarity at index 0

    return final_sims