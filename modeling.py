import math
import logging
from typing import Optional, Any, Dict

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AutoModel

logger = logging.getLogger(__name__)


def _average_query_doc_embeddings(sequence_output, token_type_ids, valid_mask):
    query_flags = (token_type_ids==0)*(valid_mask==1)  # (valid_mask == 1) seems superfluous, but is prob. there to convert to bool
    doc_flags = (token_type_ids==1)*(valid_mask==1)

    query_lengths = torch.sum(query_flags, dim=-1)
    query_lengths = torch.clamp(query_lengths, 1, None)  # length is at least 1
    doc_lengths = torch.sum(doc_flags, dim=-1)
    doc_lengths = torch.clamp(doc_lengths, 1, None)
    
    query_embeddings = torch.sum(sequence_output * query_flags[:,:,None], dim=1)
    query_embeddings = query_embeddings/query_lengths[:, None]
    doc_embeddings = torch.sum(sequence_output * doc_flags[:,:,None], dim=1)
    doc_embeddings = doc_embeddings/doc_lengths[:, None]
    return query_embeddings, doc_embeddings


def _mask_both_directions(valid_mask, token_type_ids):
    """0 masked (ignored), 1 not masked (use). There is no query-doc token interaction in self-attention calculations"""
    assert valid_mask.dim() == 2
    attention_mask = valid_mask[:, None, :]

    type_attention_mask = torch.abs(token_type_ids[:, :, None] - token_type_ids[:, None, :])
    attention_mask = attention_mask - type_attention_mask
    attention_mask = torch.clamp(attention_mask, 0, None)
    return attention_mask


class RepBERT_Train(BertPreTrainedModel):
    """
    Instead of bi-encoder, a single encoder is used, and token_type_ids with attention masking is applied to separate
    query self-attention and doc self-attention. There is no query-doc token interaction in self-attention calculations.
    Based on HF BERT implementation, this may be inefficient: attention masks (apparently) are just added, instead
    of preventing computations altogether. This means that the full O(seq_len^2) cost is there, and is therefore worse
    than O(query_len^2) + O(doc_len^2). Some minor benefit comes from avoiding separate padding for query.
    """
    def __init__(self, config):
        super(RepBERT_Train, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, valid_mask, position_ids, labels=None):
        """
        :param input_ids:
        :param token_type_ids:
        :param valid_mask: (batch_size, batch_seq_length)
        :param position_ids:
        :param labels: (batch_size, batch_size) padded (with -1) tensor of the position of relevant docs within in-batch (local) indices
        :return: Tuple[(  loss,
                        (batch_size, batch_size) tensor of similarities between each query in the batch and
                            each document in the batch
                        (batch_size, hidden_size) tensor of averaged query token embeddings
                        (batch_size, hidden_size) tensor of averaged document token embeddings
                        )]
        """
        attention_mask = _mask_both_directions(valid_mask, token_type_ids)  # (batch_size, seq_len, seq_len), 1-use, 0-mask

        # (batch_size, sequence_length, hidden_size)
        sequence_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)[0]
        
        query_embeddings, doc_embeddings = _average_query_doc_embeddings(
            sequence_output, token_type_ids, valid_mask
        )  # (batch_size, hidden_size) averaged embeddings
        
        similarities = torch.matmul(query_embeddings, doc_embeddings.T)  # (batch_size, batch_size)
        
        output = (similarities, query_embeddings, doc_embeddings)
        if labels is not None:
            loss_fct = nn.MultiLabelMarginLoss()
            loss = loss_fct(similarities, labels)
            output = loss, *output
        return output


def _average_sequence_embeddings(sequence_output, valid_mask):
    """
    :param sequence_output: (batch_size, sequence_len, d_model) tensor of output embeddings for each sequence position
    :param valid_mask: (batch_size, query_length) attention mask bool tensor for sequence positions; 1 use, 0 ignore
    :return: (batch_size, d_model) aggregate representation vector for each sequence in the batch
    """
    flags = valid_mask.bool()  # (valid_mask == 1) seems superfluous, but is prob. there to convert to bool
    lengths = torch.sum(flags, dim=-1)
    lengths = torch.clamp(lengths, 1, None)
    sequence_embeddings = torch.sum(sequence_output * flags[:, :, None], dim=1)
    sequence_embeddings = sequence_embeddings/lengths[:, None]
    return sequence_embeddings


def _select_first_embedding(sequence_output, valid_mask=None):
    """
    :param sequence_output: (batch_size, sequence_len, d_model) tensor of output embeddings for each sequence position
    :param valid_mask: not used; for compatibility
    :return: (batch_size, d_model) aggregate representation vector for each sequence in the batch
    """
    return sequence_output[:, 0, :]


class RepBERT(BertPreTrainedModel):
    """
    Takes a single type of input (either query or doc) and returns embedding
    """
    def __init__(self, config):
        super(RepBERT, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        if config.encode_type == "doc":
            self.token_type_func = torch.ones_like
        elif config.encode_type == "query":
            self.token_type_func = torch.zeros_like
        else:
            raise NotImplementedError()

    def forward(self, input_ids, valid_mask):
        token_type_ids = self.token_type_func(input_ids)
        sequence_output = self.bert(input_ids,
                            attention_mask=valid_mask, 
                            token_type_ids=token_type_ids)[0]
        
        text_embeddings = _average_sequence_embeddings(
            sequence_output, valid_mask
        )
        
        return text_embeddings


class Scorer(nn.Module):
    """
    When used as a base class, it creates the functions used to compute the query-document relevance scores.
    When used as a stand-alone class:
    Directly scores final document representations in the "decoder".
    Exploits cross-attention between encoded query states and documents in the preceding decoder layer.
    The decoder creates its Queries from the layer below it, and takes the Keys and Values from the output of the encoder.
    TODO: What is the effect of LayerNormalization? Doesn't it flatten the scores distribution?
    """

    def __init__(self, d_model, scoring_mode=None):
        super(Scorer, self).__init__()
        self.build_score_function(d_model, scoring_mode)

    def build_score_function(self, d_model, scoring_mode):
        """
        :param d_model: the dimension of the vectors (typically, embeddings of size d_model, but not in case of "dot_product" scoring)
            before the scoring function
        :param scoring_mode: string specifying how to score documents
        """

        if scoring_mode.endswith('sigmoid'):
            self.linear = nn.Linear(d_model, 1)
            self.score_func = torch.sigmoid
        elif scoring_mode.endswith('tanh'):
            self.linear = nn.Linear(d_model, 1)
            self.score_func = torch.tanh
        elif scoring_mode.endswith('softmax'):
            self.linear = nn.Linear(d_model, 2)
            self.score_func = torch.nn.LogSoftmax(dim=-1)
        else:
            self.linear = nn.Linear(d_model, 1)

            # self.linear.weight.requires_grad = False
            # self.linear.weight.data.fill_(1.00)
            # self.linear.bias.requires_grad = False
            # self.linear.bias.data.fill_(0.00)

            self.score_func = torch.nn.Identity()

    def forward(self, output_emb, *args, **kwargs):
        """
        :param output_emb: (num_docs, batch_size, doc_emb_size) transformed sequence of document embeddings
        :return: case `scoring_mode`:
            None: (batch_size, num_docs) relevance scores, floats of arbitrary range
            'sigmoid': (batch_size, num_docs) relevance scores in [0, 1]
            'tanh': (batch_size, num_docs) relevance scores in [-1, 1]
            'softmax': (batch_size, num_docs, 2) 0-> relevance log-probability, 1-> non-relevance log-probability
        """
        output_emb = output_emb.permute(1, 0, 2)  # (batch_size, num_docs, doc_emb_size)
        return self.score_func(self.linear(output_emb))


class CrossAttentionScorer(Scorer):
    """
    Applies multi-headed attention between query term embeddings and final document representations in the "decoder".
    Final document representations are used as Queries, (query) encoder representations are used as Keys and Values.
    TODO: Experiment with the reverse? doc_emb as K, V, query_emb as Q
    """

    def __init__(self, d_model, scoring_mode=None):
        super(CrossAttentionScorer, self).__init__(d_model, scoring_mode)

        self.multihead_attn = torch.nn.MultiheadAttention(d_model, num_heads=8)
        self.activation_func = torch.nn.GELU()

    def forward(self, output_emb, query_emb, query_mask):
        """
        :param output_emb: output_emb: (num_docs, batch_size, d_model) transformed sequence of document embeddings
        :param query_emb: (query_length, batch_size, d_model) final query term embeddings
        :param query_mask: (batch_size, query_length) attention mask bool tensor for query tokens; 0 use, 1 ignore
        :return: case `scoring_mode`:
            None or 'cross_attention': (batch_size, num_docs) relevance scores, floats of arbitrary range
            'sigmoid': (batch_size, num_docs) relevance scores in [0, 1]
            'tanh': (batch_size, num_docs) relevance scores in [-1, 1]
            'softmax': (batch_size, num_docs, 2) 0-> relevance log-probability, 1-> non-relevance log-probability
        """
        output_emb = self.activation_func(output_emb)  # TODO: test
        query_emb = self.activation_func(query_emb)  # TODO: test
        out = self.multihead_attn(output_emb, query_emb, query_emb, key_padding_mask=query_mask, need_weights=False)[0]
        out = out.permute(1, 0, 2)  # (batch_size, num_docs, d_model)
        return self.score_func(self.linear(out))


def get_aggregation_function(aggregation):
    """
    :param aggregation: defines how to aggregate final query token representations to obtain a single vector
            representation for the query. 'mean' will average, 'first' will simply select the first vector
    :return aggregation_func: aggregation function
    """
    if aggregation == 'mean':
        aggregation_func = _average_sequence_embeddings
    else:
        aggregation_func = _select_first_embedding
    return aggregation_func


class DotProductScorer(Scorer):
    """
    Computes scores as a dot product between the aggregate (e.g. mean) query representation and each final document
    embedding.
    """
    def __init__(self, scoring_mode='', pre_activation=None, normalize=False, aggregation='mean'):
        """
        :param scoring_mode: string, same as the option used to initialize `MDSTransformer`. At this point, the string
            must start with "doc_product" or "cosine", but the suffix can specify a (non)linear transformation to be used on scores
        :param pre_activation: activation function to use on representations BEFORE computing the inner product
        :param normalize: if True, will divide product by vector norms, i.e. will compute the cosine similarity
        :param aggregation: defines how to aggregate final query token representations to obtain a single vector
            representation for the query. 'mean' will average, 'first' will simply select the first vector
        """
        super(DotProductScorer, self).__init__(d_model=1, scoring_mode=scoring_mode)

        if pre_activation is not None:
            self.pre_activation_func = torch.nn.GELU()
        else:
            self.pre_activation_func = None

        self.aggregation_func = get_aggregation_function(aggregation)
        self.normalize = normalize

    def forward(self, output_emb, query_emb, query_mask):
        """
        :param output_emb: output_emb: (num_docs, batch_size, d_model) transformed sequence of document embeddings
        :param query_emb: (query_length, batch_size, d_model) final query term embeddings
        :param query_mask: (batch_size, query_length) attention mask bool tensor for query tokens; 0 use, 1 ignore
        :return: scores: (batch_size, num_docs) inner product between aggregate query embedding and each document embedding
            if 'dot_product_softmax': (batch_size, num_docs, 2) 0-> relevance log-probability, 1-> non-relevance log-probability
        """
        if self.pre_activation_func is not None:
            output_emb = self.pre_activation_func(output_emb)  # TODO: test
            query_emb = self.pre_activation_func(query_emb)  # TODO: test

        output_emb = output_emb.permute(1, 0, 2)  # (batch_size, num_docs, d_model)
        query_emb = query_emb.permute(1, 0, 2)  # (batch_size, query_len, d_model)
        agg_query_emb = self.aggregation_func(query_emb, ~query_mask)  # (batch_size, d_model)

        if self.normalize:
            scores = F.cosine_similarity(output_emb, agg_query_emb[:, None, :], dim=2, eps=1e-6)
        else:
            # scores = torch.matmul(output_emb, avg_query_emb[:, :, None])  # (batch_size, num_docs, 1) when using self.score_func for re-scaling
            # scores = self.score_func(self.linear(scores))
            scores = torch.matmul(output_emb, agg_query_emb[:, :, None]).squeeze()  # to disable scaling

        return scores


class RelevanceCrossEntropyLoss(nn.Module):
    """
    Special cross-entropy loss
    """
    def __init__(self):
        super(RelevanceCrossEntropyLoss, self).__init__()

    def forward(self, predictions, labels):
        """
        :param predictions: (batch_size, num_docs, 2) relevance class log-probabilities for each candidate document and query.
            Dimension [:, :, 0] corresponds to the log-prob. for the "relevant" class and [:, :, 1] to the "non-relevant" class.
        :param labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices (positions) of the
                relevant documents within its corresponding pool of candidates (docinds). If n relevant documents exist,
                then labels[0:n] are the positions of these documents inside `docinds`, and labels[n:] == -1,
                indicating non-relevance.
        :return: loss: scalar tensor. Mean loss per document
        """
        # WARNING: works assuming that `labels` aren't scores but integer indices of relevant documents padded with -1, e.g. [0, 1, 2, -1, ..., -1]

        # For the entire batch, calculate one loss component from positive documents and one from negatives, normalizing by their numbers.
        # Here, queries with more positive (and negative) documents contribute more to the loss calculation than queries
        # with a smaller number.
        is_relevant = (labels > -1)
        total_relevant = is_relevant.sum()  # total number of relevant documents in the batch
        loss_pos = torch.sum(predictions[:, :, 0] * is_relevant) / total_relevant  # scalar. loss per document, for positive documents
        loss_neg = torch.sum(predictions[:, :, 1] * ~is_relevant) / (labels.shape[0]*labels.shape[1] - total_relevant)  # scalar. loss per document, for negative documents
        loss = - (loss_pos + loss_neg)

        return loss


class RelevanceListnetLoss(nn.Module):
    """
    ListNet loss
    """
    def __init__(self):
        super(RelevanceListnetLoss, self).__init__()

    def forward(self, predictions, labels):
        """
        :param predictions: (batch_size, num_docs) relevance scores (arb. range) for each candidate document and query.
        :param labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices (positions) of the
                relevant documents within its corresponding pool of candidates (docinds). If n relevant documents exist,
                then labels[0:n] are the positions of these documents inside `docinds`, and labels[n:] == -1,
                indicating non-relevance.
        :return: loss: scalar tensor. Mean loss per query
        """
        # WARNING: works assuming that `labels` aren't scores but integer indices of relevant documents padded with -1, e.g. [0, 1, 2, -1, ..., -1]

        _labels_values = labels.new_zeros(labels.shape, dtype=torch.float32)
        is_relevant = (labels > -1)
        _labels_values[is_relevant] = 1
        # NOTE: _labels_values = _labels_values / torch.sum(is_relevant, dim=1).unsqueeze(dim=1)
        # is equivalent but interestingly much slower than setting -Inf and computing Softmax; maybe due to CUDA Softmax code
        _labels_values[labels == -1] = float("-Inf")
        labels_probs = torch.nn.Softmax(dim=1)(_labels_values)

        predictions_logprobs = torch.nn.LogSoftmax(dim=1)(predictions)  # (batch, num_docs) log-distribution over docs
        # KLDivLoss expects predictions ('inputs') as log-probabilities and 'targets' as probabilities
        loss = torch.nn.KLDivLoss(reduction='batchmean')(predictions_logprobs, labels_probs)
        
        return loss


class MultiTierLoss(nn.Module):
    """
    Uses multiple tiers of relevance for candidate documents, determined by their ranking from the candidate retrieval method.
    Encourages that the similarity score between the query and all documents in each tier is higher than the similarity
    between the query and all documents from lower tiers.
    """
    def __init__(self, num_tiers=3, tier_size=50, tier_distance=None, diff_function='maxmargin', gt_function=None, gt_factor=2, reduction='mean'):
        """
        :param num_tiers: total number of tiers (ground truth documents are not considered a separate tier)
        :param tier_size: number of documents in each tier
        :param tier_distance: number of documents acting as a buffer between each tier.
            If None, the distance will be automatically calculated so as to place the tier centers as widely apart as possible.
        :param diff_function: function to be applied to score differences between documents belonging to different tiers
        :param gt_function: special loss function to be applied for calculating the extra contribution of the ground truth
                            relevant documents. If None, no special treatment will be given to ground truth relevant
                            documents in the loss calculation, besides including them in the top tier
        :param gt_factor: scaling factor (coefficient) of special ground truth component computed by `gt_function`
        :param reduction: if 'none', a loss for each batch item (query) will be computed, otherwise the 'mean' or 'sum'
                        over queries in the batch
        """
        super(MultiTierLoss, self).__init__()

        self.num_tiers = num_tiers
        self.tier_size = tier_size
        self.tier_distance = tier_distance

        if diff_function == 'exp':
            self.diff_function = lambda x: torch.exp(-x)
        elif diff_function == 'maxmargin':
            self.diff_function = lambda x: torch.nn.functional.relu(1 - x)  # max(0, 1-x)
        else:
            raise NotImplementedError("Difference function '{}' not implemented")

        if gt_function == 'same':  # will use the same function as `diff_function`
            self.gt_function = self.compute_gt_diffs
        elif gt_function == 'multilabelmargin':  # this is equivalent to 'same' with `diff_function`=='maxmargin', but much faster (avoids Python loop over batch_size)
            self.gt_function = nn.MultiLabelMarginLoss(reduction='none')
        else:  # if None, no special treatment for ground truth relevant documents.
            self.gt_function = gt_function
        self.gt_factor = gt_factor

        self.reduction = reduction

    def compute_diffs(self, scores, inds1, inds2):
        """
        :param scores: (batch_size, num_docs) relevance scores for each candidate document and query
        :param inds1: (num_tier1docs,) indices (locations) of documents within first tier. Same across batch (i.e. for all queries)! Can be a list or range.
        :param inds2: (num_tier2docs,) indices (locations) of documents within second tier. Same across batch (i.e. for all queries)! Can be a list or range.
        :return query_losses: (batch_size,) loss for each query corresponding
            to the score differences between documents with inds1 and documents with inds2
        """

        # normalize by the number of terms: num_pos + num_neg to be consistent with MultiLabelMargin, or num_pos*num_neg to properly neutralize the effect of the tier size
        query_losses = torch.sum(
            self.diff_function(scores[:, inds1].unsqueeze(dim=2) - scores[:, inds2].unsqueeze(dim=1)), dim=(1, 2)) / (
                                   len(inds1) * len(inds2))

        return query_losses

    def compute_gt_diffs(self, scores, labels):
        """
        loop O(batch_size)
        Use `labels` to determine indices of ground truth and negative documents, and use them
        to call `compute_diffs`.
        # TODO: relies on the easy-to-lift assumption that all g.t. are at the beginning

        :param scores: (batch_size, num_docs) relevance scores for each candidate document and query.
        :param labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices (positions) of the
                relevant documents within its corresponding pool of candidates (docinds). If n relevant documents exist,
                then labels[0:n] are the positions of these documents inside `docinds`, and labels[n:] == -1,
                indicating non-relevance.
        :return query_losses: (batch_size,) loss for each query
        """
        num_relevant = (labels > -1).sum(dim=1)  # (batch_size,)

        # equivalent to:
        # torch.tensor([self.compute_diffs(scores[i, :].unsqueeze(0), range(num_relevant[i]), range(num_relevant[i], scores.shape[1])) for i in range(scores.shape[0])], device=scores.device)
        # loop O(batch_size)
        query_losses = torch.tensor([torch.sum(self.diff_function(
            scores[i, range(num_relevant[i])].unsqueeze(dim=-1) - scores[i, range(num_relevant[i], scores.shape[1])]))
                                     for i in range(scores.shape[0])]).to(device=scores.device)

        # normalize by the number of terms: num_pos + num_neg to be consistent with MultiLabelMargin, or num_pos*num_neg to properly neutralize the effect of the tier size
        return query_losses / scores.shape[1]

    def get_tier_inds(self, num_negatives):

        if self.tier_distance is None:
            # places tier centers in maximal distance from one another
            mids_distance = math.ceil((num_negatives - 2 * self.tier_size) / (self.num_tiers - 1))
            start_inds = [0] + [int(self.tier_size / 2) + i * mids_distance for i in range(1, self.num_tiers-1)] + [num_negatives - self.tier_size]
        else:  # starting from the most relevant candidates, makes tiers using the pre-specified distance to separate them
            start_inds = []  # for-loop to make it fool-proof (safely allow larger number of tiers than would be legal)
            for i in range(0, self.num_tiers):
                new_ind = i*(self.tier_distance + self.tier_size)
                if new_ind >= num_negatives:
                    break
                else:
                    start_inds.append(new_ind)
        return start_inds

    def forward(self, scores, labels):
        """
        :param scores: (batch_size, num_docs) relevance scores for each candidate document and query.
        :param labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices (positions) of the
                relevant documents within its corresponding pool of candidates (docinds). If n relevant documents exist,
                then labels[0:n] are the positions of these documents inside `docinds`, and labels[n:] == -1,
                indicating non-relevance.
        :return: loss: (batch_size,) tensor of aggregate loss per document for each query if reduction=='none',
                        otherwise scalar tensor of aggregate loss per query and document
        """

        # is_relevant = (labels > -1)
        # num_relevant = is_relevant.sum(dim=1)  # total number of relevant documents in the batch
        # num_negatives = scores.shape[1] - num_relevant
        num_negatives = scores.shape[1]  # this is approximately correct; we hereby include the g.t. relevant in tier 1
        start_inds = self.get_tier_inds(num_negatives)

        # Treating as "positives" the documents within each tier, the loss compares
        # their scores to the scores of documents in all lower tiers (and the ones in-between lower tiers)
        # complexity O(num_tiers)
        loss = torch.zeros(scores.shape[0], dtype=torch.float32, device=scores.device)  # (batch_size,) loss for each query
        for t in range(self.num_tiers - 1):
            # variant to compare with immediately lower tier: inds2 = range(start_inds[t+1], start_inds[t+1] + self.tier_size)
            loss += self.compute_diffs(scores, range(start_inds[t], start_inds[t] + self.tier_size),
                                       range(start_inds[t + 1], scores.shape[1]))

        if self.gt_function is not None:
            # in this case a special loss component will be computed from ground truth relevant documents versus
            # all other candidates using the provided function, added with a scaling factor of `self.gt_factor`
            gt_loss = self.gt_function(scores, labels)
            loss = loss + self.gt_factor * gt_loss

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


def get_loss_module(loss_type, args):

    if loss_type == 'multilabelmargin':
        return nn.MultiLabelMarginLoss()
    elif loss_type == 'crossentropy':
        return RelevanceCrossEntropyLoss()
    elif loss_type == 'listnet':
        return RelevanceListnetLoss()
    elif loss_type == 'multitier':
        return MultiTierLoss(num_tiers=args.num_tiers, tier_size=args.tier_size, tier_distance=args.tier_distance,
                             diff_function=args.diff_function,
                             gt_function=args.gt_function,
                             gt_factor=args.gt_factor)
    else:
        raise NotImplementedError("Loss type '{}' not implemented!".format(args.loss_type))


class NoCrossTransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""NoCrossTransformerDecoderLayer is a modified nn.TransformerDecoderLayer, without a cross-attention layer
    attending over the sequence of encoder embeddings.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalization="LayerNorm"):
        super(nn.TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.normalization = normalization
        if normalization == "LayerNorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        elif normalization == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
            self.norm3 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(NoCrossTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if self.normalization != 'None':
            tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if self.normalization != 'None':
            tgt = self.norm3(tgt)
        return tgt


class LinearTransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""LinearTransformerDecoderLayer is a modified nn.TransformerDecoderLayer, with the self-attention layer replaced
    by a dense layer, followed by a non-linearity.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.TransformerDecoderLayer, self).__init__()

        self.linear_attn_substitute = nn.Linear(d_model, d_model)
        self.act1 = nn.ReLU()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(LinearTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.act1(self.linear_attn_substitute(tgt))
        tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class ReducedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""ReducedTransformerDecoderLayer is a modified nn.TransformerDecoderLayer, with no self-attention layer.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.TransformerDecoderLayer, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ReducedTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MDSTransformer(nn.Module):
    r"""Multiple Document Scoring Transformer. By default, consists of a query enconder (Huggingface implementation),
    and a "decoder" using self-attention over a sequence (set) of document representations and cross-attention over
    the query term representations in the  encoder output.
    Computes a relevance score for each (transformed) document representation and can be thus used for reranking.

    Args:
        encoder_config: huggingface transformers configuration object (could be string, dir, ...)
            to instantiate a query encoder.  Used instead of `custom_encoder`.
        custom_encoder: custom encoder object initialized externally (default=None)
            Used instead of `encoder_config`.
        custom_decoder: custom decoder object initialized externally (default=None).
            If not specified, then the following are used:

        d_model: the "decoder" representation (hidden) dimension. Is also the doc embedding dimension
        num_heads: the number of heads in the multiheadattention decoder layers.
        num_decoder_layers: the number of sub-decoder-layers in the decoder.
        dim_feedforward: the dimension of the decoder feedforward network module.
        dropout: the decoder dropout value.
        activation: the activation function of the decoder intermediate layer, "relu" or "gelu".
        positional_encoding: if None, no positional encoding is used for the "decoder", and thus the output is permutation
            equivariant with respect to the document embedding sequence

    Examples::
        >>> model = MDSTransformer(enc_config, num_heads=16, num_decoder_layers=4)
        >>> model = MDSTransformer(custom_encoder=my_HF_encoder, num_heads=16, num_decoder_layers=4)
    """

    def __init__(self, encoder_config=None, custom_encoder=None, custom_decoder=None,
                 d_model: int = 768, num_heads: int = 8,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalization: str = "LayerNorm", positional_encoding=None,
                 doc_emb_dim: int = None,
                 scoring_mode='cross_attention', query_emb_aggregation='mean',
                 loss_module=None, aux_loss_module=None, aux_loss_coeff=0,
                 selfatten_mode=0, no_decoder=False, no_dec_crossatten=False, transform_doc_emb=False,
                 bias_regul_coeff=0.0, bias_regul_cutoff=100) -> None:
        super(MDSTransformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = AutoModel.from_config(encoder_config)

        self.doc_emb_dim = doc_emb_dim  # the expected document vector dimension. If None, it will be assumed to be d_model

        if custom_decoder is not None:
            self.decoder = custom_decoder
            self.d_model = self.decoder.d_model
        else:
            if d_model is None:
                self.d_model = doc_emb_dim
                logger.warning("No `d_model` provided. Will use {} dim. for transformer model dimension, "
                               "to match expected document dimension!".format(self.d_model))
            else:
                self.d_model = d_model

            assert self.d_model is not None, "One of `doc_emb_dim` or `d_model` should be not None!"

            self.no_decoder = no_decoder
            if not no_decoder:
                self.selfatten_mode = selfatten_mode  # NOTE: used for ablation study
                self.no_dec_crossatten = no_dec_crossatten
                if selfatten_mode == 2:  # transformer decoder block with the self-attention layer replaced by linear layer + non-linearity
                    decoder_layer = LinearTransformerDecoderLayer(self.d_model, num_heads, dim_feedforward, dropout, activation)
                elif selfatten_mode == 3:  # transformer decoder block with the self-attention layer simply removed
                    decoder_layer = ReducedTransformerDecoderLayer(self.d_model, num_heads, dim_feedforward, dropout, activation)
                elif no_dec_crossatten:  # decoder does not have cross-attention over encoder states
                    decoder_layer = NoCrossTransformerDecoderLayer(self.d_model, num_heads, dim_feedforward, dropout, activation, normalization)
                else:  # usual transformer decoder block
                    decoder_layer = nn.TransformerDecoderLayer(self.d_model, num_heads, dim_feedforward, dropout, activation)
                decoder_norm = nn.LayerNorm(self.d_model)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
                # Without any init call, weight parameters are initialized as by default when creating torch layers (Kaiming uniform)
                self.decoder.apply(lambda x: (torch.nn.init.xavier_uniform_ if hasattr(x, 'dim') else lambda y: y))

        if self.doc_emb_dim is None:
            self.doc_emb_dim = d_model
            logger.warning("Using {} dim. for transformer model dimension; "
                           "expecting same document embedding dimension!".format(self.d_model))

        self.num_heads = num_heads

        self.query_dim = self.encoder.config.hidden_size  # e.g. 768 for BERT-base
        self.query_emb_aggregation = query_emb_aggregation  # how to aggregate output query token representations

        # project query representation vectors to match dimensionality of doc embeddings (for cross-attention and/or scoring)
        if self.query_dim != self.d_model:
            self.project_query = nn.Linear(self.query_dim, self.d_model)

        # project document representation vectors to match dimensionality of d_model
        self.project_documents = None
        if (self.doc_emb_dim != self.d_model) or transform_doc_emb:
            self.project_documents = nn.Linear(self.doc_emb_dim, self.d_model)
            logger.warning("Using {} dim. for transformer model dimension; will project document embeddings "
                           "of dimension {} to match!".format(self.d_model, self.doc_emb_dim))

        self.scoring_mode = scoring_mode
        self.score_docs = self.get_scoring_module(scoring_mode, query_emb_aggregation)

        self.loss_module = loss_module
        self.aux_loss_module = aux_loss_module
        self.aux_loss_coeff = aux_loss_coeff
        
        self.bias_regul_coeff = bias_regul_coeff
        self.bias_regul_cutoff = bias_regul_cutoff

    def get_scoring_module(self, scoring_mode, query_emb_aggregation):

        if scoring_mode.startswith('cross_attention'):
            return CrossAttentionScorer(self.d_model, scoring_mode)
        elif scoring_mode.startswith('dot_product'):
            if 'gelu' in scoring_mode:
                pre_activation_func = 'gelu'
            else:
                pre_activation_func = None
            return DotProductScorer(scoring_mode=scoring_mode, pre_activation=pre_activation_func, aggregation=query_emb_aggregation)
        elif scoring_mode.startswith('cosine'):
            if 'gelu' in scoring_mode:
                pre_activation_func = 'gelu'
            else:
                pre_activation_func = None
            return DotProductScorer(scoring_mode=scoring_mode, pre_activation=pre_activation_func, normalize=True, aggregation=query_emb_aggregation)
        else:
            return Scorer(self.d_model, scoring_mode)

    def forward(self, query_token_ids: Tensor, query_mask: Tensor = None, doc_emb: Tensor = None,
                docinds: Tensor = None, local_emb_mat: Tensor = None, doc_padding_mask: Tensor = None,
                doc_attention_mat_mask: Tensor = None, doc_neutscore: Tensor = None, labels: Tensor = None) -> Dict[str, Tensor]:
        r"""
        num_docs is the number of candidate docs per query and corresponds to the length of the padded "decoder" sequence
        :param  query_token_ids: (batch_size, max_query_len) tensor of padded sequence of token IDs fed to the encoder
        :param  query_mask: (batch_size, query_length) attention mask bool tensor for query tokens; 0 ignore, non-0 use
        :param  doc_emb: (batch_size, num_docs, doc_emb_dim) sequence of document embeddings fed to the "decoder".
                    Mutually exclusive with `docinds`.
        :param  docinds: (batch_size, num_docs) tensor of local indices of documents corresponding to rows of the
                    `local_emb_mat` used to lookup document vectors in nn.Embedding. Mutually exclusive with `doc_emb`.
        :param  local_emb_mat: (num_unique_docIDs, doc_emb_dim) tensor of local doc embedding matrix containing emb. vectors
                    of all unique documents in the batch.  Used with `docinds` to lookup document vectors in nn.Embedding on the GPU.
                    This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth.
                    Global matrix cannot be used, because the collection size is in the order of 10M: GPU memory!
        :param  doc_padding_mask: (batch_size, num_docs) boolean/ByteTensor mask with 0 at positions of missing input
                    documents (decoder sequence length is less than the max. doc. pool size in the batch)
        :param  doc_attention_mat_mask: (num_docs, num_docs) float additive mask for the decoder sequence (optional).
                    This is for causality, and if FloatTensor, can be directly added on top of the attention matrix.
                    If BoolTensor, positions with ``True`` are ignored, while ``False`` values will be considered.
        :param  doc_neutscore: (batch_size, num_docs) sequence of document neutrality scores to calculate fairness.
        :param  labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices of the
                relevant documents within its corresponding pool of candidates (docinds).
                    Optional: If provided, the loss will be computed.

        :returns:
            dict containing:
                rel_scores: (batch_size, num_docs) relevance scores in [0, 1]
                loss: scalar mean loss over entire batch (only if `labels` is provided!)
        """
        if doc_emb is None:  # happens only in training, when additionally there is in-batch negative sampling
            doc_emb = self.lookup_doc_emb(docinds, local_emb_mat)  # (batch_size, max_docs_per_query, doc_emb_dim)
            
        if self.project_documents is not None:
            doc_emb = self.project_documents(doc_emb)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]
        doc_emb = doc_emb.permute(1, 0, 2)  # (max_docs_per_query, batch_size, doc_emb_dim) document embeddings

        if query_token_ids.size(0) != doc_emb.size(1):
            raise RuntimeError("the batch size for queries and documents must be equal")

        # HF encoder output is a weird dictionary type
        encoder_out = self.encoder(query_token_ids.to(torch.int64), attention_mask=query_mask)  # int64 required by torch nn.Embedding
        enc_hidden_states = encoder_out['last_hidden_state']  # (batch_size, max_query_len, query_dim)
        if self.query_dim != self.d_model:  # project query representation vectors to match dimensionality of doc embeddings
            enc_hidden_states = self.project_query(enc_hidden_states)  # (batch_size, max_query_len, d_model)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]
        enc_hidden_states = enc_hidden_states.permute(1, 0, 2)  # (max_query_len, batch_size, d_model)

        # The nn.MultiHeadAttention expects ByteTensor or Boolean and uses the convention that non-0 is ignored
        # and 0 is used in attention, which is the opposite of HuggingFace.
        memory_key_padding_mask = ~query_mask

        if self.no_decoder:
            output_emb = doc_emb
        else:
            if self.selfatten_mode == 1:  # NOTE: for ablation study. Turn off SA by using diagonal SA matrix (no interactions between documents)
                doc_attention_mat_mask = ~torch.eye(doc_emb.shape[0], dtype=bool).to(device=doc_emb.device)  # (max_docs_per_query, max_docs_per_query)
            # (num_docs, batch_size, doc_emb_size) transformed sequence of document embeddings
            output_emb = self.decoder(doc_emb, enc_hidden_states, tgt_mask=doc_attention_mat_mask,
                                      tgt_key_padding_mask=~doc_padding_mask,  # again, MultiHeadAttention opposite of HF
                                      memory_key_padding_mask=memory_key_padding_mask)
            # output_emb = self.act(output_emb)  # the output transformer encoder/decoder embeddings don't include non-linearity

        predictions = self.score_docs(output_emb, enc_hidden_states, memory_key_padding_mask)  # relevance scores. dimensions vary depending on scoring_mode
        
        if self.scoring_mode.endswith('softmax'):
            rel_scores = torch.exp(predictions[:, :, 0])  # (batch_size, num_docs) relevance scores
        else:
            rel_scores = predictions.squeeze()  # (batch_size, num_docs) relevance scores
            
        # Fairness regularization term  # TODO: wrap in a separate function
        bias_regul_term = None
        if doc_neutscore is not None:

            _cutoff = np.min([self.bias_regul_cutoff, doc_neutscore.shape[1]])

            _indices_sorted = torch.argsort(rel_scores, dim=1, descending=True)
            _indices_sorted[_indices_sorted < _cutoff] = -1
            _indices_sorted[_indices_sorted != -1] = 0
            _indices_sorted[_indices_sorted == -1] = 1
            _indices_mask = doc_neutscore.new_zeros(doc_neutscore.shape)    
            _indices_mask[_indices_sorted == 0] = float("-Inf")

            doc_neutscore_probs = torch.nn.Softmax(dim=1)(doc_neutscore + _indices_mask)
            rel_scores_logprobs = torch.nn.LogSoftmax(dim=1)(rel_scores + _indices_mask)

            bias_regul_term = torch.nn.KLDivLoss(reduction='batchmean')(rel_scores_logprobs, doc_neutscore_probs)

        # Compute loss
        if labels is not None:
            loss = self.loss_module(rel_scores, labels.to(torch.int64))  # loss is scalar tensor. labels are int16, convert to int64 for PyTorch losses

            if self.aux_loss_module is not None and (self.aux_loss_coeff > 0):  # add auxiliary loss, if specified
                loss += self.aux_loss_coeff * self.aux_loss_module(rel_scores, labels.to(torch.int64))
            if bias_regul_term is not None:
                if self.bias_regul_coeff < 0:
                    loss = rel_scores.new([0])[0]
                    loss += - self.bias_regul_coeff * bias_regul_term
                else:    
                    loss += self.bias_regul_coeff * bias_regul_term
            
            return {'loss': loss, 'rel_scores': rel_scores}
        return {'rel_scores': rel_scores}

    @staticmethod
    def lookup_doc_emb(docinds, local_emb_mat):
        """
        Lookup document vectors in `local_emb_mat` corresponding to rows given in `docinds`.
        This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth.
        Global matrix cannot be used, because the collection size is in the order of 10M: GPU memory!
        """
        embedding = torch.nn.Embedding.from_pretrained(local_emb_mat, freeze=True, padding_idx=local_emb_mat.shape[0]-1)
        return embedding(docinds.to(torch.int64))

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with 0.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
