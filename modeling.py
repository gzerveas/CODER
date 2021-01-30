import math
import logging
from typing import Optional, Any, Dict

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, RobertaModel

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
        :param labels: (batch_size, batch_size) padded (with -1) tensor of relevant doc in-batch (local) indices
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
    flags = valid_mask==1  # (valid_mask == 1) seems superfluous, but is prob. there to convert to bool
    lengths = torch.sum(flags, dim=-1)
    lengths = torch.clamp(lengths, 1, None)
    sequence_embeddings = torch.sum(sequence_output * flags[:,:,None], dim=1)
    sequence_embeddings = sequence_embeddings/lengths[:, None]
    return sequence_embeddings


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


class CrossAttentionScorer(nn.Module):
    """
    Exploits decoder cross-attention between encoded query states and documents.
    The decoder creates its Queries from the layer below it, and takes the Keys and Values from the output of the encoder
    TODO: What is the effect of LayerNormalization? Doesn't it flatten the scores distribution?
    TODO: consider modifying the final cross-attention layer, to allow interactions between decoder's Values
    """

    def __init__(self, d_model):
        super(CrossAttentionScorer, self).__init__()

        self.linear = nn.Linear(d_model, 1)

    def forward(self, output_emb, query_emb=None):
        """
        :param output_emb: (batch_size, num_docs, doc_emb_size) transformed sequence of document embeddings
        :param query_emb: not used
        :return: (batch_size, num_docs) relevance scores in [0, 1]
        """

        return F.sigmoid(self.linear(output_emb))


class MDSTransformer(nn.Module):
    r"""Multiple Document Scoring Transformer. By default, consists of a Roberta query enconder (Huggingface implementation),
    and a "decoder" using self-attention over a sequence (set) of document representations and cross-attention over
    the query term representations in the  encoder output.
    Computes a relevance score for each (transformed) document representation and can be thus used for reranking.

    Args:
        encoder_config: huggingface transformers configuration object for a Roberta query encoder.
            Used instead of `custom_encoder`.
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

    def __init__(self, encoder_config=None, d_model: int = 768, num_heads: int = 8,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", positional_encoding=None,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 scoring_mode=None, loss_type=None) -> None:
        super(MDSTransformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = RobertaModel(encoder_config)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.num_heads = num_heads

        self.query_dim = self.encoder.config.hidden_size  # e.g. 768 for BERT-base

        # project query representation vectors to match dimensionality of doc embeddings (for cross-attention)
        if self.query_dim != self.d_model:
            self.project_query = nn.Linear(self.query_dim, self.d_model)

        self.score_docs = self.get_scoring_module(scoring_mode)

        self.loss_func = self.get_loss_func(loss_type)

        # Without any init call, weight parameters are initialized as by default when creating torch layers (Kaiming uniform)
        # self._reset_parameters()

    def get_scoring_module(self, scoring_mode):

        if scoring_mode == 'cross_attention':
            return CrossAttentionScorer(self.d_model)

    def get_loss_func(self, loss_type):
        return nn.MultiLabelMarginLoss()

    def forward(self, query_token_ids: Tensor, query_mask: Tensor = None, doc_emb: Tensor = None,
                docinds: Tensor = None, local_emb_mat: Tensor = None, doc_padding_mask: Tensor = None,
                doc_attention_mat_mask: Tensor = None, labels: Tensor = None) -> Dict[str, Tensor]:
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
                    This is for causality and is directly added on top of the attention matrix
        :param  labels: (batch_size, num_docs) int tensor which for each query (row) contains the indices of the
                relevant documents within its corresponding pool of candidates (docinds).
                    Optional: If provided, the loss will be computed.

        :returns:
            dict containing:
                rel_scores: (batch_size, num_docs) relevance scores in [0, 1]
                loss: scalar mean loss over entire batch (only if `labels` is provided!)
        """

        if 'doc_emb' is None:  # happens only in training, when additionally there is in-batch negative sampling
            doc_emb = self.lookup_doc_emb(docinds, local_emb_mat)  # (batch_size, max_docs_per_query, doc_emb_dim)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]
        doc_emb = doc_emb.permute(1, 0, 2)  # (max_docs_per_query, batch_size, doc_emb_dim) document embeddings

        if query_token_ids.size(0) != doc_emb.size(1):
            raise RuntimeError("the batch size for queries and documents must be equal")

        enc_hidden_states = self.encoder(query_token_ids.to(torch.int64), attention_mask=query_mask)  # int64 required by torch nn.Embedding
        if self.query_dim != self.d_model:  # project query representation vectors to match dimensionality of doc embeddings
            enc_hidden_states = self.project_query(enc_hidden_states)

        # The nn.MultiHeadAttention expects ByteTensor or Boolean and uses the convention that non-0 is ignored
        # and 0 is used in attention, which is the opposite of HuggingFace.
        memory_key_padding_mask = ~(query_mask.bool())

        # (num_docs, batch_size, doc_emb_size) transformed sequence of document embeddings
        output_emb = self.decoder(doc_emb, enc_hidden_states, tgt_mask=doc_attention_mat_mask,
                              tgt_key_padding_mask=doc_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # output_emb = self.act(output_emb)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output_emb = output_emb.permute(1, 0, 2)  # (batch_size, num_docs, doc_emb_size)

        rel_scores = self.score_docs(output_emb)  # (batch_size, num_docs) relevance scores in [0, 1]

        if labels is not None:
            loss = self.loss_func(rel_scores, labels)  # scalar tensor
            return {'loss': loss, 'rel_scores': rel_scores}
        return {'rel_scores': rel_scores}

    def lookup_doc_emb(self, docinds, local_emb_mat):
        """
        Lookup document vectors in `local_emb_mat` corresponding to rows given in `docinds`.
        This is done to avoid replicating embedding vectors of in-batch negatives, thus sparing GPU bandwidth.
        Global matrix cannot be used, because the collection size is in the order of 10M: GPU memory!
        """
        embedding = torch.nn.Embedding.from_pretrained(local_emb_mat, freeze=True, padding_idx=local_emb_mat.shape[0]-1)
        return embedding(docinds.to(torch.int64))

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
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

