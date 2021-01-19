import math
import logging
from typing import Optional, Any

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.modeling_bert import BertModel, BertPreTrainedModel, RobertaModel

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
    Depending on HF BERT implementation, this may be inefficient: attention masks typically are just added, instead
    of preventing computations altogether. This means that the full O(seq_len^2) cost is there, and is therefore worse
    than O(query_len^2) + O(doc_len^2). Some minor benefit comes from avoiding separate padding for query.
    """
    def __init__(self, config):
        super(RepBERT_Train, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, valid_mask,
                position_ids, labels=None):
        # labels: (batch_size, batch_size) padded (with -1) tensor of relevant doc in-batch (local) indices
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
    Takes a single type of input (either query or doc)
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


class MDSTransformer(nn.Module):
    r"""Multiple Document Scoring Transformer. By default, consists of a Roberta query enconder (Huggingface implementation),
    and a "decoder" using self-attention over a sequence (set) of document representations and cross-attention over
    the query term representations in the  encoder output.
    Computes a relevance score for each (transformed) document representation and can be thus used for reranking.

    Args:
        encoder_config: huggingface transformers configuration object for query encoder
        ***Decoder:***
        d_model: the "decoder" representation (hidden) dimension. Is also the doc embedding dimension
        num_heads: the number of heads in the multiheadattention models.
        num_decoder_layers: the number of sub-decoder-layers in the decoder.
        dim_feedforward: the dimension of the feedforward network model.
        dropout: the dropout value.
        activation: the activation function of encoder/decoder intermediate layer, "relu" or "gelu".
        positional_encoding: if None, no positional encoding is used for the "decoder", and thus the output is permutation
            equivariant with respect to the document embedding sequence
        **************
        custom_encoder: custom encoder object initialized externally (default=None).
        custom_decoder: custom decoder object initialized externally (default=None).

    Examples::
        >>> model = MDSTransformer(enc_config, nhead=16, num_decoder_layers=4)
        >>> model = MDSTransformer(custom_encoder=my_HF_encoder, nhead=16, num_decoder_layers=4)
    """

    def __init__(self, encoder_config=None, d_model: int = 512, num_heads: int = 8,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", positional_encoding=None,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(MDSTransformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = RobertaModel.__init__(self, encoder_config)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.num_heads = num_heads

        self.query_dim = self.encoder.config.hidden_size  # e.g. 768 for BERT-base
        if self.query_dim != self.d_model:  # project query representation vectors to match dimensionality of doc embeddings
            self.project_query = nn.Linear(self.query_dim, self.d_model)

        # Without any init call, weight parameters are initialized as by default when creating torch layers (Kaiming uniform)
        # self._reset_parameters()

    def forward(self, query_token_ids: Tensor, doc_emb: Tensor, query_mask: Optional[Tensor] = None,
                doc_attention_mat_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                doc_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            query_token_ids: (batch_size, query_length) padded sequence of token IDs fed to the encoder
            doc_emb: (num_docs, batch_size, doc_emb_size) sequence of document embeddings fed to the decoder
            query_mask: (batch_size, query_length) attention mask for query tokens. 0 means ignore, non-0 means use
            doc_attention_mat_mask: (num_docs, num_docs) float additive mask for the decoder sequence (optional).
                This is for causality and is directly added on top of the attention matrix
            doc_padding_mask: (batch_size, num_docs) boolean/ByteTensor mask in case the number of input documents
                is less than the max. doc. pool size, i.e. decoder sequence length (optional).

        :returns:
            out_doc_emb: (num_docs, batch_size, doc_emb_size) transformed sequence of document embeddings
        Examples:
            >>> output = transformer_model(query_token_ids, doc_emb, src_mask=src_mask, tgt_mask=doc_attention_mat_mask)
        """

        if query_token_ids.size(1) != doc_emb.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if query_token_ids.size(2) != self.d_model or doc_emb.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        enc_hidden_states = self.encoder(query_token_ids, attention_mask=query_mask)
        if self.query_dim != self.d_model:  # project query representation vectors to match dimensionality of doc embeddings
            enc_hidden_states = self.project_query(enc_hidden_states)

        # The nn.MultiHeadAttention expects ByteTensor or Boolean and uses the convention that non-0 is ignored
        # and 0 is used in attention, which is the opposite of HuggingFace.
        memory_key_padding_mask = ~(query_mask.bool())
        output = self.decoder(doc_emb, enc_hidden_states, tgt_mask=doc_attention_mat_mask,
                              tgt_key_padding_mask=doc_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

