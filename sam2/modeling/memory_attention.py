# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

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

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn.self_attn(q, k = k, v = tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory_1, memory_2, query_pos, pos_1, pos_2):
        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image.cross_attn(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k_1=memory_1 + pos_1 if self.pos_enc_at_cross_attn_keys else memory_1,
            v_1=memory_1,
            k_2=memory_2 + pos_2 if self.pos_enc_at_cross_attn_keys else memory_2,
            v_2=memory_2
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory_1,
        memory_2,
        pos_1: Optional[Tensor] = None,
        pos_2: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory_1, memory_2, query_pos, pos_1, pos_2)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def allocate_rope_attention_weight(
        self,
        curr: torch.Tensor,  # self-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        image_size = 1024,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        output = curr

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)

        for layer in self.layers:
            if isinstance(layer.cross_attn_image, RoPEAttention):
                layer.cross_attn_image.allocate_rope_attention_weight(output, image_size = image_size)
            if isinstance(layer.self_attn, RoPEAttention):
                layer.self_attn.allocate_rope_attention_weight(output, image_size = image_size)

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory_1: torch.Tensor,  # cross-attention inputs
        memory_2: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos_1: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        memory_pos_2: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory_1.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory_1 = memory_1.transpose(0, 1)
            memory_2 = memory_2.transpose(0, 1)
            memory_pos_1 = memory_pos_1.transpose(0, 1)
            memory_pos_2 = memory_pos_2.transpose(0, 1)

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory_1=memory_1,
                memory_2=memory_2,
                pos_1=memory_pos_1,
                pos_2=memory_pos_2,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
