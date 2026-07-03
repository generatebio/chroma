# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


def _normalize_attention_mask(mask, n_head):
    if mask is None:
        return None
    if mask.dim() == 3:
        return mask.unsqueeze(1).expand(-1, n_head, -1, -1)
    return mask


def _scaled_attention(q, k, v, mask=None, bias=None):
    logits = torch.einsum("bqhc,bkhc->bhqk", q, k) / q.size(-1) ** 0.5
    if bias is not None:
        logits = logits + bias
    # Fix #1: mask before softmax so masked keys don't inflate the denominator
    # and the output remains a true convex combination of valid values.
    # nan_to_num(0.0) guards all-masked query rows (e.g. ghost chains in batches
    # with heterogeneous chain counts) which would otherwise produce NaN.
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    weights = torch.nn.functional.softmax(logits, dim=-1)
    weights = weights.nan_to_num(0.0)
    output = torch.einsum("bhqk,bkhc->bqhc", weights, v)
    return output, weights


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention as described in Eqn 1 of Vaswani et al. 2017 [https://arxiv.org/abs/1706.03762].

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Note that the dimension of the query has to match the dimension of the keys (here specified as ```d_k```) and the length of keys has to match
    the length of the values. See for instance 'The Illustrated Transformer' [http://jalammar.github.io/illustrated-transformer/]
    for pictorial depiction of attention.

    Inputs:
        Q (torch.tensor): of shape (batch_size, sequence_length_q, d_k)
        K (torch.tensor):  of shape (batch_size, sequence_length_k, d_k)
        V (torch.tensor):  of shape (batch_size, sequence_length_k, d_v)
        mask (torch.tensor):  of dtype (bool) or (byte) and shape (batch_size, 1, sequence_length_k), optional
             zeroes (or False) indicate positions that cannot contribute to attention
    Outputs:
        output (torch.tensor) of shape (batch_size, sequence_length_q, d_v). The [i-j]-entry output[i,j,:] is formed as a convex combination of values:
        \\sum_k a_k V[i,k,:] and \\sum_k a_k = 1.
        attentions (torch.tensor) of shape (batch_size, sequence_length_q, sequence_length_k)) where the [b,i,j]-element
        corresponds to the attention value (e.g relative contribution) of position j in the key-tensor to position i in the query tensor in element b of the batch.
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        output, attn = _scaled_attention(
            Q.unsqueeze(2),
            K.unsqueeze(2),
            V.unsqueeze(2),
            mask=None if mask is None else mask.unsqueeze(1),
        )
        output = output.squeeze(2)
        attn = attn.squeeze(1)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention with scaled dot product attention. See 'The Annotated Transformer'
    http://nlp.seas.harvard.edu/2018/04/03/attention.html or 'The Illustrated Transformer' http://jalammar.github.io/illustrated-transformer/
    for details and intuition.

     Args:
         n_head (int): number of attention heads
         d_k (int): dimension of the keys and queries in each attention head
         d_v (int): dimension of the values in each attention head
         d_model (int): input and output dimension for the layer
         dropout (float): dropout rate, default is 0.1

    Inputs:
        Q (torch.tensor): query tensor of shape ```(batch_size, sequence_length_q, d_model)```
        K (torch.tensor):  key tensor of shape ```(batch_size, sequence_length_k, d_model)```
        V (torch.tensor): value tensor of shape ```(batch_size, sequence_length_k, d_model)```
        mask (torch.tensor): (optional) of dtype ```bool`` or ```byte``` and size (batch_size, 1, sequence_length_k),
                    zeroes (or False) indicate positions that cannot contribute to attention

    Outputs:
        output (torch.tensor) :  of shape ```(batch_size, sequence_length_q, d_model)```
        attentions (torch.tensor): of shape ```(batch_size * n_head, sequence_length_q, sequence_length_k) where
        ```attentions[batch_size*(i):batch_size*(i+1),:,:]``` corresponds to the batch of attention blocks for i'th head. See
        ```chroma.layers.attention.ScaledDotProductAttention``` for more details
    """

    def __init__(self, n_head, d_k, d_v, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.Wq = nn.Parameter(torch.Tensor(n_head, d_model, d_k))
        self.Wk = nn.Parameter(torch.Tensor(n_head, d_model, d_k))
        self.Wv = nn.Parameter(torch.Tensor(n_head, d_model, d_v))
        self.Wo = nn.Parameter(torch.Tensor(n_head * d_v, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.Wk)
        nn.init.xavier_normal_(self.Wv)
        nn.init.kaiming_uniform_(self.Wo)

    def _project(self, x, weights):
        # Compute all head projections in one batched contraction to avoid
        # Python loops and repeated concatenation across heads.
        projected = torch.einsum("bld,hdm->hblm", x, weights)
        return projected.reshape(self.n_head * x.size(0), x.size(1), weights.size(-1))

    def forward(self, Q, K, V, bias=None, mask=None):
        mb_size, len_q, d_q_in = Q.size()
        mb_size, len_k, d_k_in = K.size()
        mb_size, len_v, d_v_in = V.size()
        d_model = self.d_model
        if d_q_in != d_model:
            raise ValueError("Dimension of Q does not match d_model.")

        if d_k_in != d_model:
            raise ValueError("Dimension of K does not match d_model.")

        if d_v_in != d_model:
            raise ValueError("Dimension of V does not match d_model.")

        # Treat the heads as an expanded batch without materializing one
        # projection pass per head in Python.
        q_s = self._project(Q, self.Wq)
        k_s = self._project(K, self.Wk)
        v_s = self._project(V, self.Wv)

        # Attention
        q = q_s.reshape(self.n_head, mb_size, len_q, self.d_k).permute(1, 2, 0, 3)
        k = k_s.reshape(self.n_head, mb_size, len_k, self.d_k).permute(1, 2, 0, 3)
        v = v_s.reshape(self.n_head, mb_size, len_v, self.d_v).permute(1, 2, 0, 3)
        outputs, attns = _scaled_attention(
            q,
            k,
            v,
            mask=_normalize_attention_mask(mask, self.n_head),
        )

        # Fold the head-major batch back to (batch, query, head * value_dim).
        outputs = outputs.reshape(mb_size, len_q, self.n_head * self.d_v)
        attns = attns.permute(1, 0, 2, 3).reshape(
            self.n_head * mb_size, len_q, len_k
        )

        # Project back to residual size
        outputs = outputs @ self.Wo
        outputs = self.dropout(outputs)
        return outputs, attns


class AttentionChainPool(nn.Module):
    """Pools residue-based representations to chain-based representations using a chain mask and attention.
    Args:
        n_head (int): number of attention heads
        d_model (int): dimension of embeddings to be pooled

    Inputs:
        h (torch.tensor): of size (batch_size, sequence_length, d_model)
        C (torch.tensor): of size (batch_size, sequence_length)

    Outputs:
        output (torch.tensor): of size (batch_size, n_chains, d_model)
        chain_mask (torch.tensor): of size (batch_size, n_chains)
    """

    def __init__(self, n_head, d_model):
        super().__init__()
        self.attention = Attention(
            n_head, d_model, d_k=d_model, d_v=d_model, gate=False
        )

    def get_query(self, x, num_queries=1):
        return x.new_ones(x.size(0), num_queries, x.size(2))

    def forward(self, h, C):
        chain_ids = C.abs().unique()
        chain_ids = chain_ids[chain_ids > 0]
        num_chains = len(chain_ids)

        mask = C.abs().unsqueeze(1) == chain_ids.view(1, num_chains, 1)
        output = self.attention(self.get_query(h, num_queries=num_chains), h, h, mask=mask)
        chain_mask = mask.any(dim=-1)
        return output, chain_mask


class Attention(nn.Module):
    """
    A multi-head attention layer with optional gating and bias as implemented in Jumper et al. (2021)
    Args:
        n_head (int): Number of heads of attention
        d_model (int): Dimension of input and outputs
        d_k (int): Dimension of keys/queries
        d_v (int): Dimension of values
        gate (bool): Whether to include a gate connection (as in Jumper et al. (2021))

    Inputs:
        Q (torch.tensor): of size (batch_size, num_queries, d_model)
        K (torch.tensor): of size (batch_size, num_keys, d_model)
        V (torch.tensor): of size (batch_size, num_keys, d_model)
        bias (torch.tensor): (optional) of size (batch_size, n_head, num_queries, num_keys)
        mask (torch.tensor): (optional) of size (batch_size, n_head, num_queries, num_keys)

    Outputs:
        output (torch.tensor): of size (batch_size, num_queries, d_model)
    """

    def __init__(self, n_head, d_model, d_k=None, d_v=None, gate=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head if d_k is None else d_k
        self.d_v = d_model // n_head if d_v is None else d_v
        self.gate = gate
        self.q_weights = nn.Parameter(torch.Tensor(d_model, n_head, self.d_k))
        self.k_weights = nn.Parameter(torch.Tensor(d_model, n_head, self.d_k))
        self.v_weights = nn.Parameter(torch.Tensor(d_model, n_head, self.d_v))
        self.o_weights = nn.Parameter(torch.Tensor(n_head, self.d_v, d_model))
        self.o_bias = nn.Parameter(torch.Tensor(d_model))
        if self.gate:
            self.g_weights = nn.Parameter(torch.Tensor(d_model, n_head, self.d_v))
            self.g_bias = nn.Parameter(torch.Tensor(n_head, self.d_v))
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_weights)
        nn.init.xavier_uniform_(self.k_weights)
        nn.init.xavier_uniform_(self.v_weights)
        nn.init.xavier_uniform_(self.o_weights)
        nn.init.zeros_(self.o_bias)
        if self.gate:
            nn.init.zeros_(self.g_weights)
            nn.init.ones_(self.g_bias)

    def forward(self, Q, K, V, bias=None, mask=None):
        self._check_inputs(Q, K, V, bias, mask)
        q = torch.einsum("bqa,ahc->bqhc", Q, self.q_weights)
        k = torch.einsum("bka,ahc->bkhc", K, self.k_weights)
        v = torch.einsum("bka,ahc->bkhc", V, self.v_weights)
        weighted_avg, _ = _scaled_attention(
            q, k, v, mask=_normalize_attention_mask(mask, self.n_head), bias=bias
        )

        if self.gate:
            gate_values = torch.einsum("bqa,ahc->bqhc", Q, self.g_weights) + self.g_bias
            gate_values = torch.sigmoid(gate_values)
            weighted_avg = weighted_avg * gate_values

        output = (
            torch.einsum("bqhc,hco->bqo", weighted_avg, self.o_weights) + self.o_bias
        )
        return output

    def _check_inputs(self, Q, K, V, bias, mask):
        batch_size_q, num_queries, d_q_in = Q.size()
        batch_size_k, num_keys, d_k_in = K.size()
        batch_size_v, num_values, d_v_in = V.size()

        if d_q_in != self.d_model:
            raise ValueError(
                f"Dimension of Q tensor needs to be (batch_size, number_queries, d_model)"
            )

        if d_k_in != self.d_model:
            raise ValueError(
                f"Dimension of K tensor needs to be (batch_size, number_keys, d_model)"
            )

        if d_v_in != self.d_model:
            raise ValueError(
                f"Dimension of V tensor needs to be (batch_size, number_values, d_model)"
            )

        if num_keys != num_values:
            raise ValueError(f"Number of keys needs to match number of values passed")

        if (batch_size_q != batch_size_k) or (batch_size_k != batch_size_v):
            raise ValueError(
                f"Found batch size mismatch among inputs, all tensors must agree in size of dimension 0"
            )

        if bias is not None:
            if (bias.dim() != 3) and (bias.dim() != 4):
                raise ValueError(
                    f"Bias specified but dimension mismatched: passed {bias.dim()}-dimensional tensor but should be 3-dimensional"
                    f"of shape (n_head, num_queries, num_keys) or 4-dimensional of shape (batch_size, n_head, num_queries, num_keys)"
                )
            if bias.dim() == 3:
                n_head_b, num_queries_b, num_keys_b = bias.size()
                if n_head_b != self.n_head:
                    raise ValueError(
                        f"Bias specified but number of heads (dim of axis=0) does not match number of heads: {self.n_head}"
                    )
                if num_queries_b != num_queries:
                    raise ValueError(
                        f"Bias specified but number of queries (dim of axis=1) does not match number of queries given in Q tensor"
                    )
                if num_keys_b != num_keys:
                    raise ValueError(
                        f"Bias specified but number of keys (dim of axis=2) does not match number of queries given in K tensor "
                        f"(dimenson of axis=1)"
                    )
            elif bias.dim() == 4:
                if bias.dim() == 3:
                    n_batch_b, n_head_b, num_queries_b, num_keys_b = bias.size()
                    if n_head_b != self.n_head:
                        raise ValueError(
                            f"Bias specified but number of heads (dim of axis=0) does not match number of heads: {self.n_head}"
                        )
                    if num_queries_b != num_queries:
                        raise ValueError(
                            f"Bias specified but number of queries (dim of axis=1) does not match number of queries given in Q tensor"
                        )
                    if num_keys_b != num_keys:
                        raise ValueError(
                            f"Bias specified but number of keys (dim of axis=2) does not match number of queries given in K tensor "
                            f"(dimenson of axis=1)"
                        )

        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError(
                    f"Mask specified but not given by correct dtype, should be torch.bool but found {mask.dtype}"
                )
            if (mask.dim() != 3) and (mask.dim() != 4):
                raise ValueError(
                    f"Mask specified but dimension mismatched: passed {mask.dim()}-dimensional tensor but should be 3-dimensional"
                    f" of shape (batch_size, num_queries, num_keys) or 4-dimensional"
                    f" of shape (batch_size, n_head, num_queries, num_keys)"
                )
            if mask.dim() == 3:
                batch_size_b, num_queries_b, num_keys_b = mask.size()
            else:
                batch_size_b, _, num_queries_b, num_keys_b = mask.size()
            if batch_size_b != batch_size_q:
                raise ValueError(
                    f"Mask specified but batch dimension does not match number of examples given in Q tensor"
                )
            if (num_queries_b != num_queries) and (num_queries_b != 1):
                raise ValueError(
                    f"Bias specified but number of queries (dim of axis=2) does not match number of queries given in Q tensor"
                )
            if (num_keys_b != num_keys) and (num_keys_b != 1):
                raise ValueError(
                    f"Bias specified but number of keys (dim of axis=3) does not match number of queries given in K tensor "
                    f"(dimenson of axis=1)"
                )
