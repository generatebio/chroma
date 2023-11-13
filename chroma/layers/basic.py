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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chroma.layers.norm import MaskedBatchNorm1d


class NoOp(nn.Module):
    """A dummy nn.Module wrapping an identity operation.

    Inputs:
        x (any)

    Outputs:
        x (any)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class Transpose(nn.Module):
    """An nn.Module wrapping ```torch.transpose```.

    Args:
        d1 (int): the first (of two) dimensions to swap
        d2 (int): the second (of two) dimensions to swap

    Inputs:
        x (torch.tensor)

    Outputs:
        y (torch.tensor): ```y = x.transpose(d1,d2)```
    """

    def __init__(self, d1=1, d2=2):
        super().__init__()
        self.d1 = d1
        self.d2 = d2

    def forward(self, x):
        return x.transpose(self.d1, self.d2)


class Unsqueeze(nn.Module):
    """An nn.Module wrapping ```torch.unsqueeze```.

    Args:
        dim (int): the dimension to unsqueeze input tensors

    Inputs:
        x (torch.tensor):

    Outputs:
        y (torch.tensor): where ```y=x.unsqueeze(dim)```
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class OneHot(nn.Module):
    """An nn.Module that wraps F.one_hot```.

    Args:
        n_tokens (int): the number of tokens comprising input sequences

    Inputs:
        x (torch.LongTensor): of size ```(batch_size, *)```

    Outputs:
        y (torch.ByteTensor): of size (batch_size, *, n_tokens) cast to input.device
    """

    def __init__(self, n_tokens):
        super().__init__()
        self.n_tokens = n_tokens

    def forward(self, x):
        return F.one_hot(x, self.n_tokens)


class MeanEmbedding(nn.Module):
    """A wrapper around ```nn.Embedding``` that allows for one-hot-like representation inputs (as well as standard tokenized representation),
    optionally applying a softmax to the last dimension if the input corresponds to a log-PMF.
    Args:
        embedding (nn.Embedding): Embedding to wrap
        use_softmax (bool): Whether to apply a softmax to the last dimension if input is one-hot-like.

    Inputs:
        x (torch.tensor): of size (batch_size, sequence_length) (standard tokenized representation) -OR- (batch_size, sequence_length, number_tokens) (one-hot representation)

    Outputs:
        y (torch.tensor): of size (batch_size, sequence_length, embedding_dimension) obtained via. lookup into ```self.embedding.weight``` if
        input is in standard tokenized form or by matrix multiplication of input with ```self.embedding.weight``` if input is one-hot-like. Note
        that if the input is a one-hot matrix the output is the same regardless of representation.
    """

    def __init__(self, embedding, use_softmax=True):
        super(MeanEmbedding, self).__init__()
        self.embedding = embedding
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if len(x.shape) == 2:
            return self.embedding(x)
        elif len(x.shape) == 3:
            if self.use_softmax:
                return self.softmax(x) @ self.embedding.weight
            else:
                return x @ self.embedding.weight
        else:
            raise (NotImplementedError)


class PeriodicPositionalEncoding(nn.Module):
    """Positional encoding, adapted from 'The Annotated Transformer'
    http://nlp.seas.harvard.edu/2018/04/03/attention.html

     Args:
         d_model (int): input and output dimension for the layer
         max_seq_len (int): maximum allowed sequence length
         dropout (float): Dropout rate

    Inputs:
        x (torch.tensor): of size (batch_size, sequence_length, d_model)

    Outputs:
        y (torch.tensor): of size (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model, max_seq_len=4000, dropout=0.0):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0.0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward using 1x1 convolutions, a building block of legacy Transformer code (not code optimized).

    Args:
         d_model (int): input and output dimension for the layer
         d_inner_hid (int): size of the hidden layer in the position-wise feed-forward sublayer

    Inputs:
        x (torch.tensor): of size (batch_size, sequence_length, d_model)
    Outputs:
        y (torch.tensor): of size (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Conv1d(d_model, d_hidden, 1)
        self.linear2 = nn.Conv1d(d_hidden, d_model, 1)
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        output = self.activation(self.linear1(x.transpose(1, 2)))
        output = self.linear2(output).transpose(1, 2)
        return self.dropout(output)


class DropNormLin(nn.Module):
    """nn.Module applying a linear layer, normalization, dropout, and activation
    Args:
        in_features (int): input dimension
        out_features (int): output dimension
        norm_type (str): ```'ln'``` for layer normalization or ```'bn'``` for batch normalization else skip normalization
        dropout (float): dropout to apply
        actn (nn.Module): activation function to apply

    Input:
        x (torch.tensor): of size (batch_size, sequence_length, in_features)
        input_mask (torch.tensor): of size (batch_size, 1, sequence_length) (optional)

    Output:
        y (torch.tensor): of size (batch_size, sequence_length, out_features)
    """

    def __init__(
        self, in_features, out_features, norm_type="ln", dropout=0.0, actn=nn.ReLU()
    ):
        super(DropNormLin, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if norm_type == "ln":
            self.norm_layer = nn.LayerNorm(out_features)
        elif norm_type == "bn":
            self.norm_layer = MaskedBatchNorm1d(out_features)
        else:
            self.norm_layer = NoOp()
        self.dropout = nn.Dropout(p=dropout)
        self.actn = actn

    def forward(self, x, input_mask=None):
        h = self.linear(x)
        if isinstance(self.norm_layer, MaskedBatchNorm1d):
            h = self.norm_layer(h.transpose(1, 2), input_mask=input_mask).transpose(
                1, 2
            )
        else:
            h = self.norm_layer(h)
        return self.dropout(self.actn(h))


class ResidualLinearLayer(nn.Module):
    """A Simple Residual Layer using a linear layer a relu and an optional layer norm.

    Args:
        d_model (int): Model Dimension
        use_norm (bool, *optional*): Optionally Use a Layer Norm. Default `True`.
    """

    def __init__(self, d_model, use_norm=True):
        super(ResidualLinearLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.ReLU = nn.ReLU()
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        z = self.linear(x)
        z = self.ReLU(z)
        if self.use_norm:
            z = self.norm(z)
        return x + z


class TriangleMultiplication(nn.Module):
    def __init__(self, d_model=512, mode="outgoing"):
        """
          Triangle multiplication as defined in Jumper et al. (2021)
        Args:
            d_model (int): dimension of the embedding at each position
            mode (str): Must be 'outgoing' (algorithm 11) or 'incoming' (algorithm 12).

        Inputs:
            X (torch.tensor): Pair representations of size (batch, nres,  nres, channels)
            mask (torch.tensor): of dtype `torch.bool` and size (batch, nres, nres, channels) (or broadcastable to this size)

        Outputs:
            Y (torch.tensor): Pair representations of size (batch, nres,  nres, channels)
        """
        super().__init__()
        self.mode = mode
        assert self.mode in ["outgoing", "incoming"]
        self.equation = (
            "bikc,bjkc->bijc" if self.mode == "outgoing" else "bkjc,bkic->bijc"
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.left_edge_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Sigmoid(), nn.Linear(d_model, d_model)
        )
        self.right_edge_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Sigmoid(), nn.Linear(d_model, d_model)
        )
        self.skip = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.combine = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def forward(self, X, mask=None):
        h = self.layer_norm(X)

        A = self.left_edge_mlp(h)
        B = self.right_edge_mlp(h)
        G = self.skip(h)

        if mask is not None:
            A = A.masked_fill(~mask, 0.0)
            B = B.masked_fill(~mask, 0.0)

        h = torch.einsum(self.equation, A, B)
        h = self.combine(h) * G
        return h


class NodeProduct(nn.Module):
    """Like Alg. 10 in Jumper et al. (2021) but instead of computing a mean over MSA dimension,
    process for single-sequence inputs.
    Args:
        d_in (int): dimension of node embeddings (inputs)
        d_out (int): dimension of edge embeddings (outputs)

    Inputs:
        node_features (torch.tensor): of size (batch_size, nres, d_model)
        node_mask (torch.tensor): of size (batch_size, nres)
        edge_mask (torch.tensor): of size (batch_size, nres, nres)

    Outputs:
        edge_features (torch.tensor): of size (batch_size, nres, nres, d_model)
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_in)
        self.left_lin = nn.Linear(d_in, d_in)
        self.right_lin = nn.Linear(d_in, d_in)
        self.edge_lin = nn.Linear(2 * d_in, d_out)

    def forward(self, node_features, node_mask=None, edge_mask=None):
        _, nres, _ = node_features.size()

        node_features = self.layer_norm(node_features)
        left_embs = self.left_lin(node_features)
        right_embs = self.right_lin(node_features)

        if node_mask is not None:
            mask = node_mask[:, :, None]
            left_embs = left_embs.masked_fill(~mask, 0.0)
            right_embs = right_embs.masked_fill(~mask, 0.0)

        left_embs = left_embs[:, None, :, :].repeat(1, nres, 1, 1)
        right_embs = right_embs[:, :, None, :].repeat(1, 1, nres, 1)
        edge_features = torch.cat([left_embs, right_embs], dim=-1)
        edge_features = self.edge_lin(edge_features)

        if edge_mask is not None:
            mask = edge_mask[:, :, :, None]
            edge_features = edge_features.masked_fill(~mask, 0.0)

        return edge_features


class FourierFeaturization(nn.Module):
    """Applies fourier featurization of low-dimensional (usually spatial) input data as described in [https://arxiv.org/abs/2006.10739] ,
    optionally trainable as described in [https://arxiv.org/abs/2106.02795].

    Args:
        d_input (int): dimension of inputs
        d_model (int): dimension of outputs
        trainable (bool): whether to learn the frequency of fourier features
        scale (float): if not trainable, controls the scale of fourier feature periods (see reference for description, this parameter matters and should be tuned!)

    Inputs:
        input (torch.tensor): of size (batch_size, *, d_input)

    Outputs:
        output (torch.tensor): of size (batch_size, *, d_output)
    """

    def __init__(self, d_input, d_model, trainable=False, scale=1.0):
        super().__init__()
        self.scale = scale

        if d_model % 2 != 0:
            raise ValueError(
                "d_model needs to be even for this featurization, try again!"
            )

        B = 2 * math.pi * scale * torch.randn(d_input, d_model // 2)
        self.trainable = trainable
        if not trainable:
            self.register_buffer("B", B)
        else:
            self.register_parameter("B", torch.nn.Parameter(B))

    def forward(self, inputs):
        h = inputs @ self.B
        return torch.cat([h.cos(), h.sin()], -1)


class PositionalEncoding(nn.Module):
    """Axis-aligned positional encodings with log-linear spacing.

    Args:
        d_input (int): dimension of inputs
        d_model (int): dimension of outputs
        period_range (tuple of floats): Min and maximum periods for the
            frequency components. Fourier features will be log-linearly spaced
            between these values (inclusive).

    Inputs:
        input (torch.tensor): of size (..., d_input)

    Outputs:
        output (torch.tensor): of size (..., d_model)
    """

    def __init__(self, d_model, d_input=1, period_range=(1.0, 1000.0)):
        super().__init__()

        if d_model % (2 * d_input) != 0:
            raise ValueError(
                "d_model needs to be divisible by 2*d_input for this featurization, "
                f"but got {d_model} versus {d_input}"
            )

        num_frequencies = d_model // (2 * d_input)
        log_bounds = np.log10(period_range)
        p = torch.logspace(log_bounds[0], log_bounds[1], num_frequencies, base=10.0)
        w = 2 * math.pi / p
        self.register_buffer("w", w)

    def forward(self, inputs):
        batch_dims = list(inputs.shape)[:-1]
        # (..., 1, num_out) * (..., num_in, 1)
        w = self.w.reshape(len(batch_dims) * [1] + [1, -1])
        h = w * inputs[..., None]
        h = torch.cat([h.cos(), h.sin()], -1).reshape(batch_dims + [-1])
        return h


class MaybeOnehotEmbedding(nn.Embedding):
    """Wrapper around :class:`torch.nn.Embedding` to support either int-encoded
    LongTensors or one-hot encoded FloatTensors.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype.is_floating_point:  # onehot
            return x @ self.weight
        return super().forward(x)
