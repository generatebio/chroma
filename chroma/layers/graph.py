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

"""Layers for building graph neural networks.

This module contains layers for building neural networks that can process
graph-structured data. The internal representations of these layers
are node and edge embeddings.
"""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm.autonotebook import tqdm

from chroma.layers.attention import Attention


class GraphNN(nn.Module):
    """Graph neural network with optional edge updates.

    Args:
        num_layers (int): Number of layers.
        dim_nodes (int): Hidden dimension of node tensor.
        dim_edges (int): Hidden dimension of edge tensor.
        dropout (float): Dropout rate.
        node_mlp_layers (int): Node update function, number of hidden layers.
            Default is 1.
        node_mlp_dim (int): Node update function, hidden dimension.
            Default is to match MLP output dimension.
        update_edge (Boolean): Include an edge-update step. Default: True
        edge_mlp_layers (int): Edge update function, number of hidden layers.
            Default is 1.
        edge_mlp_dim (int): Edge update function, hidden dimension.
            Default is to match MLP output dimension.
        mlp_activation (str): MLP nonlinearity.
            `'relu'`: Rectified linear unit.
            `'softplus'`: Softplus.
        norm (str): Which normalization function to apply between layers.
            `'transformer'`: Default layernorm
            `'layer'`: Masked Layer norm with shape (input.shape[1:])
            `'instance'`: Masked Instance norm
        scale (float): Scaling factor of edge input when updating node (default=1.0)
        attentional (bool): If True, use attention for message aggregation function
            instead of a sum. Default is False.
        num_attention_heads (int): Number of attention heads (if attentional) to use.
            Default is 4.

    Inputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (tensor, optional): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (tensor, optional): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Outputs:
        node_h_out (torch.Tensor): Updated node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h_out (torch.Tensor): Updated edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
    """

    def __init__(
        self,
        num_layers: int,
        dim_nodes: int,
        dim_edges: int,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        mlp_activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "transformer",
        scale: float = 1.0,
        skip_connect_input: bool = False,
        attentional: bool = False,
        num_attention_heads: int = 4,
        checkpoint_gradients: bool = False,
    ):
        super(GraphNN, self).__init__()

        self.skip_connect_input = skip_connect_input
        self.checkpoint_gradients = checkpoint_gradients
        self.layers = nn.ModuleList(
            [
                GraphLayer(
                    dim_nodes=dim_nodes,
                    dim_edges=dim_edges,
                    node_mlp_layers=node_mlp_layers,
                    node_mlp_dim=node_mlp_dim,
                    edge_update=edge_update,
                    edge_mlp_layers=edge_mlp_layers,
                    edge_mlp_dim=edge_mlp_dim,
                    mlp_activation=mlp_activation,
                    dropout=dropout,
                    norm=norm,
                    scale=scale,
                    attentional=attentional,
                    num_attention_heads=num_attention_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run every layer sequentially
        node_h_init = node_h
        edge_h_init = edge_h
        for i, layer in enumerate(self.layers):
            if self.skip_connect_input:
                node_h = node_h + node_h_init
                edge_h = edge_h + edge_h_init

            # Update edge and node
            node_h, edge_h = self.checkpoint(
                layer, node_h, edge_h, edge_idx, mask_i, mask_ij
            )

            if self.skip_connect_input:
                node_h = node_h - node_h_init
                edge_h = edge_h - edge_h_init

            # If mask was provided, apply it
            if mask_i is not None:
                node_h = node_h * (mask_i.unsqueeze(-1) != 0).type(torch.float32)
            if mask_ij is not None:
                edge_h = edge_h * (mask_ij.unsqueeze(-1) != 0).type(torch.float32)
        return node_h, edge_h

    def checkpoint(self, layer, *args):
        if self.checkpoint_gradients:
            return checkpoint(layer, *args)
        else:
            return layer(*args)

    def sequential(
        self,
        tensors: dict,
        pre_step_function: Callable = None,
        post_step_function: Callable = None,
    ) -> dict:
        """Decode the GNN sequentially along the node index `t`, with callbacks.

        Args:
            tensors (dict): Initial set of state tensors. At minimum this should
                include the arguments to `forward`, namely `node_h`, `edge_h`,
                `edge_idx`, `mask_i`, and `mask_ij`.
            pre_step_function (function, optional): Callback function that is
                optionally applied to `tensors` before each sequential GNN step as
                `tensors_new = pre_step_function(t, pre_step_function)` where `t` is
                the node index being updated. It should update elements of the
                `tensors` dictionary, and it can access and update the intermediate
                GNN state cache via the keyed lists of tensors in `node_h_cache` and
                `edge_h_cache`.
            post_step_function (function, optional): Same as `pre_step_function`, but
                optionally applied after each sequential GNN step.

        Returns:
            tensors (dict): Processed set of tensors.
        """

        # Initialize the state cache
        tensors["node_h_cache"], tensors["edge_h_cache"] = self.init_steps(
            tensors["node_h"], tensors["edge_h"]
        )

        # Sequential iteration
        num_steps = tensors["node_h"].size(1)
        for t in tqdm(range(num_steps), desc="Sequential decoding"):
            if pre_step_function is not None:
                tensors = pre_step_function(t, tensors)

            tensors["node_h_cache"], tensors["edge_h_cache"] = self.step(
                t,
                tensors["node_h_cache"],
                tensors["edge_h_cache"],
                tensors["edge_idx"],
                tensors["mask_i"],
                tensors["mask_ij"],
            )

            if post_step_function is not None:
                tensors = post_step_function(t, tensors)

        return tensors

    def init_steps(
        self, node_h: torch.Tensor, edge_h: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Initialize cached node and edge features.

        Args:
            node_h (torch.Tensor): Node features with shape
                `(num_batch, num_nodes, dim_nodes)`.
            edge_h (torch.Tensor): Edge features with shape
                `(num_batch, num_nodes, num_neighbors, dim_edges)`.

        Returns:
            node_h_cache (torch.Tensor): List of cached node features with `num_layers + 1`
                tensors of shape `(num_batch, num_nodes, dim_nodes)`.
            edge_h_cache (torch.Tensor): List of cached edge features with `num_layers + 1`
                tensors of shape `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        """
        num_layers = len(self.layers)
        node_h_cache = [node_h.clone() for _ in range(num_layers + 1)]
        edge_h_cache = [edge_h.clone() for _ in range(num_layers + 1)]
        return node_h_cache, edge_h_cache

    def step(
        self,
        t: int,
        node_h_cache: List[torch.Tensor],
        edge_h_cache: List[torch.Tensor],
        edge_idx: torch.LongTensor,
        mask_i: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Process GNN update for a specific node index t from cached intermediates.

        Inputs:
            t (int): Node index to decode.
            node_h_cache (List[torch.Tensor]): List of cached node features with
                `num_layers + 1` tensors of shape `(num_batch, num_nodes, dim_nodes)`.
            edge_h_cache (List[torch.Tensor]): List of cached edge features with
                `num_layers + 1` tensors of shape
                `(num_batch, num_nodes, num_neighbors, dim_edges)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_nodes, num_neighbors)`.
            mask_i (torch.Tensor, optional): Node mask with shape
                `(num_batch, num_nodes)`.
            mask_ij (torch.Tensor, optional): Edge mask with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            node_h_cache (List[torch.Tensor]): Updated list of cached node features
                with `num_layers + 1` tensors of shape
                `(num_batch, num_nodes, dim_nodes)`. This method updates the tensors
                in place for memory.
            edge_h_cache (List[torch.Tensor]): Updated list of cached edge features
                with `num_layers + 1` tensors of shape
                `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        """
        if self.skip_connect_input:
            raise NotImplementedError

        for i, layer in enumerate(self.layers):
            # Because the edge updates depend on the updated nodes,
            # we need both the input node features node_h and also
            # the previous output node states node_h
            node_h = node_h_cache[i]
            node_h_out = node_h_cache[i + 1]
            edge_h = edge_h_cache[i]
            # Update edge and node
            node_h_t, edge_h_t = checkpoint(
                layer.step, t, node_h, node_h_out, edge_h, edge_idx, mask_i, mask_ij
            )

            # Scatter them in place
            node_h_cache[i + 1].scatter_(
                1, (t * torch.ones_like(node_h_t)).long(), node_h_t
            )
            edge_h_cache[i + 1].scatter_(
                1, (t * torch.ones_like(edge_h_t)).long(), edge_h_t
            )

        return node_h_cache, edge_h_cache


class GraphLayer(nn.Module):
    """Graph layer that updates each node i given adjacent nodes and edges.

    Args:
        dim_nodes (int): Hidden dimension of node tensor.
        dim_edges (int): Hidden dimension of edge tensor.
        node_mlp_layers (int): Node update function, number of hidden layers.
            Default: 1.
        node_mlp_dim (int): Node update function, hidden dimension.
            Default: Matches MLP output dimension.
        update_edge (Boolean): Include an edge-update step. Default: True
        edge_mlp_layers (int): Edge update function, number of hidden layers.
            Default: 1.
        edge_mlp_dim (int): Edge update function, hidden dimension.
            Default: Matches MLP output dimension.
        mlp_activation (str): MLP nonlinearity.
            `'relu'`: Rectified linear unit.
            `'softplus'`: Softplus.
        dropout (float): Dropout rate.
        norm (str): Which normalization function to apply between layers.
            `'transformer'`: Default layernorm
            `'layer'`: Masked Layer norm with shape (input.shape[1:])
            `'instance'`: Masked Instance norm
        scale (float): Scaling factor of edge input when updating node (default=1.0)

    Inputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (tensor, optional): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (tensor, optional): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Outputs:
        node_h_out (torch.Tensor): Updated node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h_out (torch.Tensor): Updated edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_nodes)`.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        mlp_activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "transformer",
        scale: float = 1.0,
        attentional: bool = False,
        num_attention_heads: int = 4,
    ):
        super(GraphLayer, self).__init__()

        # Store scale
        self.scale = scale
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.attentional = attentional

        self.node_norm_layer = MaskedNorm(
            dim=1, num_features=dim_nodes, affine=True, norm=norm
        )

        self.message_mlp = MLP(
            dim_in=2 * dim_nodes + dim_edges,
            dim_out=dim_nodes,
            num_layers_hidden=edge_mlp_layers,
            dim_hidden=edge_mlp_dim,
            activation=mlp_activation,
            dropout=dropout,
        )
        self.update_mlp = MLP(
            dim_in=2 * dim_nodes,
            dim_out=dim_nodes,
            num_layers_hidden=node_mlp_layers,
            dim_hidden=node_mlp_dim,
            activation=mlp_activation,
            dropout=dropout,
        )
        self.edge_update = edge_update
        self.edge_norm_layer = MaskedNorm(
            dim=2, num_features=dim_edges, affine=True, norm=norm
        )
        if self.edge_update:
            self.edge_mlp = MLP(
                dim_in=2 * dim_nodes + dim_edges,
                dim_out=dim_edges,
                num_layers_hidden=edge_mlp_layers,
                dim_hidden=edge_mlp_dim,
                activation=mlp_activation,
                dropout=dropout,
            )

        if self.attentional:
            self.attention = Attention(n_head=num_attention_heads, d_model=dim_nodes)

    def attend(
        self, node_h: torch.Tensor, messages: torch.Tensor, mask_ij: torch.Tensor
    ) -> torch.Tensor:
        B, L, K, D = messages.size()
        queries = node_h.reshape(-1, 1, D)
        keys = messages.reshape(-1, K, D)
        values = messages.reshape(-1, K, D)
        mask = mask_ij.reshape(-1, 1, 1, K).bool() if mask_ij is not None else None
        return self.attention(queries, keys, values, mask=mask).reshape(B, L, D)

    def _normalize(self, node_h, edge_h, mask_i=None, mask_ij=None):
        # Normalize node and edge embeddings
        node_h_norm = self.node_norm_layer(node_h, mask_i)
        edge_h_norm = self.edge_norm_layer(edge_h, mask_ij)
        return node_h_norm, edge_h_norm

    def _normalize_t(
        self, edge_node_stack_t, mask_ij_t, include_nodes=True, include_edges=True
    ):
        # Apply normalization (since we have only normalized time t information)
        node_i_t = edge_node_stack_t[:, :, :, : self.dim_nodes]
        node_j_t = edge_node_stack_t[:, :, :, self.dim_nodes : 2 * self.dim_nodes]
        edge_h_t = edge_node_stack_t[:, :, :, 2 * self.dim_nodes :]
        if include_nodes:
            node_i_t = self.node_norm_layer(node_i_t, mask_ij_t)
            node_j_t = self.node_norm_layer(node_j_t, mask_ij_t)
        if include_edges:
            edge_h_t = self.edge_norm_layer(edge_h_t, mask_ij_t)
        edge_node_stack_t = torch.cat([node_i_t, node_j_t, edge_h_t], -1)
        return edge_node_stack_t

    def _update_nodes(
        self, node_h, node_h_norm, edge_h_norm, edge_idx, mask_i=None, mask_ij=None
    ):
        """Update nodes given adjacent nodes and edges"""
        # Compute messages at each ij
        edge_node_stack = pack_edges(node_h_norm, edge_h_norm, edge_idx)
        messages = self.message_mlp(edge_node_stack)
        if mask_ij is not None:
            messages = messages * mask_ij.unsqueeze(-1)

        # Aggregate messages
        if self.attentional:
            message = self.attend(node_h_norm, messages, mask_ij)
        else:
            message = messages.sum(2) / self.scale

        node_stack = torch.cat([node_h_norm, message], -1)

        # Update nodes given aggregated messages
        node_h_out = node_h + self.update_mlp(node_stack)
        if mask_i is not None:
            node_h_out = node_h_out * mask_i.unsqueeze(-1)
        return node_h_out

    def _update_nodes_t(
        self,
        t,
        node_h,
        node_h_norm_t,
        edge_h_norm_t,
        edge_idx_t,
        mask_i_t=None,
        mask_ij_t=None,
    ):
        """Update nodes at index t given adjacent nodes and edges"""
        # Compute messages at each ij
        edge_node_stack_t = mask_ij_t.unsqueeze(-1) * pack_edges_step(
            t, node_h, edge_h_norm_t, edge_idx_t
        )

        # Apply normalization of gathered tensors
        edge_node_stack_t = self._normalize_t(
            edge_node_stack_t, mask_ij_t, include_edges=False
        )

        messages_t = self.message_mlp(edge_node_stack_t)
        if mask_ij_t is not None:
            messages_t = messages_t * mask_ij_t.unsqueeze(-1)

        # Aggregate messages
        if self.attentional:
            message_t = self.attend(node_h_norm_t, messages_t, mask_ij_t)
        else:
            message_t = messages_t.sum(2) / self.scale

        node_stack_t = torch.cat([node_h_norm_t, message_t], -1)
        # Update nodes given aggregated messages
        node_h_t = node_h[:, t, :].unsqueeze(1)
        node_h_out_t = node_h_t + self.update_mlp(node_stack_t)
        if mask_i_t is not None:
            node_h_out_t = node_h_out_t * mask_i_t.unsqueeze(-1)
        return node_h_out_t

    def _update_edges(self, edge_h, node_h_out, edge_h_norm, edge_idx, mask_ij):
        """Update edges given adjacent nodes and edges"""
        edge_node_stack = pack_edges(node_h_out, edge_h_norm, edge_idx)

        edge_h_out = edge_h + self.edge_mlp(edge_node_stack)
        if mask_ij is not None:
            edge_h_out = edge_h_out * mask_ij.unsqueeze(-1)
        return edge_h_out

    def _update_edges_t(
        self, t, edge_h_t, node_h_out, edge_h_t_norm, edge_idx_t, mask_ij_t
    ):
        """Update edges given adjacent nodes and edges"""
        edge_node_stack_t = pack_edges_step(t, node_h_out, edge_h_t_norm, edge_idx_t)

        edge_h_out_t = edge_h_t + self.edge_mlp(edge_node_stack_t)
        if mask_ij_t is not None:
            edge_h_out_t = edge_h_out_t * mask_ij_t.unsqueeze(-1)
        return edge_h_out_t

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ):
        node_h_norm, edge_h_norm = self._normalize(node_h, edge_h, mask_i, mask_ij)
        if mask_i is not None:
            mask_i = (mask_i != 0).type(torch.float32)
        if mask_ij is not None:
            mask_ij = (mask_ij != 0).type(torch.float32)
        node_h_out = self._update_nodes(
            node_h, node_h_norm, edge_h_norm, edge_idx, mask_i, mask_ij
        )
        edge_h_out = None
        if self.edge_update:
            edge_h_out = self._update_edges(
                edge_h, node_h_out, edge_h_norm, edge_idx, mask_ij
            )
        return node_h_out, edge_h_out

    def step(
        self,
        t: int,
        node_h: torch.Tensor,
        node_h_out: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ):
        """Compute update for a single node index `t`.

        This function can be useful for sequential computation of graph
        updates, for example with autoregressive architectures.

        Args:
            t (int): Index of node dimension to update
            node_h (torch.Tensor): Node features with shape
                `(num_batch, num_nodes, dim_nodes)`.
            node_h_out (torch.Tensor): Cached outputs of preceding steps with shape
                `(num_batch, num_nodes, dim_nodes)`.
            edge_h (torch.Tensor): Edge features with shape
                `(num_batch, num_nodes, num_neighbors, dim_edges)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_nodes, num_neighbors)`.
            mask_i (tensor, optional): Node mask with shape `(num_batch, num_nodes)`
            mask_ij (tensor, optional): Edge mask with shape
                `(num_batch, num_nodes, num_neighbors)`

        Resturns:
            node_h_t (torch.Tensor): Updated node features with shape
                `(num_batch, 1, dim_nodes)`.
            edge_h_t (torch.Tensor): Updated edge features with shape
                `(num_batch, 1, num_neighbors, dim_nodes)`.
        """
        node_h_t = node_h[:, t, :].unsqueeze(1)
        edge_h_t = edge_h[:, t, :, :].unsqueeze(1)
        edge_idx_t = edge_idx[:, t, :].unsqueeze(1)
        mask_i_t = mask_i[:, t].unsqueeze(1)
        mask_ij_t = mask_ij[:, t, :].unsqueeze(1)

        """ For a single step we need to apply the normalization both at node t and
            also for all of the neighborhood tensors that feed in at t.
        """
        node_h_t_norm, edge_h_t_norm = self._normalize(
            node_h_t, edge_h_t, mask_i_t, mask_ij_t
        )
        node_h_t = self._update_nodes_t(
            t, node_h, node_h_t_norm, edge_h_t_norm, edge_idx_t, mask_i_t, mask_ij_t
        )

        if self.edge_update:
            node_h_out = node_h_out.scatter(
                1, (t * torch.ones_like(node_h_t)).long(), node_h_t
            )
            edge_h_t = self._update_edges_t(
                t, edge_h_t, node_h_out, edge_h_t_norm, edge_idx_t, mask_ij_t
            )
        return node_h_t, edge_h_t


class MLP(nn.Module):
    """Multilayer perceptron with variable input, hidden, and output dims.

    Args:
        dim_in (int): Feature dimension of input tensor.
        dim_hidden (int or None): Feature dimension of intermediate layers.
            Defaults to matching output dimension.
        dim_out (int or None): Feature dimension of output tensor.
            Defaults to matching input dimension.
        num_layers_hidden (int): Number of hidden MLP layers.
        activation (str): MLP nonlinearity.
            `'relu'`: Rectified linear unit.
            `'softplus'`: Softplus.
        dropout (float): Dropout rate. Default is 0.

    Inputs:
        h (torch.Tensor): Input tensor with shape `(..., dim_in)`

    Outputs:
        h (torch.Tensor): Input tensor with shape `(..., dim_in)`
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: Optional[int] = None,
        dim_out: Optional[int] = None,
        num_layers_hidden: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        # Default is dimension preserving
        dim_out = dim_out if dim_out is not None else dim_in
        dim_hidden = dim_hidden if dim_hidden is not None else dim_out

        nonlinearites = {"relu": nn.ReLU, "softplus": nn.Softplus}
        activation_func = nonlinearites[activation]

        if num_layers_hidden == 0:
            layers = [nn.Linear(dim_in, dim_out)]
        else:
            layers = []
            for i in range(num_layers_hidden):
                d_1 = dim_in if i == 0 else dim_hidden
                layers = layers + [
                    nn.Linear(d_1, dim_hidden),
                    activation_func(),
                    nn.Dropout(dropout),
                ]
            layers = layers + [nn.Linear(dim_hidden, dim_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layers(h)


def collect_neighbors(node_h: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Collect neighbor node features as edge features.

    For each node i, collect the embeddings of neighbors {j in N(i)} as edge
    features neighbor_ij.

    Args:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, num_features)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        neighbor_h (torch.Tensor): Edge features containing neighbor node information
            with shape `(num_batch, num_nodes, num_neighbors, num_features)`.
    """
    num_batch, num_nodes, num_neighbors = edge_idx.shape
    num_features = node_h.shape[2]

    # Flatten for the gather operation then reform the full tensor
    idx_flat = edge_idx.reshape([num_batch, num_nodes * num_neighbors, 1])
    idx_flat = idx_flat.expand(-1, -1, num_features)
    neighbor_h = torch.gather(node_h, 1, idx_flat)
    neighbor_h = neighbor_h.reshape((num_batch, num_nodes, num_neighbors, num_features))
    return neighbor_h


def collect_edges(
    edge_h_dense: torch.Tensor, edge_idx: torch.LongTensor
) -> torch.Tensor:
    """Collect sparse edge features from a dense pairwise tensor.

    Args:
        edge_h_dense (torch.Tensor): Dense edges features with shape
            `(num_batch, num_nodes, num_nodes, num_features)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        edge_h (torch.Tensor): Edge features with shape
            (num_batch, num_nodes, num_neighbors, num_features)`.
    """
    gather_idx = edge_idx.unsqueeze(-1).expand(-1, -1, -1, edge_h_dense.size(-1))
    edge_h = torch.gather(edge_h_dense, 2, gather_idx)
    return edge_h


def collect_edges_transpose(
    edge_h: torch.Tensor, edge_idx: torch.LongTensor, mask_ij: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect edge embeddings of reversed (transposed) edges in-place.

    Args:
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, num_features_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Returns:
        edge_h_transpose (torch.Tensor): Edge features of transpose with shape
            `(num_batch, num_nodes, num_neighbors, num_features_edges)`.
        mask_ji (torch.Tensor): Mask indicating presence of reversed edge with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """
    num_batch, num_residues, num_k, num_features = list(edge_h.size())

    # Get indices of reverse edges
    ij_to_ji, mask_ji = transpose_edge_idx(edge_idx, mask_ij)

    # Gather features at reverse edges
    edge_h_flat = edge_h.reshape(num_batch, num_residues * num_k, -1)
    ij_to_ji = ij_to_ji.unsqueeze(-1).expand(-1, -1, num_features)
    edge_h_transpose = torch.gather(edge_h_flat, 1, ij_to_ji)
    edge_h_transpose = edge_h_transpose.reshape(
        num_batch, num_residues, num_k, num_features
    )
    edge_h_transpose = mask_ji.unsqueeze(-1) * edge_h_transpose
    return edge_h_transpose, mask_ji


def scatter_edges(edge_h: torch.Tensor, edge_idx: torch.LongTensor) -> torch.Tensor:
    """Scatter sparse edge features into a dense pairwise tensor.
    Args:
         edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, num_features_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        edge_h_dense (torch.Tensor): Dense edge features with shape
            `(batch_size, num_nodes, num_nodes, dimensions)`.
    """
    assert edge_h.dim() == 4
    assert edge_idx.dim() == 3
    bs, nres, _, dim = edge_h.size()
    edge_indices = edge_idx.unsqueeze(-1).repeat(1, 1, 1, dim)
    result = torch.zeros(
        size=(bs, nres, nres, dim), dtype=edge_h.dtype, device=edge_h.device,
    )
    return result.scatter(dim=2, index=edge_indices, src=edge_h)


def pack_edges(
    node_h: torch.Tensor, edge_h: torch.Tensor, edge_idx: torch.LongTensor
) -> torch.Tensor:
    """Pack nodes and edge features into edge features.

    Expands each edge_ij by packing node i, node j, and edge ij into
    {node,node,edge}_ij.

    Args:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, num_features_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, num_features_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        edge_packed (torch.Tensor): Concatenated node and edge features with shape
            (num_batch, num_nodes, num_neighbors, num_features_nodes
                + 2*num_features_edges)`.
    """
    num_neighbors = edge_h.shape[2]
    node_i = node_h.unsqueeze(2).expand(-1, -1, num_neighbors, -1)
    node_j = collect_neighbors(node_h, edge_idx)
    edge_packed = torch.cat([node_i, node_j, edge_h], -1)
    return edge_packed


def pack_edges_step(
    t: int, node_h: torch.Tensor, edge_h_t: torch.Tensor, edge_idx_t: torch.LongTensor
) -> torch.Tensor:
    """Pack node and edge features into edge features for a single node index t.

    Expands each edge_ij by packing node i, node j, and edge ij into
    {node,node,edge}_ij.

    Args:
        t (int): Node index to decode.
        node_h (torch.Tensor): Node features at all positions with shape
            `(num_batch, num_nodes, num_features_nodes)`.
        edge_h_t (torch.Tensor): Edge features at index `t` with shape
            `(num_batch, 1, num_neighbors, num_features_edges)`.
        edge_idx_t (torch.LongTensor): Edge indices at index `t` for neighbors with shape
            `(num_batch, 1, num_neighbors)`.

    Returns:
        edge_packed (torch.Tensor): Concatenated node and edge features
            for index `t` with shape
            (num_batch, 1, num_neighbors, num_features_nodes
                + 2*num_features_edges)`.
    """
    num_nodes_i = node_h.shape[1]
    num_neighbors = edge_h_t.shape[2]
    node_h_t = node_h[:, t, :].unsqueeze(1)
    node_i = node_h_t.unsqueeze(2).expand(-1, -1, num_neighbors, -1)
    node_j = collect_neighbors(node_h, edge_idx_t)
    edge_packed = torch.cat([node_i, node_j, edge_h_t], -1)
    return edge_packed


def transpose_edge_idx(
    edge_idx: torch.LongTensor, mask_ij: torch.Tensor
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Collect edge indices of reverse edges in-place at each edge.

    The tensor `edge_idx` stores a directed graph topology as a tensor of
    neighbor indices, where an element `edge_idx[b,i,k]` corresponds to the
    node index of neighbor `k` of node `i` in batch member `b`.

    This function takes a directed graph topology and returns an index tensor
    that maps, in-place, to the reversed edges (if they exist). The indices
    correspond to the contracted dimension of `edge_index` when it is viewed as
    `(num_batch, num_nodes * num_neighbors)`. These indices can be used in
    conjunction with `torch.gather` to collect edge embeddings of `j->i` at
    `i->j`. See `collect_edges_transpose` for an example.

    For reverse `j->i` edges that do not exist in the directed graph, the
    function also returns a binary mask `mask_ji` indicating which edges
    have both `i->j` and `j->i` present in the graph.

    Args:
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Returns:
        ij_to_ji (torch.LongTensor): Flat indices for indexing ji in-place at ij with
            shape `(num_batch, num_nodes * num_neighbors)`.
        mask_ji (torch.Tensor): Mask indicating presence of reversed edge with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """
    num_batch, num_residues, num_k = list(edge_idx.size())

    # 1. Collect neighbors of neighbors
    edge_idx_flat = edge_idx.reshape([num_batch, num_residues * num_k, 1]).expand(
        -1, -1, num_k
    )
    edge_idx_neighbors = torch.gather(edge_idx, 1, edge_idx_flat)
    # (b,i,j,k) gives the kth neighbor of the jth neighbor of i
    edge_idx_neighbors = edge_idx_neighbors.reshape(
        [num_batch, num_residues, num_k, num_k]
    )

    # 2. Determine which k at j maps back to i (if it exists)
    residue_i = torch.arange(num_residues, device=edge_idx.device).reshape(
        (1, -1, 1, 1)
    )
    edge_idx_match = (edge_idx_neighbors == residue_i).type(torch.float32)
    return_mask, return_idx = torch.max(edge_idx_match, -1)

    # 3. Build flat indices
    ij_to_ji = edge_idx * num_k + return_idx
    ij_to_ji = ij_to_ji.reshape(num_batch, -1)

    # 4. Transpose mask
    mask_ji = torch.gather(mask_ij.reshape(num_batch, -1), -1, ij_to_ji)
    mask_ji = mask_ji.reshape(num_batch, num_residues, num_k)
    mask_ji = mask_ij * return_mask * mask_ji
    return ij_to_ji, mask_ji


def permute_tensor(
    tensor: torch.Tensor, dim: int, permute_idx: torch.LongTensor
) -> torch.Tensor:
    """Permute a tensor along a dimension given a permutation vector.

    Args:
        tensor (torch.Tensor): Input tensor with shape
            `([batch_dims], permutation_length, [content_dims])`.
        dim (int): Dimension to permute along.
        permute_idx (torch.LongTensor): Permutation index tensor with shape
            `([batch_dims], permutation_length)`.

    Returns:
        tensor_permute (torch.Tensor): Permuted node features with shape
            `([batch_dims], permutation_length, [content_dims])`.
    """
    # Resolve absolute dimension
    dim = range(len(list(tensor.shape)))[dim]

    # Flatten content dimensions
    shape = list(tensor.shape)
    batch_dims, permute_length = shape[:dim], shape[dim]
    tensor_flat = tensor.reshape(batch_dims + [permute_length] + [-1])

    # Exap content dimensions
    permute_idx_expand = permute_idx.unsqueeze(-1).expand(tensor_flat.shape)

    tensor_permute_flat = torch.gather(tensor_flat, dim, permute_idx_expand)
    tensor_permute = tensor_permute_flat.reshape(tensor.shape)
    return tensor_permute


def permute_graph_embeddings(
    node_h: torch.Tensor,
    edge_h: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    permute_idx: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """Permute graph embeddings given a permutation vector.

    Args:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (tensor, optional): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (tensor, optional): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
        permute_idx (torch.LongTensor): Permutation vector with shape
            `(num_batch, num_nodes)`.

    Returns:
        node_h_permute (torch.Tensor): Permuted node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h_permute (torch.Tensor): Permuted edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx_permute (torch.LongTensor): Permuted edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i_permute (tensor, optional): Permuted node mask with shape `(num_batch, num_nodes)`
        mask_ij_permute (tensor, optional): Permuted edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
    """

    # Permuting one-dimensional objects is straightforward gathering
    node_h_permute = permute_tensor(node_h, 1, permute_idx)
    edge_h_permute = permute_tensor(edge_h, 1, permute_idx)
    mask_i_permute = permute_tensor(mask_i, 1, permute_idx)
    mask_ij_permute = permute_tensor(mask_ij, 1, permute_idx)

    """
    For edge_idx, there are two-dimensions set each edge idx that
    previously pointed to j to now point to the new location
    of j which is p^(-1)[j]
    edge^(p)[i,k] = p^(-1)[edge[p(i),k]]
    """
    # First, permute on the i dimension
    edge_idx_permute_1 = permute_tensor(edge_idx, 1, permute_idx)
    # Second, permute on the j dimension by using the inverse
    permute_idx_inverse = torch.argsort(permute_idx, dim=-1)
    edge_idx_1_flat = edge_idx_permute_1.reshape([edge_idx.shape[0], -1])
    edge_idx_permute_flat = torch.gather(permute_idx_inverse, 1, edge_idx_1_flat)
    edge_idx_permute = edge_idx_permute_flat.reshape(edge_idx.shape)

    return (
        node_h_permute,
        edge_h_permute,
        edge_idx_permute,
        mask_i_permute,
        mask_ij_permute,
    )


def edge_mask_causal(edge_idx: torch.LongTensor, mask_ij: torch.Tensor) -> torch.Tensor:
    """Make an edge mask causal with mask_ij = 0 for j >= i.

    Args:
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_nodes, num_neighbors)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        mask_ij_causal (torch.Tensor): Causal edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """
    idx = torch.arange(edge_idx.size(1), device=edge_idx.device)
    idx_expand = idx.reshape([1, -1, 1])
    mask_ij_causal = (edge_idx < idx_expand).float() * mask_ij
    return mask_ij_causal


class MaskedNorm(nn.Module):
    """Masked normalization layer.

    Args:
        dim (int): Dimensionality of the normalization. Can be 1 for 1D
            normalization along dimension 1 or 2 for 2D normalization along
            dimensions 1 and 2.
        num_features (int): Channel dimension; only needed if `affine` is True.
        affine (bool): If True, inclde a learnable affine transformation
            post-normalization. Default is False.
        norm (str): Type of normalization, can be `instance`, `layer`, or
            `transformer`.
        eps (float): Small number for numerical stability.

    Inputs:
        data (torch.Tensor): Input tensor with shape
            `(num_batch, num_nodes, num_channels)` (1D) or
            `(num_batch, num_nodes, num_nodes, num_channels)` (2D).
        mask (torch.Tensor): Mask tensor with shape
            `(num_batch, num_nodes)` (1D) or
            `(num_batch, num_nodes, num_nodes)` (2D).

    Outputs:
        norm_data (torch.Tensor): Mask-normalized tensor with shape
            `(num_batch, num_nodes, num_channels)` (1D) or
            `(num_batch, num_nodes, num_nodes, num_channels)` (2D).
    """

    def __init__(
        self,
        dim: int,
        num_features: int = -1,
        affine: bool = False,
        norm: str = "instance",
        eps: float = 1e-5,
    ):
        super(MaskedNorm, self).__init__()

        self.norm_type = norm
        self.dim = dim
        self.norm = norm + str(dim)
        self.affine = affine
        self.eps = eps

        # Dimension to sum
        if self.norm == "instance1":
            self.sum_dims = [1]
        elif self.norm == "layer1":
            self.sum_dims = [1, 2]
        elif self.norm == "transformer1":
            self.sum_dims = [-1]
        elif self.norm == "instance2":
            self.sum_dims = [1, 2]
        elif self.norm == "layer2":
            self.sum_dims = [1, 2, 3]
        elif self.norm == "transformer2":
            self.sum_dims = [-1]
        else:
            raise NotImplementedError

        # Number of features, only required if affine
        self.num_features = num_features

        # Affine transformation is a linear layer on the C channel
        if self.affine:
            self.weights = nn.Parameter(torch.rand(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(
        self, data: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Add optional trailing singleton dimension and expand if necessary
        if mask is not None:
            if len(mask.shape) == len(data.shape) - 1:
                mask = mask.unsqueeze(-1)
            if data.shape != mask.shape:
                mask = mask.expand(data.shape)

        # Input shape is Batch, Channel, Dim1, (dim2 if 2d)
        dims = self.sum_dims
        if (mask is None) or (self.norm_type == "transformer"):
            mask_mean = data.mean(dim=dims, keepdim=True)
            mask_std = torch.sqrt(
                (((data - mask_mean)).pow(2)).mean(dim=dims, keepdim=True) + self.eps
            )

            # Norm
            norm_data = (data - mask_mean) / mask_std

        else:
            # Zeroes vector to sum all mask data
            norm_data = torch.zeros_like(data).to(data.device).type(data.dtype)
            for mask_id in mask.unique():
                # Skip zero, since real mask
                if mask_id == 0:
                    continue

                # Transform mask to temp mask that match mask id
                tmask = (mask == mask_id).type(torch.float32)

                # Sum mask for mean
                mask_sum = tmask.sum(dim=dims, keepdim=True)

                # Data is tmask, so that mean is only for unmasked pos
                mask_mean = (data * tmask).sum(dim=dims, keepdim=True) / mask_sum
                mask_std = torch.sqrt(
                    (((data - mask_mean) * tmask).pow(2)).sum(dim=dims, keepdim=True)
                    / mask_sum
                    + self.eps
                )

                # Calculate temp norm, apply mask
                tnorm = ((data - mask_mean) / mask_std) * tmask
                # Sometime mask is empty, so generate nan that are conversted to 0
                tnorm[tnorm != tnorm] = 0

                # Add to init zero norm data
                norm_data += tnorm

        # Apply affine
        if self.affine:
            norm_data = norm_data * self.weights + self.bias

        # If mask, apply mask
        if mask is not None:
            norm_data = norm_data * (mask != 0).type(data.dtype)
        return norm_data
