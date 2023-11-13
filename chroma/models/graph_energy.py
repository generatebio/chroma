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

"""Models for building energy functions for protein sequence and structure.

This module contains pytorch models for building energy functions that score
protein sequence and structure and that can be used for partial and full
protein de novo design.
"""


import torch.nn as nn

from chroma.layers import graph


class GraphHarmonicFeatures(nn.Module):
    """Layer for quadratic node and edge features.

    Args:
        dim_nodes (int): Hidden dimension of node tensor.
        dim_edges (int): Hidden dimension of edge tensor.
        node_mlp_layers (int): Number of hidden layers for node update.
        node_mlp_dim (int): Node update function, hidden dimension.
        edge_mlp_layers (int): Edge update function, number of hidden layers.
        edge_mlp_dim (int): Edge update function, hidden dimension.
        mlp_activation (str): MLP nonlinearity.
            `'relu'`: Rectified linear unit.
            `'softplus'`: Softplus.

    Inputs:
        node_h (Tensor): Node embeddings with shape
            `(num_batch, num_nodes, dim_nodes)`.
        node_feature (Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (Tensor): Edge embeddings with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_feature (Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (tensor, optional): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (tensor, optional): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.

    Outputs:
        node_h (Tensor): Updated node embeddings with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (Tensor): Updated edge embeddings with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
    """

    def __init__(
        self,
        dim_nodes,
        dim_edges,
        node_mlp_layers,
        node_mlp_dim,
        edge_mlp_layers,
        edge_mlp_dim,
        mlp_activation="softplus",
        dropout=0.0,
    ):
        super(GraphHarmonicFeatures, self).__init__()
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.node_mlp = graph.MLP(
            dim_in=dim_nodes,
            dim_out=2 * dim_nodes,
            num_layers_hidden=node_mlp_layers,
            dim_hidden=node_mlp_dim,
            activation=mlp_activation,
            dropout=dropout,
        )
        self.edge_mlp = graph.MLP(
            dim_in=dim_edges,
            dim_out=2 * dim_edges,
            num_layers_hidden=edge_mlp_layers,
            dim_hidden=edge_mlp_dim,
            activation=mlp_activation,
            dropout=dropout,
        )
        self.node_out = nn.Linear(dim_nodes, dim_nodes)
        self.edge_out = nn.Linear(dim_edges, dim_edges)

    def forward(self, node_h, node_feature, edge_h, edge_feature, mask_i, mask_ij):
        node_h_pred = self.node_mlp(node_h)
        node_mu = node_h_pred[:, :, : self.dim_nodes]
        node_coeff = node_h_pred[:, :, self.dim_nodes :]
        node_errors = node_coeff * (node_feature - node_mu) ** 2
        node_h = node_h + self.node_out(node_errors)
        node_h = mask_i.unsqueeze(-1) * node_h

        edge_h_pred = self.edge_mlp(edge_h)
        edge_mu = edge_h_pred[:, :, :, : self.dim_edges]
        edge_coeff = edge_h_pred[:, :, :, self.dim_edges :]
        edge_errors = edge_coeff * (edge_feature - edge_mu) ** 2
        edge_h = edge_h + self.edge_out(edge_errors)
        edge_h = mask_ij.unsqueeze(-1) * edge_h
        return node_h, edge_h
