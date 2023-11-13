from unittest import TestCase

import numpy as np
import torch

from chroma.models import graph_energy


class TestGraphHarmonicFeatures(TestCase):
    def test_sample(self):
        num_batch = 1
        num_nodes = 10
        num_neighbors = 8
        dim_nodes = 128
        dim_edges = 64

        layer = graph_energy.GraphHarmonicFeatures(
            dim_nodes=dim_nodes,
            dim_edges=dim_edges,
            node_mlp_layers=2,
            node_mlp_dim=dim_nodes,
            edge_mlp_layers=2,
            edge_mlp_dim=dim_edges,
        )

        node_h = torch.ones(num_batch, num_nodes, dim_nodes)
        node_features = torch.ones(num_batch, num_nodes, dim_nodes)
        edge_h = torch.ones(num_batch, num_nodes, num_neighbors, dim_edges)
        edge_features = torch.ones(num_batch, num_nodes, num_neighbors, dim_edges)
        mask_i = torch.ones(num_batch, num_nodes)
        mask_ij = torch.ones(num_batch, num_nodes, num_neighbors)

        node_out, edge_out = layer(
            node_h, node_features, edge_h, edge_features, mask_i, mask_ij
        )

        self.assertTrue(node_out.shape == (1, num_nodes, dim_nodes))
        self.assertTrue(edge_out.shape == (1, num_nodes, num_neighbors, dim_edges))
