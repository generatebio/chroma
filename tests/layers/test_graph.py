from unittest import TestCase

import numpy as np
import pytest
import torch

from chroma.layers.graph import (
    MLP,
    GraphLayer,
    GraphNN,
    collect_edges_transpose,
    edge_mask_causal,
    permute_graph_embeddings,
)


class Testcollect_edges_transpose(TestCase):
    # Simple case of 3 noddes that are connected to each other
    edge_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
    mask_ij = torch.tensor([[[1, 1], [1, 1], [1, 1]]])
    edge_h = torch.tensor([[[[1], [2]], [[3], [4]], [[5], [6]]]])

    edge_h_transpose, mask_ji = collect_edges_transpose(edge_h, edge_idx, mask_ij)

    # Manually inspected the tensor so that it work
    # I view(-1) so that it is easier to write
    assert (
        torch.tensor([3.0, 5.0, 1.0, 6.0, 2.0, 4.0]) != edge_h_transpose.view(-1)
    ).detach().numpy().sum() == 0
    # Assert that shape stay the sample_input
    assert edge_h.shape == edge_h_transpose.shape

    # Kind of dumb, but if all mask, all edege shoudl be zero
    edge_h_transpose, mask_ji = collect_edges_transpose(
        edge_h, edge_idx, torch.zeros_like(mask_ij)
    )
    assert edge_h_transpose.abs().sum() == 0

    # Masking connection between 1,2
    mask_ij = torch.tensor([[[1, 1], [1, 0], [1, 0]]])
    edge_h_transpose, mask_ji = collect_edges_transpose(edge_h, edge_idx, mask_ij)
    print(edge_h_transpose.view(-1))
    assert (
        torch.tensor([3.0, 5.0, 1.0, 0.0, 2.0, 0.0]) != edge_h_transpose.view(-1)
    ).detach().numpy().sum() == 0

    # Masking 0 vers 2 mais pas 2 vers 0
    # 2 vers 0 should be masked in the transpose
    mask_ij = torch.tensor([[[1, 0], [1, 0], [1, 0]]])
    edge_h_transpose, mask_ji = collect_edges_transpose(edge_h, edge_idx, mask_ij)
    assert (
        torch.tensor([3.0, 0.0, 1.0, 0.0, 0.0, 0.0]) != edge_h_transpose.view(-1)
    ).detach().numpy().sum() == 0


class TestGraphNN(TestCase):
    def test_sample(self):
        dim_nodes = 128
        dim_edges = 64

        model = GraphNN(num_layers=6, dim_nodes=dim_nodes, dim_edges=dim_edges,)

        num_nodes = 10
        num_neighbors = 8
        node_h_out, edge_h_out = model(
            torch.ones(1, num_nodes, dim_nodes),
            torch.ones(1, num_nodes, num_neighbors, dim_edges),
            torch.ones(1, num_nodes, num_neighbors, dtype=torch.long),
        )
        self.assertTrue(node_h_out.shape == (1, num_nodes, dim_nodes))
        self.assertTrue(edge_h_out.shape == (1, num_nodes, num_neighbors, dim_edges))


class TestGraphLayer(TestCase):
    def test_sample(self):

        dim_nodes = 128
        dim_edges = 64

        graph_layer = GraphLayer(
            dim_nodes=dim_nodes, dim_edges=dim_edges, dropout=0, edge_update=True
        )

        num_parameters = sum([np.prod(p.size()) for p in graph_layer.parameters()])

        # self.assertEqual(num_parameters, 131712)

        num_nodes = 10
        num_neighbors = 8
        node_h_out, edge_h_out = graph_layer(
            torch.ones(1, num_nodes, dim_nodes),
            torch.ones(1, num_nodes, num_neighbors, dim_edges),
            torch.ones(1, num_nodes, num_neighbors, dtype=torch.long),
        )
        self.assertTrue(node_h_out.shape == (1, num_nodes, dim_nodes))
        self.assertTrue(edge_h_out.shape == (1, num_nodes, num_neighbors, dim_edges))


class TestMLP(TestCase):
    def test_sample(self):
        dim_in = 10
        sample_input = torch.rand(dim_in)
        prediction = MLP(dim_in)(sample_input)
        self.assertTrue(prediction.shape[-1] == dim_in)

        sample_input = torch.rand(dim_in)
        dim_out = 8
        model = MLP(dim_in, dim_out=dim_out)
        prediction = model(sample_input)
        self.assertTrue(prediction.shape[-1] == dim_out)

        sample_input = torch.rand(dim_in)
        dim_hidden = 5
        model = MLP(dim_in, dim_hidden=5, dim_out=5)
        prediction = model(sample_input)
        self.assertTrue(prediction.shape[-1] == dim_hidden)

        sample_input = torch.rand(dim_in)

        model = MLP(dim_in, num_layers_hidden=0, dim_out=dim_out)
        prediction = model(sample_input)
        self.assertTrue(prediction.shape[-1] == dim_out)


class TestGraphFunctions(TestCase):
    def hello():
        print("hello")


def test_graph_permutation():
    B, N, K, H = 2, 7, 4, 3
    # Create a random graph embedding
    node_h = torch.randn([B, N, H])
    edge_h = torch.randn([B, N, K, H])
    edge_idx = torch.randint(low=0, high=N, size=[B, N, K])
    mask_i = torch.ones([B, N])
    mask_ij = torch.ones([B, N, K])

    # Create a random permutation matrix embedding
    permute_idx = torch.argsort(torch.randn([B, N]), dim=-1)

    # Permute
    node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p = permute_graph_embeddings(
        node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
    )

    # Inverse permute
    permute_idx_inverse = torch.argsort(permute_idx, dim=-1)
    node_h_pp, edge_h_pp, edge_idx_pp, mask_i_pp, mask_ij_pp = permute_graph_embeddings(
        node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p, permute_idx_inverse
    )

    # Test round-trip of permutation . inverse permutation
    assert torch.allclose(node_h, node_h_pp)
    assert torch.allclose(edge_h, edge_h_pp)
    assert torch.allclose(edge_idx, edge_idx_pp)
    assert torch.allclose(mask_i, mask_i_pp)
    assert torch.allclose(mask_ij, mask_ij_pp)

    # Test permutation equivariance of GNN layers
    gnn = GraphNN(num_layers=1, dim_nodes=H, dim_edges=H)
    outs = gnn(node_h, edge_h, edge_idx, mask_i, mask_ij)
    outs_perm = gnn(node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p)
    outs_pp = permute_graph_embeddings(
        outs_perm[0], outs_perm[1], edge_idx_p, mask_i_p, mask_ij_p, permute_idx_inverse
    )

    assert torch.allclose(outs[0], outs_pp[0])
    assert torch.allclose(outs[1], outs_pp[1])
    return


def test_autoregressive_gnn():
    B, N, K, H = 1, 3, 3, 4

    torch.manual_seed(0)

    # Build random GNN input
    node_h = torch.randn([B, N, H])
    edge_h = torch.randn([B, N, K, H])
    # edge_idx = torch.randint(low=0, high=N, size=[B, N, K])
    edge_idx = torch.arange(K).reshape([1, 1, K]).expand([B, N, K]).contiguous()
    mask_i = torch.ones([B, N])
    mask_ij = torch.ones([B, N, K])
    mask_ij = edge_mask_causal(edge_idx, mask_ij)

    error = lambda x, y: (torch.abs(x - y) / (torch.abs(y) + 1e-3)).mean()

    # Parallel mode computation
    for mode in [True, False]:
        gnn = GraphNN(num_layers=4, dim_nodes=H, dim_edges=H, attentional=mode)

        node_h_gnn, edge_h_gnn = gnn(node_h, edge_h, edge_idx, mask_i, mask_ij)

        # Step wise computation
        node_h_cache, edge_h_cache = gnn.init_steps(node_h, edge_h)
        for t in range(N):
            node_h_cache, edge_h_cache = gnn.step(
                t, node_h_cache, edge_h_cache, edge_idx, mask_i, mask_ij
            )
        node_h_sequential = node_h_cache[-1]
        edge_h_sequential = edge_h_cache[-1]

        assert torch.allclose(node_h_gnn, node_h_sequential)
        assert torch.allclose(edge_h_gnn, edge_h_sequential)
    return
