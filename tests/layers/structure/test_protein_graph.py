import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from chroma.data import Protein
from chroma.layers.structure.backbone import RigidTransformer
from chroma.layers.structure.protein_graph import ProteinFeatureGraph
from tests.helpers import cif_path


def test_protein_feature_graph():
    torch.manual_seed(10)

    dim_nodes, dim_edges = 128, 64
    num_neighbors = 30
    feature_graph = ProteinFeatureGraph(
        dim_nodes=dim_nodes,
        dim_edges=dim_edges,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=(
            "distances_6mer",
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
            "orientations_chain",
            "position_2mer",
        ),
        num_neighbors=num_neighbors,
        graph_kwargs={"mask_interfaces": False, "cutoff": None},
    )

    X, C, S = Protein(cif_path("5imm")).to_XCS()

    node_h, edge_h, edge_idx, mask_i, mask_ij = feature_graph(X, C)
    num_nodes = X.shape[1]

    # Test shapes
    assert node_h.shape == (1, num_nodes, dim_nodes)
    assert edge_h.shape == (1, num_nodes, num_neighbors, dim_edges)
    assert edge_idx.shape == (1, num_nodes, num_neighbors)
    assert mask_i.shape == (1, num_nodes)
    assert mask_ij.shape == (1, num_nodes, num_neighbors)

    # Test masks
    masked_sum_i = torch.abs((1.0 - mask_i).unsqueeze(-1) * node_h).sum()
    masked_sum_ij = torch.abs((1.0 - mask_ij).unsqueeze(-1) * edge_h).sum()
    assert masked_sum_i == 0
    assert masked_sum_ij == 0

    transformer = RigidTransformer(center_rotation=False)
    q_rotate = torch.Tensor([0.0, 0.1, -1.2, 0.5]).unsqueeze(0)
    dX_rotate = torch.Tensor([0.0, 1.0, -1.0]).unsqueeze(0)
    _rotate = lambda X_input: transformer(X_input, dX_rotate, q_rotate)

    # Test feature invariance to rotation and translation
    X_transformed = _rotate(X)
    node_h_r, edge_h_r, edge_idx_r, mask_i_r, mask_ij_r = feature_graph(
        X_transformed, C
    )
    assert not torch.allclose(X, X_transformed, atol=1e-3)
    assert torch.allclose(node_h_r, node_h, atol=1e-3)
    assert torch.allclose(edge_h_r, edge_h, atol=1e-3)
    assert torch.allclose(edge_idx, edge_idx_r, atol=1e-3)


def test_masked_interfaces():
    torch.manual_seed(10)
    dim_nodes, dim_edges = 128, 64
    num_neighbors = 30
    feature_graph = ProteinFeatureGraph(
        dim_nodes=dim_nodes,
        dim_edges=dim_edges,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=(
            ("distances_6mer", {"require_contiguous": True}),
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
            "orientations_chain",
        ),
        num_neighbors=num_neighbors,
        graph_kwargs={"mask_interfaces": True, "cutoff": None},
    )

    X, C, S = Protein(cif_path("5imm")).to_XCS()

    node_h, edge_h, edge_idx, mask_i, mask_ij = feature_graph(X, C)
    num_nodes = X.shape[1]

    # Test feature invariance to *single-chain* rotation and translation
    rigid_transformer = RigidTransformer(center_rotation=False)

    chain_mask = (C == 2).type(torch.float32)
    q = torch.Tensor([2.0, 1.0, 1.0, 0.5]).unsqueeze(0)
    dX = torch.Tensor([1.0, 2.0, -1.0]).unsqueeze(0)
    X_transformed = rigid_transformer(X, dX, q, mask=chain_mask)

    node_h_r, edge_h_r, edge_idx_r, mask_i_r, mask_ij_r = feature_graph(
        X_transformed, C
    )

    assert torch.allclose(node_h_r, node_h, atol=1e-3)
    assert torch.allclose(edge_h_r, edge_h, atol=1e-3)
    assert torch.allclose(edge_idx, edge_idx_r, atol=1e-3)


def test_centering_init_does_not_compute_remote_stats(monkeypatch, tmp_path):
    def _fail(*args, **kwargs):
        raise AssertionError("constructor should not compute reference stats")

    monkeypatch.setattr(ProteinFeatureGraph, "_reference_stats", _fail)

    feature_graph = ProteinFeatureGraph(
        dim_nodes=8,
        dim_edges=8,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=("distances_2mer",),
        num_neighbors=2,
        centered=True,
        centered_pdb="mock",
        centering_file=str(tmp_path / "missing.params"),
    )

    assert torch.count_nonzero(feature_graph.node_means_0) == 0
    assert torch.count_nonzero(feature_graph.edge_means_0) == 0


def test_centering_stats_can_be_loaded_explicitly(tmp_path):
    X = torch.randn(1, 6, 4, 3)
    C = torch.ones(1, 6, dtype=torch.long)

    source_graph = ProteinFeatureGraph(
        dim_nodes=8,
        dim_edges=8,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=("distances_2mer",),
        num_neighbors=2,
        centered=True,
        centered_pdb="mock",
        centering_file=str(tmp_path / "centering.params"),
    )
    stats = source_graph._feature_stats(X, C)
    source_graph._write_centering_params(
        str(tmp_path / "centering.params"), "mock", stats
    )

    loaded_graph = ProteinFeatureGraph(
        dim_nodes=8,
        dim_edges=8,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=("distances_2mer",),
        num_neighbors=2,
        centered=True,
        centered_pdb="mock",
        centering_file=str(tmp_path / "centering.params"),
    )

    assert torch.allclose(
        loaded_graph.node_means_0,
        torch.tensor(stats[json.dumps(loaded_graph.node_features[0])]).view(1, 1, -1),
    )
    assert torch.allclose(
        loaded_graph.edge_means_0,
        torch.tensor(stats[json.dumps(loaded_graph.edge_features[0])]).view(1, 1, -1),
    )


def test_packaged_centering_stats_are_available_by_default():
    feature_graph = ProteinFeatureGraph(
        dim_nodes=8,
        dim_edges=8,
        node_features=(("internal_coords", {"log_lengths": True}),),
        edge_features=("distances_2mer", "orientations_2mer", "distances_chain"),
        num_neighbors=2,
        centered=True,
        centered_pdb="2g3n",
    )

    assert torch.count_nonzero(feature_graph.node_means_0) > 0
    assert torch.count_nonzero(feature_graph.edge_means_0) > 0
