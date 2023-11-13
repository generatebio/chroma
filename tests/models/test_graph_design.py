import pytest
import torch

from chroma.models.graph_design import GraphDesign, ProteinTraversalSpatial


@pytest.fixture
def model():
    model = GraphDesign(predict_S_marginals=True, predict_S_potts=True)
    model.eval()
    return model


def test_sequential_decoding(model, XCS):
    """Test that the sequential and parallelized decoding of GNN agree."""

    from chroma.data import xcs

    X, C, S = XCS
    permute_idx = torch.argsort(torch.randn_like(C.float()), dim=-1)

    # Fix a permutation
    scores_parallel = model(X, C, S, permute_idx=permute_idx)

    _, _, _, scores_sequential = model.sample(
        X, C, S, permute_idx=permute_idx, clamped=True, return_scores=True
    )

    assert torch.allclose(
        scores_parallel["logp_S"], scores_sequential["logp_S"], atol=1e-3
    )
    assert torch.allclose(
        scores_parallel["logp_chi"], scores_sequential["logp_chi"], atol=1e-3
    )

    # =============Fix a permutation ========
    X_sample, S_sample, _, scores_sequential = model.sample(
        X, C, S, permute_idx=permute_idx, clamped=False, return_scores=True
    )
    scores_parallel = model(X_sample, C, S_sample, permute_idx=permute_idx)

    assert torch.allclose(
        scores_parallel["logp_S"], scores_sequential["logp_S"], atol=1e-3
    )
    assert torch.allclose(
        scores_parallel["logp_chi"], scores_sequential["logp_chi"], atol=1e-3
    )
    return


def test_deterministic_traversal(XCS):
    """Check deterministic flag on ProteinTraversalSpatial module."""
    traversal = ProteinTraversalSpatial(deterministic=True)
    X, C, _ = XCS
    permute_idx = traversal(X, C)
    permute_idx_2 = traversal(X, C)
    assert torch.allclose(permute_idx, permute_idx_2)
    return


def test_graph_design_outputs(model, XCS):
    """Smoke test all GraphDesign outputs."""
    X, C, S = XCS
    outputs = model(X, C, S)
    for key in ["logp_S", "logp_S_marginals", "logp_S_potts"]:
        assert outputs[key].shape == X.shape[:2]
    assert torch.allclose(outputs["X_noise"], X)
    for key in ["chi", "logp_chi"]:
        assert outputs[key].shape[:-1] == X.shape[:2] and outputs[key].shape[-1] == 4
