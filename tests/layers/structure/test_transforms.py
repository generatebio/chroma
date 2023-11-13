import pytest
import torch

from chroma.layers.structure.geometry import rotations_from_quaternions
from chroma.layers.structure.transforms import (
    average_transforms,
    collect_neighbor_transforms,
    compose_inner_transforms,
    compose_transforms,
    compose_translation,
    equilibrate_transforms,
    fuse_gaussians_isometric_plus_radial,
)


@pytest.fixture
def vec():
    torch.manual_seed(0)
    return torch.rand(3)


@pytest.fixture
def rotations():
    torch.manual_seed(0)
    q = torch.rand(2, 4)
    return rotations_from_quaternions(q, normalize=True).unbind()


@pytest.fixture
def translations():
    torch.manual_seed(0)
    return torch.rand(2, 3).unbind()


def test_compose_transforms(vec, rotations, translations):
    R_a, R_b = rotations
    t_a, t_b = translations
    inter = R_b @ vec + t_b
    result = R_a @ inter + t_a
    R_composed, t_composed = compose_transforms(R_a, t_a, R_b, t_b)
    assert torch.allclose(result, R_composed @ vec + t_composed)


def test_compose_translation(vec, rotations, translations):
    R_a, _ = rotations
    t_a, t_b = translations
    inter = vec + t_b
    result = R_a @ inter + t_a
    t_composed = compose_translation(R_a, t_a, t_b)
    assert torch.allclose(result, R_a @ vec + t_composed)


def test_compose_inner_transforms(vec, rotations, translations):
    R_a, R_b = rotations
    t_a, t_b = translations
    R_a_inv = torch.inverse(R_a)
    inter = R_b @ vec + t_b
    result = R_a_inv @ (inter - t_a)
    R_composed, t_composed = compose_inner_transforms(R_a, t_a, R_b, t_b)
    # bump up tolerance because of matrix inversion
    assert torch.allclose(result, R_composed @ vec + t_composed, atol=1e-3, rtol=1e-2)


def test_fuse_gaussians_isometric_plus_radial(vec):
    p_iso = torch.tensor([0.3, 0.7])
    p_rad = torch.zeros_like(p_iso)
    x = torch.stack([vec, 2 * vec])
    direction = torch.zeros_like(x)
    x_fused, P_fused = fuse_gaussians_isometric_plus_radial(
        x, p_iso, p_rad, direction, 0
    )
    assert torch.allclose((p_iso[0] + 2 * p_iso[1]) * vec, P_fused @ x_fused)


def test_collect_neighbor_transforms(rotations, translations):
    R_i = torch.stack(rotations).unsqueeze(0)
    t_i = torch.stack(translations).unsqueeze(0)
    edge_idx = torch.LongTensor([[1], [0]]).unsqueeze(0)
    R_j, t_j = collect_neighbor_transforms(R_i, t_i, edge_idx)
    assert torch.allclose(R_j, torch.flip(R_i, [1]).unsqueeze(2))
    assert torch.allclose(t_j, torch.flip(t_i, [1]).unsqueeze(2))


def test_equilibrate_transforms(rotations, translations):
    R_i = torch.stack(rotations).unsqueeze(0)
    t_i = torch.stack(translations).unsqueeze(0)
    R_ji = torch.eye(3).expand(1, 2, 1, 3, 3)
    t_ji = torch.zeros(1, 2, 1, 3)
    logit_ij = torch.ones(1, 2, 1, 1)
    mask_ij = torch.ones(1, 2, 1)
    edge_idx = torch.LongTensor([[1], [0]]).unsqueeze(0)
    # two transforms on nodes that are each other's neighbors, so a single
    # iteration will just swap the transforms
    R_eq, t_eq = equilibrate_transforms(
        R_i, t_i, R_ji, t_ji, logit_ij, mask_ij, edge_idx, iterations=1
    )
    assert torch.allclose(R_eq, torch.flip(R_i, [1]), atol=1e-3, rtol=1e-2)
    assert torch.allclose(t_eq, torch.flip(t_i, [1]), atol=1e-3, rtol=1e-2)
    # two iterations moves the transforms back to themselves
    R_eq, t_eq = equilibrate_transforms(
        R_i, t_i, R_ji, t_ji, logit_ij, mask_ij, edge_idx, iterations=2
    )
    assert torch.allclose(R_eq, R_i, atol=1e-3, rtol=1e-2)
    assert torch.allclose(t_eq, t_i, atol=1e-3, rtol=1e-2)


def test_average_transforms(rotations, translations):
    R = torch.stack([rotations[0], torch.eye(3)])
    t = torch.stack([translations[0], torch.zeros(3)])
    w = torch.ones(2, 2)
    mask = torch.ones(2)
    # average of a transform with the identity is "half" the transform
    R_avg, t_avg = average_transforms(R, t, w, mask, dim=0, dither=False)
    R_total_fromavg, _ = compose_transforms(
        R_avg, torch.zeros(3), R_avg, torch.zeros(3)
    )
    _, t_total_fromavg = compose_transforms(torch.eye(3), t_avg, torch.eye(3), t_avg)
    assert torch.allclose(R_total_fromavg, R[0], atol=1e-3, rtol=1e-2)
    assert torch.allclose(t_total_fromavg, t[0], atol=1e-3, rtol=1e-2)
