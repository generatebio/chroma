import numpy as np
import pytest
import torch

from chroma.data import Protein
from chroma.layers.structure.backbone import RigidTransformer
from chroma.layers.structure.rmsd import (
    BackboneRMSD,
    CrossRMSD,
    LossFragmentPairRMSD,
    LossFragmentRMSD,
    LossNeighborhoodRMSD,
)


@pytest.fixture
def backbones():
    bb1 = torch.tensor(
        [
            -5.68175,
            -2.183,
            3.27979,
            -4.82875,
            -3.256,
            2.79379,
            -3.34475,
            -2.899,
            2.79579,
            -2.51375,
            -3.697,
            3.21979,
            -3.01675,
            -1.713,
            2.29979,
            -1.62875,
            -1.289,
            2.22979,
            -0.95775,
            -1.094,
            3.58379,
            0.20325,
            -1.46,
            3.75679,
            -1.69375,
            -0.547,
            4.54479,
            -1.16675,
            -0.358,
            5.88579,
            -0.94175,
            -1.732,
            6.50679,
            0.03125,
            -1.943,
            7.23679,
        ]
    ).reshape(-1, 12, 3)

    bb2 = torch.tensor(
        [
            3.91725,
            1.271,
            -1.22921,
            3.22825,
            0.099,
            -1.74321,
            2.09025,
            0.535,
            -2.66521,
            1.91025,
            -0.018,
            -3.74821,
            1.34825,
            1.553,
            -2.23921,
            0.24325,
            2.085,
            -3.02321,
            0.76425,
            2.518,
            -4.38621,
            0.10225,
            2.315,
            -5.41221,
            1.96925,
            3.085,
            -4.38121,
            2.61525,
            3.562,
            -5.59821,
            3.27725,
            2.453,
            -6.40421,
            4.07425,
            2.713,
            -7.30321,
        ]
    ).reshape(-1, 12, 3)
    return bb1, bb2


def test_pairedRMSD(backbones):
    bb1, bb2 = backbones
    cross_rmsd = CrossRMSD()
    predicted_rmsd = cross_rmsd.pairedRMSD(bb1, bb2)
    assert torch.isclose(predicted_rmsd, torch.tensor(0.3542), rtol=1e-3)


def test_pairedRMSD_symeig(backbones):
    bb1, bb2 = backbones
    cross_rmsd = CrossRMSD(method="symeig")
    predicted_rmsd = cross_rmsd.pairedRMSD(bb1, bb2)
    assert torch.isclose(predicted_rmsd, torch.tensor(0.35), rtol=1e1)


def test_sample(backbones):
    bb1, bb2 = backbones

    cross_rmsd = CrossRMSD()
    input_x = torch.cat([bb1, bb1])
    predicted = cross_rmsd(input_x, input_x)

    assert predicted.shape == (input_x.shape[0], input_x.shape[0])
    assert torch.allclose(predicted, torch.zeros_like(predicted), atol=1e-1)

    predicted = cross_rmsd(bb1, bb2)
    assert all(torch.isclose(predicted, torch.tensor(0.35), rtol=1e-1))


def test_sample_symeigh(backbones):
    bb1, bb2 = backbones
    cross_rmsd = CrossRMSD(method="symeig")
    input_x = torch.cat([bb1, bb1])

    predicted = cross_rmsd(input_x, input_x)
    assert torch.allclose(predicted, torch.zeros_like(predicted), atol=1e-2)


def test_backbone_rmsd(backbones):
    bb1, bb2 = backbones
    for method in ["symeig", "power"]:
        backbone_rmsd = BackboneRMSD(method=method)

        X, C, S = Protein("5imm").to_XCS()

        rigid_transformer = RigidTransformer()
        dX = torch.Tensor([[1, 4, 2]])
        q = torch.Tensor([[0.5, 1, 0, 1]])
        X_transform = rigid_transformer(X, dX, q)
        X_transform_aligned, rmsd = backbone_rmsd.align(
            X_transform, X, C, align_unmasked=True
        )

        assert not torch.allclose(X, X_transform, atol=1e-2)
        assert torch.allclose(X, X_transform_aligned, atol=1e-2)
        assert rmsd < 1e-2


def test_fragment_rmsd(debug=False):
    X, C, S = Protein("1SHG").to_XCS()

    loss_frags = LossFragmentRMSD()

    X_noise = X + torch.randn_like(X)
    rmsd = loss_frags(X, X, C)
    rmsd_noised = loss_frags(X_noise, X, C)
    assert rmsd.mean() < 1e-2
    assert rmsd_noised.mean() > 1.0

    if debug:
        from chroma.layers.structure import diffusion

        noise = diffusion.DiffusionChainCov(complex_scaling=True)
        X_noise = noise(X, C, t=0.6)

        rmsd, X_frag_target, X_frag_mobile, X_frag_mobile_align = loss_frags(
            X_noise, X, C, return_coords=True
        )
        print(rmsd)

        def _trajectory(X_frags):
            B, I, _, _ = list(X_frags.shape)
            X_frags = X_frags.reshape([B * I, -1, 4, 3])
            X_trajectory = [X_t[None, ...] for X_t in X_frags.unbind(0)]
            return X_trajectory

        C = torch.ones([1, loss_frags.k])
        X_trajectory_1 = _trajectory(X_frag_target)
        X_trajectory_2 = _trajectory(X_frag_mobile_align)
        X_trajectory_3 = _trajectory(X_frag_mobile)
        # Fight pymol confusion
        index = 10
        X_trajectory_3 = [X_trajectory_1[index]] + X_trajectory_3
        X_trajectory_2 = [X_trajectory_1[index]] + X_trajectory_2
        X_trajectory_1 = [X_trajectory_1[index]] + X_trajectory_1
        Protein.from_XCS_trajectory(X_trajectory_1, C, 0.0 * C).to_CIF(
            "X_frag_target.cif"
        )
        Protein.from_XCS_trajectory(X_trajectory_2, C, 0.0 * C).to_CIF(
            "X_frag_noise_aligned.cif"
        )
        Protein.from_XCS_trajectory(X_trajectory_3, C, 0.0 * C).to_CIF(
            "X_frag_noise.cif"
        )
    return


def test_fragment_pair_rmsd(debug=False):
    X, C, S = Protein("1SHG").to_XCS()

    loss_pairs = LossFragmentPairRMSD()

    X_noise = X + torch.randn_like(X)
    rmsd, mask_ij = loss_pairs(X, X, C)
    rmsd_noised, mask_ij = loss_pairs(X_noise, X, C)
    assert rmsd.mean() < 1e-2
    assert rmsd_noised.mean() > 1.0

    if debug:
        from chroma.layers.structure import diffusion

        noise = diffusion.DiffusionChainCov(complex_scaling=True)
        X_noise = noise(X, C, t=0.6)

        rmsd, mask_ij, X_pair_target, X_pair_mobile, X_pair_mobile_align = loss_pairs(
            X_noise, X, C, return_coords=True
        )
        print(rmsd)

        def _trajectory(X_pairs):
            B, I, J, _, _ = list(X_pairs.shape)
            X_pairs = X_pairs.reshape([B * I * J, -1, 4, 3])
            X_trajectory = [X_t[None, ...] for X_t in X_pairs.unbind(0)]
            return X_trajectory

        C = torch.cat(
            [torch.ones([1, loss_pairs.k]), 2 * torch.ones([1, loss_pairs.k])], -1
        )
        X_trajectory_1 = _trajectory(X_pair_target)
        X_trajectory_2 = _trajectory(X_pair_mobile_align)
        X_trajectory_3 = _trajectory(X_pair_mobile)
        # Fight pymol confusion
        index = 1579
        X_trajectory_3 = [X_trajectory_1[index]] + X_trajectory_3
        X_trajectory_2 = [X_trajectory_1[index]] + X_trajectory_2
        X_trajectory_1 = [X_trajectory_1[index]] + X_trajectory_1
        Protein.from_XCS_trajectory(X_trajectory_1, C, 0.0 * C).to_CIF(
            "X_pair_target.cif"
        )
        Protein.from_XCS_trajectory(X_trajectory_2, C, 0.0 * C).to_CIF(
            "X_pair_noise_aligned.cif"
        )
        Protein.from_XCS_trajectory(X_trajectory_3, C, 0.0 * C).to_CIF(
            "X_pair_noise.cif"
        )
    return


def test_neighborhood_rmsd(debug=False):
    X, C, S = Protein("1SHG").to_XCS()

    loss_nb = LossNeighborhoodRMSD()

    X_noise = X + torch.randn_like(X)
    rmsd, mask = loss_nb(X, X, C)
    rmsd_noised, mask = loss_nb(X_noise, X, C)
    assert rmsd.mean() < 1e-2
    assert rmsd_noised.mean() > 1.0

    if debug:
        from chroma.layers.structure import diffusion

        noise = diffusion.DiffusionChainCov(complex_scaling=True)
        X_noise = noise(X, C, t=0.7)

        rmsd, mask, X_nb_target, X_nb_mobile, X_nb_mobile_align = loss_nb(
            X_noise, X, C, return_coords=True
        )
        print(rmsd, X_nb_target.shape)

        def _trajectory(X_nbs):
            B, I, _, _ = list(X_nbs.shape)
            X_nbs = X_nbs.reshape([B * I, -1, 4, 3])
            X_trajectory = [X_t[None, ...] for X_t in X_nbs.unbind(0)]
            return X_trajectory

        C = torch.ones([1, X_nb_target.shape[2] // 4])
        X_trajectory_1 = _trajectory(X_nb_target)
        X_trajectory_2 = _trajectory(X_nb_mobile_align)
        X_trajectory_3 = _trajectory(X_nb_mobile)
        # Fight pymol confusion
        index = 10
        X_trajectory_3 = [X_trajectory_1[index]] + X_trajectory_3
        X_trajectory_2 = [X_trajectory_1[index]] + X_trajectory_2
        X_trajectory_1 = [X_trajectory_1[index]] + X_trajectory_1
        Protein.from_XCS_trajectory(X_trajectory_1, C, 0.0 * C).to_CIF(
            "X_nb_target.cif"
        )
        Protein.from_XCS_trajectory(X_trajectory_2, C, 0.0 * C).to_CIF(
            "X_nb_noise_aligned.cif"
        )
        Protein.from_XCS_trajectory(X_trajectory_3, C, 0.0 * C).to_CIF("X_nb_noise.cif")
    return
