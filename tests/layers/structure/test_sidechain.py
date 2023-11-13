from unittest import TestCase

import numpy as np
import pytest
import torch

from chroma import constants
from chroma.data import Protein
from chroma.layers.structure import backbone, sidechain


class TestSideChain(TestCase):
    def setUp(self):
        self.builder = sidechain.SideChainBuilder()
        self.chi_angles = sidechain.ChiAngles()
        self.rmsd_loss = sidechain.LossSideChainRMSD()
        self.clash_loss = sidechain.LossSidechainClashes()
        self.frame_loss = sidechain.LossFrameAlignedGraph(distance_eps=1e-9)
        self.distance_loss = sidechain.LossAllAtomDistances()
        self.frame_builder = sidechain.AllAtomFrameBuilder()

        pdb_id = "1SHG"
        self.X, self.C, self.S = Protein(pdb_id).to_XCS(all_atom=True)

    def test_chi_cartesian_round_trip(self):
        X, C, S = self.X, self.C, self.S

        X_bb = X[:, :, :4, :]
        chi, mask_chi = self.chi_angles(X, C, S)
        X_reference, mask_X = self.builder(X_bb, C, S, chi)

        # Test round trip processing
        chi_direct, _ = self.chi_angles(X_reference, C, S)
        X_cycle, _ = self.builder(X_bb, C, S, chi_direct)
        chi_cycle, _ = self.chi_angles(X_cycle, C, S)

        _embed = lambda a: torch.stack([torch.cos(a), torch.sin(a)], -1)

        self.assertTrue(torch.allclose(X_reference, X_cycle, atol=1e-1))
        self.assertTrue(torch.allclose(_embed(chi), _embed(chi_cycle), atol=1e-2))

        loss = self.rmsd_loss(X, X_cycle, C, S)
        loss = self.clash_loss(X_cycle, C, S)

    def test_integration(self):
        num_letters = 20
        chi = np.pi * torch.rand([1, num_letters, 4])

        X_bb = backbone.ProteinBackbone(num_letters, init_state="beta")()
        S = torch.arange(num_letters).unsqueeze(0)
        C = torch.ones_like(S)
        X, mask_X = self.builder(X_bb, C, S, chi)
        chi, mask_chi = self.chi_angles(X, C, S)

        self.assertTrue(
            np.allclose(
                mask_X.sum([-1, -2]).data.numpy(), np.asarray(constants.AA20_NUM_ATOMS)
            )
        )
        self.assertTrue(
            np.allclose(
                mask_chi.sum(-1).data.numpy(), np.asarray(constants.AA20_NUM_CHI)
            )
        )

    def test_frame_builder_round_trip(self):
        X, C, S = self.X, self.C, self.S

        x, q, chi = self.frame_builder.inverse(X, C, S)
        X_cycle, mask_atoms = self.frame_builder(x, q, chi, C, S)

        x = x + torch.randn_like(x) * 10.0
        q = q + torch.randn_like(q) * 2.0
        X_perturb, mask_atoms = self.frame_builder(x, q, chi, C, S)

        mask = (C > 0).float()

        _loss = lambda loss: (loss * mask).sum() / mask.sum()
        loss_cycle_avg = _loss(self.frame_loss(X, X_cycle, C, S))
        loss_perturb_avg = _loss(self.frame_loss(X, X_perturb, C, S))
        print(loss_cycle_avg, loss_perturb_avg)
        self.assertTrue(loss_cycle_avg.item() < 1.0)
        self.assertTrue(loss_perturb_avg.item() > 1.0)

        loss_cycle_avg = _loss(self.distance_loss(X, X_cycle, C, S))
        loss_perturb_avg = _loss(self.distance_loss(X, X_perturb, C, S))
        print(loss_cycle_avg, loss_perturb_avg)
        self.assertTrue(loss_cycle_avg.item() < 1.0)
        self.assertTrue(loss_perturb_avg.item() > 1.0)
