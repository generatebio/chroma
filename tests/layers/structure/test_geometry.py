from pathlib import Path
from unittest import TestCase

import numpy as np
import torch
import torch.nn.functional as F

import chroma
from chroma.data import Protein
from chroma.layers.structure import geometry


class TestDistances(TestCase):
    def test_sample(self):
        distances = geometry.Distances()
        torch.manual_seed(7)
        input_x = torch.rand(1, 2, 4, 3)
        dim = -2
        predicted = distances(input_x, None, dim)
        self.assertTrue(predicted.shape == (1, 2, 4, 4))
        expected = torch.tensor(
            [
                [
                    [
                        [0.0316, 0.2681, 0.6169, 0.7371],
                        [0.2681, 0.0316, 0.6037, 0.6646],
                        [0.6169, 0.6037, 0.0316, 0.7079],
                        [0.7371, 0.6646, 0.7079, 0.0316],
                    ],
                    [
                        [0.0316, 0.6395, 0.8179, 0.6187],
                        [0.6395, 0.0316, 1.1853, 0.6260],
                        [0.8179, 1.1853, 0.0316, 0.8764],
                        [0.6187, 0.6260, 0.8764, 0.0316],
                    ],
                ]
            ]
        )
        self.assertTrue(torch.allclose(predicted, expected, rtol=1e-3))


class TestRotations(TestCase):
    def setUp(self):
        self.R = torch.tensor(
            [
                [
                    [0.9027011, -0.1829866, -0.3894183],
                    [-0.3146039, 0.3367128, -0.8874959],
                    [0.2935220, 0.9236560, 0.2463827],
                ],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [
                    [-0.6638935, 0.6988353, 0.2662229],
                    [-0.6322795, -0.3344426, -0.6988353],
                    [-0.3993345, -0.6322795, 0.6638935],
                ],
            ]
        )

        self.q = torch.tensor(
            [
                [0.7883205, 0.5743704, -0.2165808, -0.0417398],
                [1.0, 0.0, 0.0, 0.0],
                [0.4079085, 0.0407909, 0.4079085, -0.815817],
            ]
        )

    def test_rotations_from_quaternions(self):
        R_from_q = geometry.rotations_from_quaternions(self.q)
        self.assertTrue(torch.allclose(self.R, R_from_q, atol=1e-3))

    def test_quaternions_from_rotations(self):
        q_from_R = geometry.quaternions_from_rotations(self.R, eps=0.0)
        self.assertTrue(torch.allclose(self.q, q_from_R, atol=1e-3))

    def test_round_trip(self):
        R_from_q = geometry.rotations_from_quaternions(self.q)
        q_round_trip = geometry.quaternions_from_rotations(R_from_q, eps=0.0)
        R_from_round_trip = geometry.rotations_from_quaternions(q_round_trip)

        self.assertTrue(torch.allclose(self.q, q_round_trip, atol=1e-3))
        self.assertTrue(torch.allclose(self.R, R_from_round_trip, atol=1e-3))


class TestExtendAtoms(TestCase):
    def test_extend_atoms_round_trip(self):
        # Test cycle-consistency of geometry measurement and building routines
        num_batch, num_residues = 10, 30
        X1, X2, X3 = torch.randn([num_batch, num_residues, 3, 3]).unbind(-1)
        L = torch.exp(torch.randn([num_batch, num_residues])) + 1.0
        A = np.pi * torch.sigmoid(torch.randn([num_batch, num_residues]))
        D = np.pi * torch.randn([num_batch, num_residues])

        X4 = geometry.extend_atoms(X1, X2, X3, L, A, D, distance_eps=1e-6)

        L_recover = geometry.lengths(X3, X4, distance_eps=0.0)
        A_recover = geometry.angles(X2, X3, X4, distance_eps=0.0)
        D_recover = geometry.dihedrals(X1, X2, X3, X4, distance_eps=0.0)

        _embed = lambda a: torch.stack([torch.cos(a), torch.sin(a)], -1)

        self.assertTrue(torch.allclose(L, L_recover, atol=1e-2))
        self.assertTrue(torch.allclose(A, A_recover, atol=1e-2))
        self.assertTrue(torch.allclose(_embed(D), _embed(D_recover), atol=1e-2))
        return


class TestVirtualAtomsCA(TestCase):
    def test_atom_placement(self):
        # Load test case
        file_cif = str(
            Path(Path(chroma.__file__).parent.parent, "tests", "resources", "5jg9.cif",)
        )
        X, C, S = Protein(file_cif).to_XCS()

        for v_type in ["cbeta", "dicons"]:
            # Place atoms
            atom_placer = geometry.VirtualAtomsCA(virtual_type=v_type)
            X_virtual = atom_placer(X, C)
            # DEBUG: Sanity check is useful for testing
            # geometry.debug_pymol_virtual_atoms(X, X_virtual, 'test_5jg9.pml')

            # Test that generated angles are correct
            X_N, X_CA, X_C, X_O = X.unbind(2)

            bonds = torch.norm(X_virtual - X_CA, dim=-1)
            angles = geometry.angles(
                X_N, X_CA, X_virtual, distance_eps=1e-6, degrees=True
            )
            dihedrals = geometry.dihedrals(
                X_C, X_N, X_CA, X_virtual, distance_eps=1e-6, degrees=True
            )

            bond_t, angle_t, dihedral_t = atom_placer.geometry()
            mask = (C > 0).type(torch.float32)
            bond_error = mask * (bonds - bond_t)
            angle_error = mask * (angles - angle_t)
            dihedral_error = mask * (dihedrals - dihedral_t)

            self.assertTrue(
                torch.allclose(bond_error, torch.zeros_like(bond_error), atol=1e-2)
            )
            self.assertTrue(
                torch.allclose(angle_error, torch.zeros_like(angle_error), atol=1e-2)
            )
            self.assertTrue(
                torch.allclose(
                    dihedral_error, torch.zeros_like(dihedral_error), atol=1e-2
                )
            )
