from unittest import TestCase

import pytest
import torch

from chroma.layers.structure.backbone import (
    BackboneBuilder,
    LossBackboneResidueDistance,
    ProteinBackbone,
    RigidTransform,
    RigidTransformer,
)


class TestProteinBackbone(TestCase):
    def test_cuda(self):
        if torch.cuda.is_available():
            try:
                protein_backbone = ProteinBackbone(1).cuda()
            except Exception:
                protein_backbone = None

            self.assertTrue(protein_backbone is not None)

    def test_sample(self):
        protein_backbone = ProteinBackbone(1)

        expected = torch.Tensor(
            [
                [
                    [
                        [0.1331, -1.6303, -0.7377],
                        [0.0414, -0.1759, -0.8080],
                        [-0.3710, 0.4114, 0.5376],
                        [0.1965, 1.3947, 1.0081],
                    ]
                ]
            ]
        )

        predicted = protein_backbone()
        self.assertEqual((1, 1, 4, 3), predicted.shape)
        self.assertTrue(torch.allclose(expected, predicted, rtol=1e-03))

    def test_random_init_backbone(self):
        protein_backbone = ProteinBackbone(1, init_state="")
        predicted = protein_backbone()
        self.assertEqual((1, 1, 4, 3), predicted.shape)

    def test_sample_cartesian(self):

        protein_backbone = ProteinBackbone(1, use_internal_coords=False)

        expected = torch.Tensor(
            [
                [
                    [
                        [0.1331, -1.6303, -0.7377],
                        [0.0414, -0.1759, -0.8080],
                        [-0.3710, 0.4114, 0.5376],
                        [0.1965, 1.3947, 1.0081],
                    ]
                ]
            ]
        )

        predicted = protein_backbone()
        self.assertEqual((1, 1, 4, 3), predicted.shape)
        self.assertTrue(torch.allclose(expected, predicted, rtol=1e-03))

    def test_initialized_sample(self):

        torch.manual_seed(7)
        input_x = torch.rand(1, 2, 4, 3)
        predicted = ProteinBackbone(1, use_internal_coords=False, X_init=input_x)()

        expected = torch.Tensor(
            [
                [
                    [
                        [-5.3644e-07, -2.6469e-01, 1.4716e-01],
                        [1.2197e-01, -2.3073e-01, -8.6988e-02],
                        [-3.2784e-01, 1.6625e-01, -1.4673e-01],
                        [3.1635e-01, 3.9145e-01, 3.8886e-02],
                    ],
                    [
                        [-2.4808e-01, -2.5717e-01, -6.6959e-02],
                        [-1.7564e-01, 2.5689e-01, -4.3900e-01],
                        [4.3500e-01, -3.5570e-01, 3.7083e-01],
                        [-1.2175e-01, 2.9370e-01, 1.8280e-01],
                    ],
                ]
            ]
        )

        self.assertTrue(torch.allclose(predicted, expected, rtol=1e-03))


class TestRigidTransform(TestCase):
    def test_sample(self):
        # Default behavior should be identity transformation
        rigid_transform = RigidTransform()
        torch.manual_seed(7)
        input_x = torch.rand(1, 1, 4, 3)
        predicted = rigid_transform(input_x)
        self.assertTrue(torch.allclose(predicted, input_x, rtol=1e-3))


class TestRigidTransformer(TestCase):
    def test_sample(self):
        rigid_transformer = RigidTransformer(center_rotation=True, keep_centered=True)

        input_x = torch.rand(1, 1, 4, 3)
        mean_centered = input_x - torch.mean(input_x.reshape(1, -1, 3), axis=-2)
        # Test Identity
        no_translation = torch.zeros(1, 3)
        identity_q = torch.Tensor([[1.0, 0, 0, 0]])

        predicted = rigid_transformer(input_x, no_translation, identity_q)
        self.assertTrue(torch.allclose(predicted, mean_centered, rtol=1e-3))

        # Test Translation
        x_translation = torch.Tensor([[1, 0, 0]])
        expected = mean_centered + x_translation

        predicted = rigid_transformer(input_x, x_translation, identity_q)
        self.assertTrue(torch.allclose(predicted, expected, rtol=1e-3))


class TestBackboneBuilder(TestCase):
    def test_sample(self):
        phi_tensor = torch.Tensor([[-1.0472]])
        psi_tensor = torch.Tensor([[-0.7854]])
        backbone_builder = BackboneBuilder()

        expected = torch.Tensor(
            [
                [
                    [
                        [-1.2286, 0.2223, -1.2286],
                        [-1.3203, 1.6767, -1.2989],
                        [-1.7327, 2.2640, 0.0468],
                        [-1.1652, 3.2473, 0.5172],
                    ]
                ]
            ]
        )

        predicted = backbone_builder(phi_tensor, psi_tensor)

        self.assertTrue(torch.allclose(expected, predicted, rtol=1e-3))

    def test_custom_sample(self):
        num_residues = 1
        phi_tensor = torch.Tensor([[-1.0472]])
        psi_tensor = torch.Tensor([[-0.7854]])
        backbone_builder = BackboneBuilder()

        expected = torch.Tensor(
            [
                [
                    [
                        [-1.2286, 0.2223, -1.2286],
                        [-1.3203, 1.6767, -1.2989],
                        [-1.7327, 2.2640, 0.0468],
                        [-1.1652, 3.2473, 0.5172],
                    ]
                ]
            ]
        )

        predicted = backbone_builder(phi_tensor, psi_tensor)

        lengths = torch.tensor(
            [[backbone_builder.lengths[key] for key in ["C_N", "N_CA", "CA_C"]]],
            dtype=torch.float32,
        )
        lengths = lengths.repeat(1, 1)  # (1,3)

        angles = torch.tensor(
            [[backbone_builder.angles[key] for key in ["CA_C_N", "C_N_CA", "N_CA_C"]]],
            dtype=torch.float32,
        )
        angles = angles.repeat(1, 1)  # (1,3)

        omega = backbone_builder.angles["omega"] * torch.ones(1, 1)  # (1,1)

        predicted = backbone_builder(phi_tensor, psi_tensor, omega, angles, lengths)
        self.assertTrue(torch.allclose(expected, predicted, rtol=1e-3))

        lengths = torch.tensor(
            [[backbone_builder.lengths[key] for key in ["C_N", "N_CA", "CA_C"]]],
            dtype=torch.float32,
        )
        lengths = lengths.repeat(1, num_residues)  # (1,3)

        angles = torch.tensor(
            [[backbone_builder.angles[key] for key in ["CA_C_N", "C_N_CA", "N_CA_C"]]],
            dtype=torch.float32,
        )
        angles = angles.repeat(1, num_residues)  # (1,3)

        omega = backbone_builder.angles["omega"] * torch.ones(1, num_residues)  # (1,1)

        predicted = backbone_builder(phi_tensor, psi_tensor, omega, angles, lengths)
        self.assertTrue(torch.allclose(expected, predicted, rtol=1e-3))
