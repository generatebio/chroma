import pytest
import torch

from chroma.layers.structure import backbone, symmetry


class Test_symmetry:
    @pytest.mark.parametrize("group", ["C_2", "C_4", "D_2", "D_4", "T", "O", "I"])
    def test_point_groups(self, group):
        G = symmetry.get_point_group(group)

        # test if the determinants are ones for all rotation matrices
        assert torch.allclose(torch.det(G), torch.ones(G.shape[0]))

        # iterate the group multiplication table and check closure under multiplication
        for g1 in G:
            for g2 in G:
                assert (g1 @ g2) in G

        # check identity exists
        assert torch.eye(3) in G

        # check inverse is also in G
        for g in G:
            assert g.inverse() in G
