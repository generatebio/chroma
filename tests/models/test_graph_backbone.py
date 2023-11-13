import pytest
import torch

from chroma.data import Protein
from chroma.layers.structure import backbone, protein_graph
from chroma.models.graph_backbone import GraphBackbone


def test_denoiser(dim_nodes=32, dim_edges=32):
    X, C, S = Protein("1SHG").to_XCS()
    model = GraphBackbone(dim_nodes=dim_nodes, dim_edges=dim_edges)

    # check if denoiser is working as expected
    model.CA_dist_scaling = False
    X0 = model.denoise(X, C, 0.0)
    assert X0.shape == X.shape

    # test if prediction_type="scale" is working
    model = GraphBackbone(prediction_type="scale")
    X0 = model.denoise(X, C, 0.0)
    assert torch.allclose(X0, X, rtol=1e-2)

    # check if CA_dist scale is working as expected
    model.CA_dist_scaling = False
    X0 = model.denoise(0.25 * X, C, 1e-4)
    assert X0.shape == X.shape
    # assert model._D_backbone_CA(X0, C).min().item() < model.min_CA_bb_distance


@pytest.mark.parametrize("t", [0.1, 0.7, 1.0])
def test_equivariance_denoiser(t, dim_nodes=32, dim_edges=32, seed=10, debug=False):
    X = backbone.ProteinBackbone(num_batch=1, num_residues=20, init_state="alpha")()
    C = torch.ones(X.shape[:2])
    S = torch.zeros_like(C).long()

    model = GraphBackbone(dim_nodes=dim_nodes, dim_edges=dim_edges).eval()

    # Test rotation equivariance
    transformer = backbone.RigidTransformer(center_rotation=False)
    q_transform = torch.Tensor([0.0, 0.1, -1.2, 0.5]).unsqueeze(0)
    dX_transform = torch.Tensor([-3.0, 30.0, 7.0]).unsqueeze(0)
    _transform = lambda X_input: transformer(X_input, dX_transform, q_transform)

    # Add noise
    X_noised = model.noise_perturb(X, C, t=t)
    X_noised_transform = _transform(X_noised)

    # Synchronize random seeds for random graph generation
    torch.manual_seed(seed)
    X_denoised = model.denoise(X_noised, C, t=t)
    X_denoised_transform = _transform(X_denoised)
    torch.manual_seed(seed)
    X_transform_denoised = model.denoise(X_noised_transform, C, t=t)

    if debug:
        print((X_denoised_transform - X_transform_denoised).abs().max())

        Protein(X, C, S).to_CIF("X_denoised.cif")
        Protein(X_denoised, C, S).to_CIF("X_denoised.cif")
        Protein(X_denoised_transform, C, S).to_CIF("X_denoised_transform.cif")
        Protein(X_transform_denoised, C, S).to_CIF("X_transform_denoised.cif")

    # The oxygen atom of the final carboxy terminus residue in each chain
    # is disambiguated via zero-padding (non-equivariant), so it can be up to \
    # ~1 angstrom off depending on global pose
    assert torch.allclose(
        X_denoised_transform[:, :-1, :, :],
        X_transform_denoised[:, :-1, :, :],
        atol=1e-1,
    )
    # Nevertheless at this adjusted tolerance we are equivariant
    assert torch.allclose(X_denoised_transform, X_transform_denoised, atol=3.0)
    assert not torch.allclose(X_denoised, X_transform_denoised, atol=1e-1)


@pytest.mark.parametrize("num_transform_weights", [1, 2, 3])
@pytest.mark.parametrize("dim_nodes", [32])
@pytest.mark.parametrize("dim_edges", [32])
def test_equivariance_graph_update(
    num_transform_weights, dim_nodes, dim_edges, output_structures=False
):
    torch.manual_seed(10.0)

    # Initialize layers
    bb_update = backbone.GraphBackboneUpdate(
        dim_nodes=dim_nodes,
        dim_edges=dim_edges,
        method="neighbor_global_affine",
        num_transform_weights=num_transform_weights,
    ).eval()
    pg = protein_graph.ProteinFeatureGraph(
        dim_nodes=dim_nodes, dim_edges=dim_edges, num_neighbors=5
    )

    # Test rotation equivariance
    transformer = backbone.RigidTransformer(center_rotation=False)
    q_rotate = torch.Tensor([0.0, 0.1, -1.2, 0.5]).unsqueeze(0)
    dX_rotate = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0)
    _rotate = lambda X_input: transformer(X_input, dX_rotate, q_rotate)

    # Load test structure and canonicalize
    X, C, S = Protein("1qys").to_XCS()
    R, t, _ = bb_update.frame_builder.inverse(X, C)
    X = bb_update.frame_builder.forward(R, t, C)

    # Apply transformation
    node_h, edge_h, edge_idx, mask_i, mask_ij = pg(X, C)
    X_update, _, _, _ = bb_update(X, C, node_h, edge_h, edge_idx, mask_i, mask_ij)

    # Compute for rotated system
    X_rotate = _rotate(X)
    X_rotate_update, _, _, _ = bb_update(
        X_rotate, C, node_h, edge_h, edge_idx, mask_i, mask_ij
    )
    X_update_rotate = _rotate(X_update)

    assert torch.allclose(X_rotate_update, X_update_rotate, atol=1e-2)

    if output_structures:
        from chroma.layers.structure.rmsd import BackboneRMSD

        bb_rmsd = BackboneRMSD()
        X_aligned, rmsd = bb_rmsd.align(X_rotate_update, X_update_rotate, C)
        print(rmsd)
        Protein.from_XCS_trajectory(
            [X, X_update, X_rotate, X_rotate_update, X_update_rotate], C, S
        ).to_CIF("test_equi.cif")
    return
