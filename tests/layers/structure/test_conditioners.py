from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import chroma
from chroma.data import xcs
from chroma.data.protein import Protein
from chroma.layers.structure import backbone, conditioners, rmsd, symmetry
from chroma.models.graph_backbone import GraphBackbone
from chroma.models.procap import ProteinCaption


@pytest.fixture(scope="session")
def XCO():
    repo = Path(chroma.__file__).parent.parent
    test_cif = str(Path(repo, "tests", "resources", "6wgl.cif"))
    protein = Protein.from_CIF(test_cif)
    X, C, S = protein.to_XCS()
    X.requires_grad = True
    O = F.one_hot(S, 20)
    return X, C, O


@pytest.fixture(scope="session")
def protein():
    pdb_id = "1drf"
    protein = Protein.from_PDBID(pdb_id, canonicalize=True)
    protein.sys.save_selection(gti=list(range(15)), selname="clamp")
    protein.sys.save_selection(gti=list(range(15, 25)), selname="semirigid")
    return protein


@pytest.fixture
def test_conditioner_pointgroup_conditioner(XCO):
    X, C, O = XCO
    conditioner = conditioners.SymmetryConditioner(
        G=symmetry.get_point_group("I"), num_chain_neighbors=3
    )
    X_constrained, _, _, _, _ = conditioner(X, C, O, 0.0, 0.5)
    return conditioner, X_constrained, C


@pytest.fixture
def test_conditioner_screw_conditioner(XCO):
    X, C, O = XCO
    conditioner = conditioners.ScrewConditioner(theta=np.pi / 4, tz=5.0, M=10)
    X_constrained, _, _, _, _ = conditioner(X, C, O, 0.0, 0.5)
    return conditioner, X_constrained, C


@pytest.fixture
def test_conditioner_Rg_conditioner(XCO):
    X, C, O = XCO
    conditioner = conditioners.RgConditioner()
    conditioner(X, C, O, 0.0, 0.5)
    return conditioner, X, C


@pytest.fixture
def test_conditioner_symmetry_and_substructure(protein):
    bb_model = GraphBackbone(dim_nodes=16, dim_edges=16)
    protein.get_mask("namesel clamp")
    sub_conditioner = conditioners.SubstructureConditioner(
        protein, bb_model, "namesel clamp"
    )

    sym_conditioner = conditioners.SymmetryConditioner(
        G=symmetry.get_point_group("C_3"), num_chain_neighbors=1, freeze_com=True
    )

    composed_conditioner = conditioners.ComposedConditioner(
        [sub_conditioner, sym_conditioner]
    )

    X, C, S = protein.to_XCS()
    X.requires_grad = True
    O = F.one_hot(S, 20)
    X_constrained, _, _, _, _ = composed_conditioner(X, C, O, 0.0, torch.tensor([0.0]))
    return composed_conditioner, X_constrained, C


@pytest.fixture
def test_conditioner_substructure_conditioner(protein):
    aligner = rmsd.BackboneRMSD()
    bb_model = GraphBackbone(dim_nodes=16, dim_edges=16)
    X, C, S = protein.to_XCS()
    O = F.one_hot(S, 20)
    conditioner = conditioners.SubstructureConditioner(
        protein, bb_model, "namesel clamp"
    )
    X_conditioned, _, _, _, _ = conditioner(
        torch.randn_like(X), C, O, 0.0, torch.tensor([0.0])
    )
    D = protein.get_mask("namesel clamp")
    _, rmsd1 = aligner.align(X_conditioned, X, D)
    assert rmsd1.isclose(torch.tensor(0.0), atol=1e-1)

    return conditioner, X, C


@pytest.fixture
def test_conditioner_procap_conditioner(XCO):
    model = ProteinCaption()
    X, C, O = XCO
    conditioner = conditioners.ProCapConditioner("Test caption", -1, model=model)
    conditioner(X, C, O, 0, 0.5)
    return conditioner, X, C


def collect_conditioners():
    return [v for k, v in globals().items() if k.startswith("test_conditioner_")]


@pytest.fixture(params=["globular"])
def gaussian_noise(request):
    from chroma.layers.structure.diffusion import DiffusionChainCov

    covariance_model = request.param
    return DiffusionChainCov(
        covariance_model=covariance_model,
        complex_scaling=False,
        noise_schedule="log_snr",
    )


@pytest.mark.parametrize("conditioner", collect_conditioners())
def test_sampling(gaussian_noise, conditioner, request):
    conditioner_cls, X_native, C = request.getfixturevalue(conditioner.__name__)

    def X0_func(X, C, t):
        return X_native

    out = gaussian_noise.sample_sde(
        X0_func=X0_func, C=C, X_init=None, N=2, conditioner=conditioner_cls
    )


def test_proclass_conditioner(protein):
    """Smoke test for secondary structure conditioning"""
    SECONDARY_STRUCTURE = "CCEEEEEEEETTTTECTTTTTTTTCCCHHHHHHHHHHHHCCCTTTTEEEEEECHHHHHHCTGGTTTTTTTEEEEETTTTTTTTTTTCEEECTHHHHHHHHHCHGHGGHCCEEEEEECHHHHHHHHHCTCEEEEEEEEETTCCCTTEECCCCTGGGTEEETETTTTTCCEEEETTEEEEEEEEEEEC"
    X, C, S = protein.to_XCS()
    X.detach()
    X.requires_grad = True
    O = F.one_hot(S, 20)
    conditioner = conditioners.ProClassConditioner(
        "secondary_structure", SECONDARY_STRUCTURE, device="cpu"
    )
    conditioner(X, C, O, 0, 0.5)
