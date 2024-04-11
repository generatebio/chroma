from math import isclose
from pathlib import Path

import pytest
import torch

import chroma
from chroma.data.protein import Protein
from chroma.layers.structure import conditioners
from chroma.models.chroma import Chroma

BB_MODEL_PATH = "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_backbone_v1.0.pt"  #'named:nature_v3'
GD_MODEL_PATH = "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_design_v1.0.pt"  #'named:nature_v3'

BASE_PATH = str(Path(chroma.__file__).parent.parent)
PROTEIN_SAMPLE = BASE_PATH + "/tests/resources/steps200_seed42_len100.cif"


@pytest.fixture(scope="session")
def chroma():
    return Chroma(BB_MODEL_PATH, GD_MODEL_PATH, device="cpu")


def test_chroma(chroma):

    # Fixed Protein Value
    protein = Protein.from_CIF(PROTEIN_SAMPLE)

    # Fixed value test score
    torch.manual_seed(42)
    scores = chroma.score(protein, num_samples=5)
    assert isclose(scores["elbo"].score, 5.890165328979492, abs_tol=1e-3)

    # Test Sampling & Design
    # torch.manual_seed(42)
    # sample = chroma.sample(steps=200)

    # Xs, _, Ss = sample.to_XCS()
    # X , _, S  = protein.to_XCS()
    # assert torch.allclose(X,Xs)
    # assert (S == Ss).all()

    # test postprocessing
    from chroma.layers.structure import conditioners

    X, C, S = protein.to_XCS()
    c_symmetry = conditioners.SymmetryConditioner(G="C_8", num_chain_neighbors=1)

    X_s, C_s, S_s = (
        torch.cat([X, X], dim=1),
        torch.cat([C, C], dim=1),
        torch.cat([S, S], dim=1),
    )
    protein_sym = Protein(X_s, C_s, S_s)

    chroma._postprocess(c_symmetry, protein_sym, output_dictionary=None)


@pytest.mark.parametrize(
    "conditioner",
    [
        conditioners.Identity(),
        conditioners.SymmetryConditioner(G="C_3", num_chain_neighbors=1),
    ],
)
@pytest.mark.parametrize(
    "conditioner",
    [
        conditioners.Identity(),
        conditioners.SymmetryConditioner(G="C_3", num_chain_neighbors=1),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_sample(chroma, conditioner, batch_size):
    # Generate a batch of proteins with the specified batch size
    proteins = [Protein.from_CIF(PROTEIN_SAMPLE) for _ in range(batch_size)]
    # Stack proteins into a batch
    protein_batch = Protein.stack(proteins)
    # Sample with the specified conditioner and batch of proteins
    chroma.sample(steps=3, conditioner=conditioner, protein_batch=protein_batch, design_method=None)


@pytest.mark.parametrize(
    "conditioner",
    [
        conditioners.Identity(),
        conditioners.SymmetryConditioner(G="C_3", num_chain_neighbors=1),
    ],
)
def test_sample_backbone(chroma, conditioner):
    chroma._sample(steps=3, conditioner=conditioner)


@pytest.mark.parametrize("design_method", ["autoregressive", "potts",])
@pytest.mark.parametrize("potts_proposal", ["dlmc", "chromatic"])
def test_design(chroma, design_method, potts_proposal):
    protein = Protein.from_CIF(PROTEIN_SAMPLE)
    chroma.design(
        protein,
        design_method=design_method,
        potts_proposal=potts_proposal,
        potts_mcmc_depth=20,
    )
