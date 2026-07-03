import os
import socket
from math import isclose
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import torch

import chroma
import chroma.utility.api as api
from chroma.data.protein import Protein
from chroma.layers.structure import conditioners
from chroma.models.chroma import Chroma

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BB_MODEL_PATH = "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_backbone_v1.0.pt"  #'named:nature_v3'
GD_MODEL_PATH = "https://chroma-weights.generatebiomedicines.com/downloads?weights=chroma_design_v1.0.pt"  #'named:nature_v3'

BASE_PATH = str(Path(chroma.__file__).parent.parent)
PROTEIN_SAMPLE = BASE_PATH + "/tests/resources/steps200_seed42_len100.cif"
EXPECTED_ELBO = 6.28411340713501


def _resolve_weight_path(url: str, env_var: str) -> str:
    override = os.getenv(env_var)
    if override is not None:
        if not os.path.exists(override):
            pytest.skip(f"{env_var} is set but the file does not exist: {override}")
        return override

    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    weights_name = parse_qs(parsed.query).get("weights", [None])[0]
    if weights_name is None:
        raise ValueError(f"Could not determine weights name from {url}")

    cached_path = api.download_cache_path(base_url, weights_name)
    if os.path.exists(cached_path):
        return cached_path

    try:
        socket.getaddrinfo(parsed.netloc, 443)
    except socket.gaierror:
        pytest.skip("requires cached weights or network access")

    try:
        return api.download_from_generate(base_url, weights_name, exist_ok=True)
    except FileNotFoundError:
        pytest.skip("requires cached weights or a configured access token")


@pytest.fixture(scope="session")
def chroma():
    bb_weights = _resolve_weight_path(BB_MODEL_PATH, "CHROMA_TEST_BB_WEIGHTS")
    gd_weights = _resolve_weight_path(GD_MODEL_PATH, "CHROMA_TEST_GD_WEIGHTS")
    return Chroma(bb_weights, gd_weights, device="cpu")


def test_chroma(chroma):

    # Fixed Protein Value
    protein = Protein.from_CIF(PROTEIN_SAMPLE)

    # Fixed value test score
    torch.manual_seed(42)
    scores = chroma.score(protein, num_samples=5)
    assert isclose(scores["elbo"].score, EXPECTED_ELBO, abs_tol=1e-3)

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
def test_sample(chroma, conditioner):
    chroma.sample(steps=3, conditioner=conditioner, design_method=None)


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
