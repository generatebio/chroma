from pathlib import Path
from unittest import SkipTest, TestCase

import numpy as np
import pytest
import torch

import chroma
from chroma.data import Protein
from chroma.layers.structure.backbone import impute_masked_X
from chroma.layers.structure.mvn import (
    BackboneMVNGlobular,
    BackboneMVNResidueGas,
    ConditionalBackboneMVNGlobular,
)
from chroma.layers.structure.rmsd import BackboneRMSD


@pytest.fixture(params=["brownian", "globular", "residue_gas"])
def noise(request):
    covariance_model = request.param
    if covariance_model in ["brownian", "globular"]:
        return BackboneMVNGlobular(
            covariance_model=covariance_model, complex_scaling=True,
        )
    else:
        return BackboneMVNResidueGas(
            covariance_model=covariance_model, complex_scaling=True,
        )


@pytest.fixture(params=["real", "synthetic"])
def XCS(request):
    xcs_type = request.param
    if xcs_type == "real":
        repo = Path(chroma.__file__).parent.parent
        test_cif = str(Path(repo, "tests", "resources", "6wgl.cif"))
        X, C, S = Protein(test_cif).to_XCS()
    else:
        num_batch, num_residues = 5, 100
        X = 10 * torch.randn([num_batch, num_residues * 4, 3])
        C = torch.ones([num_batch, num_residues])
        S = C.clone()
    return X, C, S


def test_full_covariance_and_sqrt_covariance_computation():
    num_batch, num_residues = 1, 100
    X = 10 * torch.randn([num_batch, num_residues, 4, 3])
    C = torch.ones([num_batch, num_residues])
    S = C.clone()
    D = torch.randint(low=0, high=2, size=(C.size()))

    # Fill in missing pieces
    X = impute_masked_X(X, C)
    C = torch.abs(C)

    mvn = BackboneMVNGlobular(covariance_model="globular", complex_scaling=True,)
    cmvn = ConditionalBackboneMVNGlobular(
        covariance_model="globular", complex_scaling=True, X=X, C=C, D=D
    )

    # Test R
    Z = torch.randn_like(X).reshape(X.shape[0], -1, 3)
    RZ_mvn_implicit = mvn._multiply_R(Z, C)
    RZ_mvn_dense = (cmvn.R @ Z).reshape(RZ_mvn_implicit.shape)
    assert torch.allclose(RZ_mvn_implicit, RZ_mvn_dense, atol=1e-2)

    # Test RRt
    RRt_Z_implicit = mvn.multiply_covariance(Z, C)
    RRt_Z_dense = cmvn.RRt @ Z
    assert torch.allclose(RRt_Z_implicit, RRt_Z_dense, atol=1e-2)


def test_invertibility_R(noise, XCS):
    """Test invertibility of the covariance square root."""
    X, C, S = XCS
    X = X.reshape([X.shape[0], -1, 3])

    Ri_X = noise._multiply_R_inverse(X, C)
    R_Ri_X = noise._multiply_R(Ri_X, C)

    Rti_X = noise._multiply_R_inverse_transpose(X, C)
    Rt_Rti_X = noise._multiply_R_transpose(Rti_X, C)
    X = X.reshape(X.shape[0], C.shape[1], -1, 3)

    Ri_X = Ri_X.reshape(X.shape)
    R_Ri_X = R_Ri_X.reshape(X.shape)
    Rti_X = Rti_X.reshape(X.shape)
    Rt_Rti_X = Rt_Rti_X.reshape(X.shape)

    if False:
        from chroma.layers.structure.diffusion import _debug_viz_XZC

        _debug_viz_XZC(X, Ri_X, C)

    assert torch.allclose(X, R_Ri_X, atol=1e-2)
    assert torch.allclose(X, Rt_Rti_X, atol=1e-2)
    assert not torch.allclose(Ri_X, R_Ri_X, atol=1e-2)
    assert not torch.allclose(Rti_X, Rt_Rti_X, atol=1e-2)


def test_invertibility_covariance(noise, XCS, debug=False):
    """Test invertibility of the covariance matrix.

    Note: the covariance matrix is poorly conditioned for all but
    the smallest systems, so for numerical verification the system needs
    to be small with a large tolerance.
    """
    X, C, S = XCS

    # Cycle constraint
    Ci_X = noise.multiply_inverse_covariance(X, C)
    C_Ci_X = noise.multiply_covariance(Ci_X, C)

    if debug and not torch.allclose(X, C_Ci_X, atol=1e-1):
        from matplotlib import pyplot as plt

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot((X - C_Ci_X).data.numpy().flatten(), ".")
        plt.subplot(3, 1, 2)
        plt.plot(X.data.numpy().flatten())
        plt.subplot(3, 1, 3)
        plt.plot(C.data.numpy().flatten())
        plt.savefig(f"test_icov.pdf")

    assert torch.allclose(X, C_Ci_X, atol=1e-1)
    assert not torch.allclose(Ci_X, C_Ci_X, atol=1e-1)


def test_log_determinant(noise):
    """Test log determinant of the covariance matrix."""
    X, C, S = Protein("5imm").to_XCS()

    X = X[0:1, ...]
    C = C[0:1, ...]
    C = torch.abs(C)
    X = impute_masked_X(X, C)

    if hasattr(noise, "covariance_model"):
        # Use the conditional covariance model to build a dense RRt
        cmvn = ConditionalBackboneMVNGlobular(
            covariance_model=noise.covariance_model,
            complex_scaling=noise.complex_scaling,
            X=X,
            C=C,
            D=C.ne(0).float(),
        )

        R, RRt = cmvn._materialize_RRt(C)
        R = R.data.numpy()
        logdet_dense = 3.0 * np.linalg.slogdet(R)[1]
        logdet = noise.log_determinant(C)

        assert logdet.item() == pytest.approx(logdet_dense.item())


def test_cmvn(noise):
    if isinstance(noise, BackboneMVNResidueGas):
        pass
    else:
        aligner = BackboneRMSD()
        protein = Protein("1drf")
        X, C, S = protein.to_XCS()
        protein.sys.save_selection(gti=list(range(14)), selname="clamp")
        cmvn = ConditionalBackboneMVNGlobular(
            covariance_model=noise.covariance_model,
            complex_scaling=noise.complex_scaling,
            X=X,
            C=C,
            D=protein.get_mask("namesel clamp"),
        )

        X_sample = cmvn.sample()
        _, rmsd = aligner.align(X, X_sample, protein.get_mask("namesel clamp"))
        assert rmsd.item() < 1e-1
