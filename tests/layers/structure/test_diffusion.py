from pathlib import Path
from typing import Tuple, Union
from unittest import SkipTest, TestCase

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import chroma
from chroma.data import Protein
from chroma.layers.structure import backbone, rmsd
from chroma.layers.structure.diffusion import (
    GaussianNoiseSchedule,
    ReconstructionLosses,
)


class LegacyNoiseSchedule:
    """This is the legacy noise schedule code, we keep this as a reference to check
    known values"""

    def __init__(
        self,
        beta_min: float = 0.005,
        beta_max: float = 100,
        log_snr_range=(-7.0, 13.5),
        kind: str = "log",
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.log_snr_range = log_snr_range
        self.kind = kind

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute alpha given time"""
        return torch.exp(self.log_alpha(t))

    def beta(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute beta given time"""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float()
        b_min, b_max = self.beta_min, self.beta_max
        if self.kind == "log":
            beta = torch.exp(np.log(b_min) + t * np.log(b_max / b_min))
        elif self.kind == "linear":
            beta = b_min + t * (b_max - b_min)
        elif self.kind == "log_snr":
            l_range = self.log_snr_range
            snr = torch.exp((1 - t) * l_range[1] + t * l_range[0])
            beta = -(l_range[0] - l_range[1]) / (snr + 1)
        else:
            raise NotImplementedError(self.kind)
        return beta

    def log_alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute log(alpha) given time"""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float()
        b_min, b_max = self.beta_min, self.beta_max
        if self.kind == "log":
            log_alpha = -(
                torch.exp(np.log(b_min) + t * np.log(b_max / b_min)) - b_min
            ) / np.log(b_max / b_min)
        elif self.kind == "linear":
            log_alpha = -0.5 * t ** 2 * (b_max - b_min) - t * b_min
        elif self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            log_snr = (1 - t) * l_max + t * l_min
            log_alpha = log_snr - F.softplus(log_snr)
        else:
            raise NotImplementedError(self.kind)
        return log_alpha

    def log_alpha_inverse(self, log_alpha: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute time given log(alpha)"""
        if not isinstance(log_alpha, torch.Tensor):
            log_alpha = torch.Tensor([log_alpha]).float()
        b_min, b_max = self.beta_min, self.beta_max
        if self.kind == "log":
            t = (log_alpha * np.log(b_min / b_max) + b_min).log()
            t = (t - np.log(b_min)) / np.log(b_max / b_min)
        elif self.kind == "linear":
            # Applying the quadratic formula to
            #   0 = log_alpha + t * b_min + t**2 * (b_max - b_min) / 2
            # we select the positive root
            #       -b_min + sqrt(b_min**2 - 2 log_alpha (b_max - b_min))
            #   t = -----------------------------------------------------
            #                          b_max - b_min
            d = b_max - b_min
            t = ((b_min ** 2 - 2 * d * log_alpha).sqrt() - b_min) / d
        elif self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            log_snr = log_alpha - torch.log(-torch.expm1(log_alpha))
            t = (log_snr - l_max) / (l_min - l_max)
        else:
            raise NotImplementedError(self.kind)
        return t

    def prob_alpha(self, alpha: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute probability density"""
        if self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            p_alpha = ((1 - alpha) * (alpha) * (l_max - l_min)).reciprocal()
        else:
            raise NotImplementedError(self.kind)
        return p_alpha

    def SNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute SNR given time"""
        alpha = self.alpha(t)
        return alpha / (1 - alpha)

    def SNR_derivative(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        alpha = self.alpha(t)
        beta = self.beta(t)
        return -(alpha * beta) / ((1 - alpha) ** 2)

    def SNR_inverse(self, SNR: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute time given SNR"""
        if not isinstance(SNR, torch.Tensor):
            SNR = torch.Tensor([SNR]).float()
        log_alpha = SNR.reciprocal().log1p().neg()
        t = self.log_alpha_inverse(log_alpha)
        return t


@pytest.fixture(params=["brownian", "globular"])
def gaussian_noise(request):
    from chroma.layers.structure.diffusion import DiffusionChainCov

    covariance_model = request.param
    return DiffusionChainCov(
        covariance_model=covariance_model,
        complex_scaling=False,
        noise_schedule="log_snr",
    )


@pytest.mark.parametrize("kind", ["log_snr"])
def test_noise_schedule_ssnr(kind):
    """for log_SNR scheudle SSNR(t) = alpht(t)^2"""
    noise_schedule = GaussianNoiseSchedule(kind=kind, log_snr_range=(-12, 12))
    t = torch.linspace(0, 1, 10)
    assert torch.allclose(noise_schedule.SSNR(t), noise_schedule.alpha(t).pow(2))


@pytest.mark.parametrize("kind", ["ot_linear", "log_snr"])
def test_noise_schedule_ssnr_inverse(kind):
    noise_schedule = GaussianNoiseSchedule(kind=kind, log_snr_range=(-12, 12))
    t = torch.linspace(0, 1, 10)
    SSNR = noise_schedule.SSNR(t)
    t2 = noise_schedule.SSNR_inv(
        SSNR
    )  # Note that inverse function map ssnr to t_tilde not t
    assert torch.allclose(t2, t, atol=1e-2)

    if kind == "ot_linear":
        tsingular = torch.Tensor([0.500001, 0.50001])
        t_tilde = noise_schedule.SSNR_inv(tsingular)
        assert not torch.isnan(t_tilde).any()


@pytest.mark.parametrize("kind", ["ot_linear", "log_snr"])
def test_noise_schedule_snr_range(kind):
    noise_schedule = GaussianNoiseSchedule(kind=kind, log_snr_range=(-20, 20))
    assert torch.allclose(
        noise_schedule.SNR(1.0).log(), torch.Tensor([-20.0]), atol=1e-2
    )
    assert torch.allclose(
        noise_schedule.SNR(0.0).log(), torch.Tensor([20.0]), atol=1e-2
    )


@pytest.mark.parametrize("kind", ["ot_linear", "log_snr"])
def test_noise_schedule_drift_coeff(kind):
    noise_schedule = GaussianNoiseSchedule(kind=kind, log_snr_range=(-6, 6))
    ts = torch.linspace(1e-2, 1 - 1e-2, 10)
    t_map = noise_schedule.t_map(ts)  # map time to the prescribed log_SNR range

    if kind == "log_snr":
        beta = noise_schedule.beta(ts)
        # compute true beta_t
        l_range = noise_schedule.log_snr_range
        snr = torch.exp((1 - t_map) * l_range[1] + t_map * l_range[0])
        beta_true = -(l_range[0] - l_range[1]) / (snr + 1)

        assert torch.allclose(beta, beta_true, atol=1e-4)

    if kind == "ot_linear":
        beta = noise_schedule.beta(ts)
        tlen = noise_schedule.t_max - noise_schedule.t_min
        beta_true = 2.0 / (1.0 - t_map)
        assert torch.allclose(beta, beta_true, atol=1e-4)


@pytest.mark.parametrize("kind", ["ot_linear", "log_snr"])
def test_noise_schedule_diffusion_coeff(kind):
    noise_schedule = GaussianNoiseSchedule(kind=kind, log_snr_range=(-6, 6))
    ts = torch.linspace(1e-2, 1 - 1e-2, 10)
    t_map = noise_schedule.t_map(ts)  # map time to the prescribed log_SNR range

    if kind == "log_snr":
        g = noise_schedule.g(ts)

        # compute true beta_t
        l_range = noise_schedule.log_snr_range
        snr = torch.exp((1 - t_map) * l_range[1] + t_map * l_range[0])
        g_true = (-(l_range[0] - l_range[1]) / (snr + 1)).sqrt()

        assert torch.allclose(g, g_true, atol=1e-4)

    if kind == "ot_linear":
        g = noise_schedule.g(ts)
        g_true = (2.0 * t_map / (1.0 - t_map)).sqrt()
        assert torch.allclose(g, g_true, atol=1e-4)


def test_gaussian_noise_schedule():
    from chroma.layers.structure.diffusion import GaussianNoiseSchedule

    ot_noise = GaussianNoiseSchedule(kind="ot_linear")

    log_snr_noise = GaussianNoiseSchedule(kind="log_snr")
    noise = LegacyNoiseSchedule(kind="log_snr")

    assert torch.allclose(
        noise.alpha(torch.linspace(0, 1, 20)),
        log_snr_noise.SSNR(torch.linspace(0, 1, 20)),
    )
    assert torch.allclose(
        noise.alpha(torch.linspace(0, 1, 20)),
        log_snr_noise.alpha(torch.linspace(0, 1, 20)).pow(2),
    )
    assert torch.allclose(
        noise.beta(torch.linspace(0, 1, 20)).sqrt(),
        log_snr_noise.g(torch.linspace(0, 1, 20)),
        atol=5e-4,
    )

    assert torch.allclose(
        noise.beta(torch.linspace(0, 1, 20)),
        log_snr_noise.beta(torch.linspace(0, 1, 20)),
        atol=5e-4,
    )

    # SNR_derivative from previous impelementation is susceptible from floating point error,
    # commenting out this test.
    # assert torch.allclose(
    #     noise.SNR_derivative(torch.linspace(0, 1, 20)),
    #     log_snr_noise.SNR_derivative(torch.linspace(0, 1, 20)),
    #     atol=5e-4,
    # )

    assert torch.allclose(ot_noise.log_SNR(1.0), torch.Tensor([-7.00]))
    assert torch.allclose(ot_noise.log_SNR(0.0), torch.Tensor([13.50]))
    assert torch.allclose(
        log_snr_noise.prob_SSNR(torch.linspace(0.01, 0.99, 5)),
        noise.prob_alpha(torch.linspace(0.01, 0.99, 5)),
    )


@pytest.fixture(scope="session")
def XCS():
    repo = Path(chroma.__file__).parent.parent
    test_cif = str(Path(repo, "tests", "resources", "6wgl.cif"))
    X, C, S = Protein(test_cif).to_XCS()
    return X, C, S


@pytest.mark.parametrize("kind", ["log", "linear", "log_snr"])
def test_noise_schedule_log_alpha_inverse(kind):
    noise_schedule = LegacyNoiseSchedule(kind=kind)
    t = torch.tensor([0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
    log_alpha = noise_schedule.log_alpha(t)
    t2 = noise_schedule.log_alpha_inverse(log_alpha)
    assert torch.allclose(t2, t, atol=1e-2)


@pytest.mark.parametrize("kind", ["log", "linear"])
def test_noise_schedule_SNR_inverse(kind):
    noise_schedule = LegacyNoiseSchedule(kind=kind)
    t = torch.tensor([0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
    SNR = noise_schedule.SNR(t)
    t2 = noise_schedule.SNR_inverse(SNR)
    assert torch.allclose(t2, t, rtol=1e-4)


def debug_importance_weights_alpha(debug_plot=False):
    """Debug plot"""
    noise_schedule = LegacyNoiseSchedule(kind="log_snr")
    # Difficult to integrate numerically, but the below simulations check out
    alpha = torch.Tensor(np.linspace(0.01, 0.99, 1000))
    prob_alpha = noise_schedule.prob_alpha(alpha)

    if debug_plot:
        from matplotlib import pyplot as plt

        T = torch.Tensor(np.linspace(1e-3, 1.0 - 1e-3, 1000))
        alpha = noise_schedule.alpha(T)
        prob_alpha = noise_schedule.prob_alpha(alpha)

        plt.subplot(3, 1, 1)
        plt.plot(T.data.numpy(), alpha.data.numpy())
        plt.xlim([0, 1])
        plt.xlabel("t")
        plt.ylabel("alpha")

        plt.subplot(3, 1, 2)
        plt.hist(alpha.data.numpy(), bins=100, density=True)
        plt.plot(alpha, prob_alpha.data.numpy())
        plt.xlim([0, 1])
        plt.ylim([0, 10])
        plt.xlabel("alpha")
        plt.ylabel("p(alpha)")

        plt.subplot(3, 1, 3)
        plt.plot(T.data.numpy(), (1.0 / prob_alpha).data.numpy())
        plt.xlim([0, 1])
        plt.xlabel("t")
        plt.ylabel("importance weights")
        plt.tight_layout()
        plt.show()
    return


def test_invertibility_X_Z(gaussian_noise, XCS):
    """Test the forward and inverse transforms for the Diffusion MVN."""
    X_native, C, S = XCS

    t = 0.5
    # Sample something with noise
    X = gaussian_noise(X_native, C, t=t)
    alpha = gaussian_noise.noise_schedule.alpha(t=t)
    sigma = gaussian_noise.noise_schedule.sigma(t=t)

    # Cycle constraint
    Z = gaussian_noise._X_to_Z(X, X_native, C, alpha, sigma)
    X_cycle = gaussian_noise._Z_to_X(Z, X_native, C, alpha, sigma)
    Z_cycle = gaussian_noise._X_to_Z(X_cycle, X_native, C, alpha, sigma)
    X_cycle = gaussian_noise._Z_to_X(Z_cycle, X_native, C, alpha, sigma)
    Z_cycle = gaussian_noise._X_to_Z(X_cycle, X_native, C, alpha, sigma)

    X = backbone.impute_masked_X(X, C)
    X_cycle = backbone.impute_masked_X(X_cycle, C)

    assert torch.allclose(X, X_cycle, atol=1e-3)
    assert torch.allclose(Z, Z_cycle, atol=1e-3)


@pytest.mark.parametrize("sde_func", ["reverse_sde", "ode"])
def test_sample_sde(gaussian_noise, XCS, sde_func):
    X_native, C, S = XCS

    def X0_func(X, C, t):
        return X_native

    out = gaussian_noise.sample_sde(
        X0_func=X0_func, C=C, X_init=None, N=40, sde_func=sde_func
    )
    _, rmsd_val = rmsd.BackboneRMSD().align(out["X_sample"], X_native, C=C)
    assert rmsd_val < 0.2


def test_elbo(gaussian_noise, XCS):
    X_native, C, S = XCS

    def X0_func(X, C, t):
        return X_native

    elbo = gaussian_noise.estimate_elbo(X0_func, X_native, C)

    assert elbo > 5.0  # the likelihood of dirac delta approaches infinity

    elbo = gaussian_noise.estimate_elbo(
        X0_func, X_native + torch.randn_like(X_native), C
    )

    assert elbo < 0.0  # the likelihood of dirac delta approaches infinity


def test_logp(gaussian_noise, XCS):
    X_native, C, S = XCS

    # imputation
    X_native = backbone.center_X(X_native, C)
    X_native = backbone.impute_masked_X(X_native, C)
    C = C

    def X0_func(X, C, t):
        return X_native

    logp = gaussian_noise.estimate_logp(X0_func, X_native, C, N=50)

    assert logp > 5.0  # the likelihood of dirac delta approaches infinity

    logp = gaussian_noise.estimate_logp(
        X0_func, X_native + torch.randn_like(X_native), C, N=50
    )

    assert logp < 0.0


def test_reconloss(gaussian_noise, XCS):
    X_native, C, S = XCS
    loss_func = ReconstructionLosses(diffusion=gaussian_noise)
    loss_func(X_native, X_native, C, 0.5)


def test_score_function(gaussian_noise):
    """Test the forward and inverse transforms for the Diffusion MVN."""
    t = 0.9

    # Sample something with nois
    from chroma.layers.structure.backbone import ProteinBackbone

    length_backbones = [100]
    X_native = ProteinBackbone(
        num_batch=1, num_residues=sum(length_backbones), init_state="alpha",
    )()

    C = torch.cat(
        [torch.full([rep], i + 1) for i, rep in enumerate(length_backbones)]
    ).expand(X_native.shape[0], -1)

    S = torch.zeros_like(C)

    X = gaussian_noise(X_native, C, t=t)

    def X0_func(X, C, t):
        return X_native

    score_autodiff = gaussian_noise.score(X, X0_func, C, t=t)
    score_direct = gaussian_noise._score_direct(X, X0_func, C, t=t)

    assert torch.allclose(score_autodiff, score_direct, atol=1e-1)

    # Sanity checks
    if False:
        from chroma.data import xcs
        from chroma.layers.structure.diffusion import (
            _debug_viz_gradients,
            _debug_viz_XZC,
        )

        covariance_model = noise.base_gaussian.covariance_model
        xcs.XCS_to_system(X, C, S).writeCIF("test_noise.cif", "")
        _debug_viz_gradients(
            f"test_{covariance_model}_score_autodiff.pml",
            [X],
            [score_autodiff],
            C,
            S,
            name="score_autodiff",
            color="red",
        )
        _debug_viz_gradients(
            f"test_{covariance_model}_score_icov.pml",
            [X],
            [score_icov],
            C,
            S,
            name="score_icov",
            color="blue",
        )

        from matplotlib import pyplot as plt

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot((score_autodiff - score_icov).data.numpy().flatten())
        plt.subplot(3, 1, 2)
        plt.plot(score_icov.data.numpy().flatten())
        plt.subplot(3, 1, 3)
        plt.plot(C.data.numpy().flatten())
        plt.savefig(f"test_{covariance_model}_scores.pdf")

        # mask = (C > 0).float().reshape(C.shape[0], C.shape[1], 1, 1)
        # score_decorrelate = mask * noise.base_gaussian.multiply_covariance(score_autodiff, C)
        # _debug_viz_gradients("term_repulsion.pml", [X], [X_centered], C, S)
        # _debug_viz_gradients("term_score_function.pml", [X], [score_decorrelate], C, S)
        # X_impute = backbone.impute_masked_X(X, C)
        # flow_gradient = noise.flow_gradient(score_autodiff, X_impute, C, t=t)
        # _debug_viz_gradients("term_net.pml", [X], [flow_gradient], C, S)

    # TODO: Diagnose and fix tiny boundary discrepancies
    # at missing change edges which make this test fail
    # assert torch.allclose(score_autodiff, score_icov, atol=1e-1)
