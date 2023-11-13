# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers for perturbing protein structure with noise.

This module contains pytorch layers for perturbing protein structure with noise,
which can be useful both for data augmentation, benchmarking, or denoising based
training.
"""


from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from tqdm.auto import tqdm

from chroma.constants import AA20
from chroma.data.xcs import validate_XC
from chroma.layers import basic, sde
from chroma.layers.structure import backbone, hbonds, mvn, rmsd


class GaussianNoiseSchedule:
    """
    A general noise schedule for the General Gaussian Forward Path, where noise is added
    to the input signal.

    The noise is modeled as Gaussian noise with mean `alpha_t x_0` and variance
     `sigma_t^2`, with 'x_0 ~ p(x_0)' The time range of the noise schedule is
     parameterized with a user-specified logarithmic signal-to-noise ratio (SNR) range,
    where  `snr_t = alpha_t^2 / sigma_t^2` is the SNR at time `t`.

    In addition, the object defines a quantity called the scaled signal-to-noise ratio
    (`ssnr_t`), which is given by `ssnr_t = alpha_t^2 / (alpha_t^2 + sigma_t^2)`
    and is a helpful quantity for analyzing the performance of signal processing
    algorithms under different noise conditions.

    This object implements a few standard noise schedule:

        'log_snr': variance-preserving process with linear log SNR schedule
        (https://arxiv.org/abs/2107.00630)

        'ot_linear': OT schedule
        (https://arxiv.org/abs/2210.02747)

        've_log_snr': variance-exploding process with linear log SNR s hedule
        (https://arxiv.org/abs/2011.13456 with log SNR noise schedule)

    User can also implement their own schedules by specifying alpha_func, sigma_func
    and compute_t_range.

    """

    def __init__(
        self, log_snr_range: Tuple[float, float] = (-7.0, 13.5), kind: str = "log_snr",
    ) -> None:
        super().__init__()

        if kind not in ["log_snr", "ot_linear", "ve_log_snr"]:
            raise NotImplementedError(
                f"noise type {kind} is not implemented,                            only"
                " log_snr and ot_linear are supported "
            )
        self.kind = kind
        self.log_snr_range = log_snr_range

        l_min, l_max = self.log_snr_range

        # map t \in [0, 1] to match the prescribed log_snr range
        self.t_max = self.compute_t_range(l_min)
        self.t_min = self.compute_t_range(l_max)
        self._eps = 1e-5

    def t_map(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """map t in [0, 1] to [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time

        Returns:
            torch.Tensor: mapped time
        """
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float()

        t_max = self.t_max.to(t.device)
        t_min = self.t_min.to(t.device)
        t_tilde = t_min + (t_max - t_min) * t

        return t_tilde

    def derivative(self, t: torch.Tensor, func: Callable) -> torch.Tensor:
        """compute derivative of a function, it supports bached single variable inputs

        Args:
            t (torch.Tensor): time variable at which derivatives are taken
            func (Callable): function for derivative calculation

        Returns:
            torch.Tensor: derivative that is detached from the computational graph
        """
        with torch.enable_grad():
            t.requires_grad_(True)
            derivative = grad(func(t).sum(), t, create_graph=False)[0].detach()
            t.requires_grad_(False)
        return derivative

    def tensor_check(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """convert input to torch.Tensor if it is a float

        Args:
            t ( Union[float, torch.Tensor]): input

        Returns:
            torch.Tensor: converted torch.Tensor
        """
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float()
        return t

    def alpha_func(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """alpha function that scales the mean, usually goes from 1. to 0.

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: alpha value
        """

        t = self.tensor_check(t)

        if self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            t_min, t_max = self.t_min, self.t_max
            log_snr = (1 - t) * l_max + t * l_min

            log_alpha = 0.5 * (log_snr - F.softplus(log_snr))
            alpha = log_alpha.exp()
            return alpha

        elif self.kind == "ve_log_snr":
            return 1 - torch.relu(-t)  # make this differentiable

        elif self.kind == "ot_linear":
            return 1 - t

    def sigma_func(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """sigma function that scales the standard deviation, usually goes from 0. to 1.

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: sigma value
        """
        t = self.tensor_check(t)
        l_min, l_max = self.log_snr_range

        if self.kind == "log_snr":
            alpha = self.alpha(t)
            return (1 - alpha.pow(2)).sqrt()

        elif self.kind == "ve_log_snr":
            # compute sigma value given snr range

            l_min, l_max = self.log_snr_range
            t_min, t_max = self.t_min, self.t_max
            log_snr = (1 - t) * l_max + t * l_min
            return torch.exp(-log_snr / 2)

        elif self.kind == "ot_linear":
            return t

    def alpha(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute alpha value for the mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: alpha value
        """
        return self.alpha_func(self.t_map(t))

    def sigma(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute sigma value for mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: sigma value
        """
        return self.sigma_func(self.t_map(t))

    def alpha_deriv(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute alpha derivative for mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: time derivative of alpha_func
        """
        t_tilde = self.t_map(t)
        alpha_deriv_t = self.derivative(t_tilde, self.alpha_func).detach()
        return alpha_deriv_t

    def sigma_deriv(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute sigma derivative for the mapped time in [t_min, t_max]

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: sigma derivative
        """
        t_tilde = self.t_map(t)
        sigma_deriv_t = self.derivative(t_tilde, self.sigma_func).detach()
        return sigma_deriv_t

    def beta(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute the drift coefficient for the OU process of the form
        $dx = -\frac{1}{2} \beta(t) x dt + g(t) dw_t$

        Args:
            t (Union[float, torch.Tensor]): t in [0, 1]

        Returns:
            torch.Tensor: beta(t)
        """
        # t = self.t_map(t)
        alpha = self.alpha(t).detach()
        t_map = self.t_map(t)
        alpha_deriv_t = self.alpha_deriv(t)
        beta = -2.0 * alpha_deriv_t / alpha

        return beta

    def g(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute drift coefficient for the OU process:
        $dx = -\frac{1}{2} \beta(t) x dt + g(t) dw_t$

        Args:
            t (Union[float, torch.Tensor]): t in [0, 1]

        Returns:
            torch.Tensor: g(t)
        """
        if self.kind == "log_snr":
            t = self.t_map(t)
            g = self.beta(t).sqrt()

        else:
            alpha_deriv = self.alpha_deriv(t)
            alpha_prime_div_alpha = alpha_deriv / self.alpha(t)
            sigma_deriv = self.sigma_deriv(t)
            sigma_prime_div_sigma = sigma_deriv / self.sigma(t)

            g_sq = (
                2
                * (sigma_deriv - alpha_prime_div_alpha * self.sigma(t))
                * self.sigma(t)
            )
            g = g_sq.sqrt()

        return g

    def SNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Signal-to-Noise(SNR) ratio  mapped in the allowed log_SNR range

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: SNR value
        """
        t = self.tensor_check(t)

        if self.kind == "log_snr":
            SNR = self.log_SNR(t).exp()
        else:
            SNR = self.alpha(t).pow(2) / (self.sigma(t).pow(2))

        return SNR

    def log_SNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """log SNR value

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: log SNR value
        """
        t = self.tensor_check(t)

        if self.kind == "log_snr":
            l_min, l_max = self.log_snr_range
            log_snr = (1 - t) * l_max + t * l_min

        elif self.kind == "ot_linear":
            log_snr = self.SNR(t).log()

        return log_snr

    def compute_t_range(self, log_snr: Union[float, torch.Tensor]) -> torch.Tensor:
        """Given log(SNR) range : l_max, l_min to compute the time range.
        Hand-derivation is required for specific noise schedules.
        This function is essentially the inverse of logSNR(t)

        Args:
            log_snr (Union[float, torch.Tensor]): logSNR value

        Returns:
            torch.Tensor: the inverse logSNR
        """
        log_snr = self.tensor_check(log_snr)
        l_min, l_max = self.log_snr_range

        if self.kind == "log_snr":
            t = (1 / (l_min - l_max)) * (log_snr - l_max)

        elif self.kind == "ot_linear":
            t = ((0.5 * log_snr).exp() + 1).reciprocal()

        elif self.kind == "ve_log_snr":
            t = (1 / (l_min - l_max)) * (log_snr - l_max)

        return t

    def SNR_derivative(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """the derivative of SNR(t)

        Args:
            t (Union[float, torch.Tensor]): t in [0, 1]

        Returns:
            torch.Tensor: SNR derivative
        """
        t = self.tensor_check(t)

        if self.kind == "log_snr":
            snr_deriv = self.SNR(t) * (self.log_snr_range[0] - self.log_snr_range[1])

        elif self.kind == "ot_linear":
            snr_deriv = self.derivative(t, self.SNR)
        return snr_deriv

    def SSNR(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Signal to Signal+Noise Ratio (SSNR) = alpha^2 / (alpha^2 + sigma^2)
           SSNR monotonically goes from 1 to 0 as t going from 0 to 1.

        Args:
            t (Union[float, torch.Tensor]): time in [0, 1]

        Returns:
            torch.Tensor: SSNR value
        """
        t = self.tensor_check(t)
        return self.SNR(t) / (self.SNR(t) + 1)

    def SSNR_inv(self, ssnr: torch.Tensor) -> torch.Tensor:
        """the inverse of SSNR

        Args:
            ssnr (torch.Tensor): ssnr in [0, 1]

        Returns:
            torch.Tensor: time in [0, 1]
        """
        l_min, l_max = self.log_snr_range
        if self.kind == "log_snr":
            return ((ssnr / (1 - ssnr)).log() - l_max) / (l_min - l_max)
        elif self.kind == "ot_linear":
            # the value of SNNR_inv(t=0.5) need to be determined with L'Hôpital rule
            # the inver SNNR_function is solved anyltically:
            # see woflram alpha result: https://tinyurl.com/bdh4es5a
            singularity_check = (ssnr - 0.5).abs() < self._eps
            ssnr_mask = singularity_check.float()
            ssnr = ssnr_mask * (0.5 + self._eps) + (1.0 - ssnr_mask) * ssnr

            return (ssnr + (-ssnr * (ssnr - 1)).sqrt() - 1) / (2 * ssnr - 1)

    def SSNR_inv_deriv(self, ssnr: Union[float, torch.Tensor]) -> torch.Tensor:
        """SSNR_inv derivative. SSNR_inv is a CDF like quantity, so its derivative is a PDF-like quantity

        Args:
            ssnr (Union[float, torch.Tensor]): SSNR in [0, 1]

        Returns:
            torch.Tensor: derivative of SSNR
        """
        ssnr = self.tensor_check(ssnr)
        deriv = self.derivative(ssnr, self.SSNR_inv)
        return deriv

    def prob_SSNR(self, ssnr: Union[float, torch.Tensor]) -> torch.Tensor:
        """compute prob (SSNR(t)), the minus sign is accounted for the inversion of integration range

        Args:
            ssnr (Union[float, torch.Tensor]): SSNR value

        Returns:
            torch.Tensor: Prob(SSNR)
        """
        return -self.SSNR_inv_deriv(ssnr)

    def linear_logsnr_grid(self, N: int, tspan: Tuple[float, float]) -> torch.Tensor:
        """Map uniform time grid to respect logSNR schedule

        Args:
            N (int): number of steps
            tspan (Tuple[float, float]): time span (t_start, t_end)

        Returns:
            torch.Tensor: time grid as torch.Tensor
        """

        logsnr_noise = GaussianNoiseSchedule(
            kind="log_snr", log_snr_range=self.log_snr_range
        )

        ts = torch.linspace(tspan[0], tspan[1], N + 1)
        SSNR_vp = logsnr_noise.SSNR(ts)
        grid = self.SSNR_inv(SSNR_vp)

        # map from t_tilde back to t
        grid = (grid - self.t_min) / (self.t_max - self.t_min)

        return grid


class NoiseTimeEmbedding(nn.Module):
    """
    A class that implements a noise time embedding layer.

    Args:
        dim_embedding (int): The dimension of the output embedding vector.
            noise_schedule (GaussianNoiseSchedule): A GaussianNoiseSchedule object that
            defines the noise schedule function.
        rff_scale (float, optional): The scaling factor for the random Fourier features.
            Default is 0.8.
        feature_type (str, optional): The type of feature to use for the time embedding.
            Either "t" or "log_snr". Default is "log_snr".

    Inputs:
        t (float): time in (1.0, 0.0).
        log_alpha (torch.Tensor, optional): A tensor of log alpha values with
            shape `(batch_size,)`.

    Outputs:
        time_h (torch.Tensor): A tensor of noise time embeddings with shape
         `(batch_size, dim_embedding)`.
    """

    def __init__(
        self,
        dim_embedding: int,
        noise_schedule: GaussianNoiseSchedule,
        rff_scale: float = 0.8,
        feature_type: str = "log_snr",
    ) -> None:
        super(NoiseTimeEmbedding, self).__init__()
        self.noise_schedule = noise_schedule
        self.feature_type = feature_type
        self.fourier_features = basic.FourierFeaturization(
            d_input=1, d_model=dim_embedding, trainable=False, scale=rff_scale
        )

    def forward(
        self, t: torch.Tensor, log_alpha: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(self.fourier_features.B.device)
        if t.dim() == 0:
            t = t[None]

        h = {"t": lambda: t, "log_snr": lambda: self.noise_schedule.log_SNR(t)}[
            self.feature_type
        ]()

        time_h = self.fourier_features(h[:, None, None])
        return time_h


class DiffusionChainCov(nn.Module):
    def __init__(
        self,
        log_snr_range: Tuple[float, float] = (-7.0, 13.5),
        noise_schedule: str = "log_snr",
        sigma_translation: float = 1.0,
        covariance_model: str = "brownian",
        complex_scaling: bool = False,
        **kwargs,
    ) -> None:
        """Diffusion backbone noise, with chain-structured covariance.

        This class implements a diffusion backbone noise model. The model uses a
        chain-structured covariance matrix capturing the spatial correlations between
        residues along the backbone. The model also supports different noise schedules
        and integration schemes for the stochastic differential equation (SDE) that
        defines the diffusion process. This class also implemented various inference
        algorithm by reversing the forward diffusion with user-specified
        conditioner program.

        Args:
            log_snr_range (tuple, optional): log SNR range. Defaults to (-7.0, 13.5).
            noise_schedule (str, optional): noise schedule type. Defaults to "log_snr".
            sigma_translation (float, optional): Scaling factor for the translation
                component of the covariance matrix. Defaults to 1.0.
            covariance_model (str, optional): covariance mode,. Defaults to "brownian".
            complex_scaling (bool, optional): Whether to scale the complex component
                of the covariance matrix by the translation component. Defaults to False.
            **kwargs: Additional arguments for the base Gaussian distribution and
                 the SDE integration.
        """
        super().__init__()

        self.noise_schedule = GaussianNoiseSchedule(
            log_snr_range=log_snr_range, kind=noise_schedule,
        )

        if covariance_model in ["brownian", "globular"]:
            self.base_gaussian = mvn.BackboneMVNGlobular(
                sigma_translation=sigma_translation,
                covariance_model=covariance_model,
                complex_scaling=complex_scaling,
            )
        elif covariance_model == "residue_gas":
            self.base_gaussian = mvn.BackboneMVNResidueGas()

        self.loss_rmsd = rmsd.BackboneRMSD()
        self._eps = 1e-5
        self.sde_funcs = {
            "langevin": self.langevin,
            "reverse_sde": self.reverse_sde,
            "ode": self.ode,
        }
        self.integrate_funcs = {
            "euler_maruyama": sde.sde_integrate,
            "heun": sde.sde_integrate_heun,
        }

    def sample_t(
        self,
        C: torch.LongTensor,
        t: Optional[torch.Tensor] = None,
        inverse_CDF: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Sample a random time index for each batch element

        Inputs:
            C (torch.LongTensor): Chain tensor with shape `(batch_size, num_residues)`.
            t (torch.Tensor, optional): Time index with shape `(batch_size,)`.
                If not given, a random time index will be sampled. Defaults to None.

        Outputs:
            t (float): Time index with shape `(batch_size,)`.
        """
        if t is not None:
            if not isinstance(t, torch.Tensor):
                t = torch.Tensor([t]).float()
            return t

        num_batch = C.size(0)
        if self.training:
            # Sample correlated but marginally uniform t
            # for variance reduction (Kingma et al 2021)
            u = torch.rand([])
            ix = torch.arange(num_batch) / num_batch
            t = torch.remainder(u + ix, 1)
        else:
            t = torch.rand([num_batch])
        if inverse_CDF is not None:
            t = inverse_CDF(t)
        t = t.to(C.device)
        return t

    def sde_forward(self, X, C, t, Z=None):
        """Sample an Euler-Maruyama step on forwards SDE.

        That is to say, Euler-Maruyama integration would
        correspond to the update.
            `X_new = X + dt * f + sqrt(dt) * gZ`

        Args:

        Returns:
            f (Tensor): Drift term with shape `()`.
            gZ (Tensor): Diffusion term  with shape `()`.
        """

        # Sample random perturbation
        if Z is None:
            Z = torch.randn_like(X)
        Z = Z.reshape(X.shape[0], -1, 3)
        R_Z = self.base_gaussian._multiply_R(Z, C).reshape(X.shape)

        X = backbone.center_X(X, C)
        beta = self.noise_schedule.beta(t)
        f = -beta * X / 2.0
        gZ = self.noise_schedule.g(t)[:, None, None] * R_Z

        return f, gZ

    def _schedule_coefficients(
        self,
        t: torch.Tensor,
        inverse_temperature: float = 1.0,
        langevin_isothermal: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        A method that computes the schedule coefficients for sampling in the reverse time

        Args:
            t (float): time in (1.0, 0.0).
            inverse_temperature (float, optional): The inverse temperature parameter for
                he Langevin dynamics. Default is 1.0.
            langevin_isothermal (bool, optional): A flag that indicates whether to use
                isothermal or non-isothermal Langevin dynamics. Default is True.

        Returns:
            alpha (torch.Tensor): A tensor of alpha values with shape `(batch_size, 1, 1)`.
            sigma (torch.Tensor): A tensor of sigma values with shape `(batch_size, 1, 1)`.
            beta (torch.Tensor): A tensor of beta values with shape `(batch_size, 1, 1)`.
            g (torch.Tensor): A tensor of g values with shape `(batch_size, 1, 1)`.
            lambda_t (float): A tensor of lambda_t values with shape `(batch_size, 1, 1)`.
            lambda_langevin (torch.Tensor): A tensor of lambda_langevin values with
                shape `(batch_size, 1, 1)`.
        """

        # Schedule coeffiecients
        alpha = self.noise_schedule.alpha(t)[:, None, None].to(t.device)
        sigma = self.noise_schedule.sigma(t)[:, None, None].to(t.device)
        beta = self.noise_schedule.beta(t)[:, None, None].to(t.device)
        g = self.noise_schedule.g(t)[:, None, None].to(t.device)

        # Temperature coefficients
        lambda_t = (
            inverse_temperature
            * (sigma.pow(2) + alpha.pow(2))
            / (inverse_temperature * sigma.pow(2) + alpha.pow(2))
        )
        lambda_langevin = inverse_temperature if langevin_isothermal else lambda_t
        return alpha, sigma, beta, g, lambda_t, lambda_langevin

    @validate_XC()
    def langevin(
        self,
        X: torch.Tensor,
        X0_func: Callable,
        C: torch.LongTensor,
        t: Union[torch.Tensor, float],
        conditioner: Callable = None,
        Z: Union[torch.Tensor, None] = None,
        inverse_temperature: float = 1.0,
        langevin_factor: float = 0.0,
        langevin_isothermal: bool = True,
        align_X0: bool = True,
    ):
        """Return the drift and diffusion components of the Langevin dynamics for the
            reverse process

        Args:
            X (torch.Tensor): A tensor of protein backbone structure with shape
                `(batch_size, num_residues, 4, 3)`.
            X0_func (Callable): A function a denoising function for protein backbon
                e geometry.
            C (torch.LongTensor): A chain map tensor with shape `(batch_size, num_residues)`.
            t (float): time in (1.0, 0.0).
            conditioner (Callable, optional): A conditioner the performs constrained
                transformation (see examples in chroma.layers.structure.conditioners).
            Z (torch.Tensor, optional): A tensor of random noise with
                 shape `(batch_size, num_residues, 4, 3)`. Default is None.
            inverse_temperature (float, optional): The inverse temperature parameter
                 for the Langevin dynamics. Default is 1.0.
            langevin_factor (float, optional): The scaling factor for the Langevin noise.
                 Default is 1.0.
            langevin_isothermal (bool, optional): A flag that indicates whether to use
                 isothermal or non-isothermal Langevin dynamics. Default is True.
            align_X0 (bool, optional): A flag that indicates whether to align the noised
                 X and denoised X for score function calculation.

        Returns:
            f (torch.Tensor): A tensor of drift terms with shape
                `(batch_size, num_residues, 4, 3)`.
            gZ (torch.Tensor): A tensor of diffusion terms with shape
                `(batch_size, num_residues, 4, 3)`.
        """

        alpha, sigma, beta, g, lambda_t, lambda_langevin = self._schedule_coefficients(
            t,
            inverse_temperature=inverse_temperature,
            langevin_isothermal=langevin_isothermal,
        )

        Z = torch.randn_like(X) if Z is None else Z

        score = self.score(X, X0_func, C, t, conditioner, align_X0=align_X0)
        score_transformed = self.base_gaussian.multiply_covariance(score, C)
        f = -g.pow(2) * lambda_langevin * langevin_factor / 2.0 * score_transformed
        gZ = g * np.sqrt(langevin_factor) * self.base_gaussian._multiply_R(Z, C)
        return f, gZ

    @validate_XC()
    def reverse_sde(
        self,
        X: torch.Tensor,
        X0_func: Callable,
        C: torch.LongTensor,
        t: Union[torch.Tensor, float],
        conditioner: Callable = None,
        Z: Union[torch.Tensor, None] = None,
        inverse_temperature: float = 1.0,
        langevin_factor: float = 0.0,
        langevin_isothermal: bool = True,
        align_X0: bool = True,
    ):
        """Return the drift and diffusion components of the reverse SDE.

        Args:
            X (torch.Tensor): A tensor of protein backbone structure with shape
                `(batch_size, num_residues, 4, 3)`.
            X0_func (Callable): A function a denoising function for the protein backbone
                geometry.
            C (torch.LongTensor): A tensor of condition features with shape
                `(batch_size, num_residues)`.
            t (float): time in (1.0, 0.0).
            conditioner (Callable, optional): A conditioner the performs constrained
                 transformation (see examples in chroma.layers.structure.conditioners).
            Z (torch.Tensor, optional): A tensor of random noise with shape
                 `(batch_size, num_residues, 4, 3)`. Default is None.
            inverse_temperature (float, optional): The inverse temperature parameter
                for the Langevin dynamics. Default is 1.0.
            langevin_factor (float, optional): The scaling factor for the Langevin noise.
                 Default is 0.0.
            langevin_isothermal (bool, optional): A flag that indicates whether to use
                isothermal or non-isothermal Langevin dynamics. Default is True.
            align_X0 (bool, optional): A flag that indicates whether to align the noised
                 X and denoised X for score function calculation.

        Returns:
            f (torch.Tensor): A tensor of drift terms with shape
                 `(batch_size, num_residues, 4, 3)`.
            gZ (torch.Tensor): A tensor of diffusion terms with shape
                 `(batch_size, num_residues, 4, 3)`.
        """

        # Schedule management
        alpha, sigma, beta, g, lambda_t, lambda_langevin = self._schedule_coefficients(
            t,
            inverse_temperature=inverse_temperature,
            langevin_isothermal=langevin_isothermal,
        )
        score_scale_t = lambda_t + lambda_langevin * langevin_factor / 2.0

        # Impute missing data
        Z = torch.randn_like(X) if Z is None else Z

        # X = backbone.center_X(X, C)
        score = self.score(X, X0_func, C, t, conditioner, align_X0=align_X0)
        score_transformed = self.base_gaussian.multiply_covariance(score, C)

        f = (
            beta * (-1 / 2) * backbone.center_X(X, C)
            - g.pow(2) * score_scale_t * score_transformed
        )
        gZ = g * np.sqrt(1.0 + langevin_factor) * self.base_gaussian._multiply_R(Z, C)
        return f, gZ

    @validate_XC()
    def ode(
        self,
        X: torch.Tensor,
        X0_func: Callable,
        C: torch.LongTensor,
        t: Union[torch.Tensor, float],
        conditioner: Callable = None,
        Z: Union[torch.Tensor, None] = None,
        inverse_temperature: float = 1.0,
        langevin_factor: float = 0.0,
        langevin_isothermal: bool = True,
        align_X0: bool = True,
        detach_X0: bool = True,
    ):
        """Return the drift and diffusion components of the probability flow ODE.

        Args:
            X (torch.Tensor): A tensor of protein backbone structure with shape
                 `(batch_size, num_residues, 4, 3)`.
            X0_func (Callable): A denoising function that returns a protein backbone
                 geometry `(batch_size, num_residues, 4, 3)`.
            C (torch.LongTensor): A tensor of condition features with shape
                `(batch_size, num_residues)`.
            t (float): time in (1.0, 0.0).
            conditioner (Callable, optional): A conditioner the performs constrained
                transformation (see examples in chroma.layers.structure.conditioners).
            Z (torch.Tensor, optional): A tensor of random noise with shape
                 `(batch_size, num_residues, 4, 3)`. Default is None.
            inverse_temperature (float, optional): The inverse temperature parameter
                 for the Langevin dynamics. Default is 1.0.
            langevin_factor (float, optional): The scaling factor for the Langevin
                 noise. Default is 0.0.
            langevin_isothermal (bool, optional): A flag that indicates whether to use
                isothermal or non-isothermal Langevin dynamics. Default is True.
            align_X0 (bool, optional): A flag that indicates whether to align
                the noised X and denoised X for score function calculation.

        Returns:
            f (torch.Tensor): A tensor of drift terms with shape
                `(batch_size, num_residues, 4, 3)`.
            gZ (torch.Tensor): A tensor of diffusion terms with shape
                 `(batch_size, num_residues, 4, 3)`.
        """

        # Schedule management
        alpha, sigma, beta, g, lambda_t, lambda_langevin = self._schedule_coefficients(
            t,
            inverse_temperature=inverse_temperature,
            langevin_isothermal=langevin_isothermal,
        )

        # Impute missing data
        X = backbone.center_X(X, C)
        score = self.score(
            X, X0_func, C, t, conditioner, align_X0=align_X0, detach_X0=detach_X0
        )
        score_transformed = self.base_gaussian.multiply_covariance(score, C)
        f = (-1 / 2) * beta * X - 0.5 * lambda_langevin * g.pow(2) * score_transformed
        gZ = torch.zeros_like(f)
        return f, gZ

    @validate_XC()
    def energy(
        self,
        X: torch.Tensor,
        X0_func: Callable,
        C: torch.Tensor,
        t: torch.Tensor,
        detach_X0: bool = True,
        align_X0: bool = True,
    ) -> torch.Tensor:
        """Compute the diffusion energy as a function of denoised X

        Args:
            X (torch.Tensor): A tensor of protein backbone coordinates with shape
                 `(batch_size, num_residues, 4, 3)`.
            X0_func (Callable): A function a denoising function for protein backbone
                 geometry.
            C (torch.LongTensor): A tensor of condition features with shape
                `(batch_size, num_residues)`.
            t (float): time in (1.0, 0.0).
            detach_X0 (bool, optional): A flag that indicates whether to detach the
                denoise X for score function evaluation
            align_X0 (bool, optional): A flag that indicates whether to align the
                 noised X and denoised X for score function calculation.

        Returns:
            U_diffusion (torch.Tensor): A tensor of diffusion energy values with
                 shape `(batch_size,)`.
        """

        X = backbone.impute_masked_X(X, C)
        alpha = self.noise_schedule.alpha(t).to(X.device)
        sigma = self.noise_schedule.sigma(t).to(X.device)
        if detach_X0:
            with torch.no_grad():
                X0 = X0_func(X, C, t=t)
        else:
            X0 = X0_func(X, C, t=t)
        if align_X0:
            X0, _ = self.loss_rmsd.align(X0, X, C, align_unmasked=True)
        if detach_X0:
            X0 = X0.detach()
        Z = self._X_to_Z(X, X0, C, alpha, sigma)
        U_diffusion = (0.5 * (Z ** 2)).sum([1, 2, 3])
        return U_diffusion

    @validate_XC()
    def score(
        self,
        X: torch.Tensor,
        X0_func: Callable,
        C: torch.Tensor,
        t: Union[torch.Tensor, float],
        conditioner: Callable = None,
        detach_X0: bool = True,
        align_X0: bool = True,
        U_traj: List = [],
    ) -> torch.Tensor:
        """Compute the score function

        Args:
            X (torch.Tensor): A tensor of protein back geometry with shape
                 `(batch_size, num_residues, 4, 3)`.
            X0_func (Callable): A function a denoising function for protein backbone
                 geometry.
            C (torch.LongTensor): A tensor of chain map with shape
                `(batch_size, num_residues)`.
            t (Union[torch.Tensor, float]): time in (1.0, 0.0).
            conditioner (Callable, optional): A conditioner the performs constrained
                transformation (see examples in chroma.layers.structure.conditioners).
            detach_X0 (bool, optional): A flag that indicates whether to detach the
                 denoised X for score function evaluation
            align_X0 (bool, optional): A flag that indicates whether to align the
                 noised X and denoised X for score function calculation.
            U_traj (List, optional): Record diffusion energy as a list.

        Returns:
            score (torch.Tensor): A tensor of score values with shape
                 `(batch_size, num_residues, 4, 3)`.
        """

        X = backbone.impute_masked_X(X, C)
        with torch.enable_grad():
            X = X.detach().clone()
            X.requires_grad = True

            # Apply optional conditioner transformations to state and energy
            Xt, Ct, U_conditioner = X, C, 0.0
            St = torch.zeros(Ct.shape, device=Xt.device).long()
            Ot = F.one_hot(St, len(AA20)).float()
            if conditioner is not None:
                Xt, Ct, _, U_conditioner, _ = conditioner(X, C, Ot, U_conditioner, t)
            U_conditioner = torch.as_tensor(U_conditioner)

            # Compute system energy
            U_diffusion = self.energy(
                Xt, X0_func, Ct, t, detach_X0=detach_X0, align_X0=align_X0
            )

            U_traj.append(U_diffusion.detach().cpu())

            # Compute score function as negative energy gradient
            U_total = U_diffusion.sum() + U_conditioner.sum()
            U_total.backward()
            score = -X.grad
            score = score.masked_fill((C <= 0)[..., None, None], 0.0)
        return score

    def elbo(self, X0_pred, X0, C, t):
        """ITD ELBO as a weighted average of denoising error,
        inspired by https://arxiv.org/abs/2302.03792"""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(X0.device)

        # Interpolate missing data with Brownian Bridge posterior
        X0 = backbone.impute_masked_X(X0, C)
        X0_pred = backbone.impute_masked_X(X0_pred, C)

        # Compute whitened residual
        dX = (X0 - X0_pred).reshape([X0.shape[0], -1, 3])
        R_inv_dX = self.base_gaussian._multiply_R_inverse(dX, C)

        # Average per atom, including over "missing" positions that we filled in

        weight = 0.5 * self.noise_schedule.SNR_derivative(t)[:, None, None, None]
        snr = self.noise_schedule.SNR(t)[:, None, None, None]
        loss_itd = (
            weight * (R_inv_dX.pow(2) - 1 / (1 + snr))
            - 0.5 * np.log(np.pi * 2.0 * np.e)
        ).reshape(X0.shape)

        # Compute average per-atom loss (including over missing regions)
        mask = (C != 0).float()
        mask_atoms = mask.reshape(mask.shape + (1, 1)).expand([-1, -1, 4, 1])

        # Per-complex
        elbo_gap = (mask_atoms * loss_itd).sum([1, 2, 3])
        logdet = self.base_gaussian.log_determinant(C)
        elbo_unnormalized = elbo_gap - logdet

        # Normalize per atom
        elbo = elbo_unnormalized / (mask_atoms.sum([1, 2, 3]) + self._eps)

        # Compute batch average
        weights = mask_atoms.sum([1, 2, 3])
        elbo_batch = (weights * elbo).sum() / (weights.sum() + self._eps)
        return elbo, elbo_batch

    def pseudoelbo(self, loss_per_residue, C, t):
        """Compute pseudo-ELBOs as weighted averages of other errors."""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(C.device)

        # Average per atom, including over x"missing" positions that we filled in
        weight = 0.5 * self.noise_schedule.SNR_derivative(t)[:, None]
        loss = weight * loss_per_residue

        # Compute average loss
        mask = (C > 0).float()
        pseudoelbo = (mask * loss).sum(-1) / (mask.sum(-1) + self._eps)
        pseudoelbo_batch = (mask * loss).sum() / (mask.sum() + self._eps)
        return pseudoelbo, pseudoelbo_batch

    def _baoab_sample_step(
        self,
        _x,
        p,
        C,
        t,
        dt,
        score_func,
        gamma=2.0,
        kT=1.0,
        n_equil=1,
        ode_boost=True,
        langevin_isothermal=False,
    ):
        gamma = torch.Tensor([gamma]).to(_x.device)
        (
            alpha,
            sigma,
            beta,
            g,
            lambda_t,
            lambda_langevin,
        ) = self._schedule_coefficients(
            t, inverse_temperature=1 / kT, langevin_isothermal=langevin_isothermal,
        )

        def baoab_step(_x, p, t):
            Z = torch.randn_like(_x)
            c1 = torch.exp(-gamma * dt)
            c3 = torch.sqrt((1 / lambda_t) * (1 - c1 ** 2))

            # BAOAB scheme
            p_half = p + score_func(t, C, _x) * dt / 2  # B
            _x_half = (
                _x
                + g.pow(2) * self.base_gaussian.multiply_covariance(p_half, C) * dt / 2
            )  # A
            p_half2 = c1 * p_half + c3 * (
                1 / g
            ) * self.base_gaussian._multiply_R_inverse_transpose(
                Z, C
            )  # O
            _x = (
                _x_half
                + g.pow(2) * self.base_gaussian.multiply_covariance(p_half2, C) * dt / 2
            )  # A
            p = p_half2 + score_func(t, C, _x) * dt / 2  # B

            return _x, p

        def ode_step(t, _x):
            score = score_func(t, C, _x)
            score_transformed = self.base_gaussian.multiply_covariance(score, C)
            _x = _x + 0.5 * (_x + score_transformed) * g.pow(2) * dt
            return _x

        for i in range(n_equil):
            _x, p = baoab_step(_x, p, t)

        if ode_boost:
            _x = ode_step(t, _x)

        return _x, p

    @torch.no_grad()
    def sample_sde(
        self,
        X0_func: Callable,
        C: torch.LongTensor,
        X_init: Optional[torch.Tensor] = None,
        conditioner: Optional[Callable] = None,
        N: int = 100,
        tspan: Tuple[float, float] = (1.0, 0.001),
        inverse_temperature: float = 1.0,
        langevin_factor: float = 0.0,
        langevin_isothermal: bool = True,
        sde_func: str = "reverse_sde",
        integrate_func: str = "euler_maruyama",
        initialize_noise: bool = True,
        remap_time: bool = False,
        remove_drift_translate: bool = False,
        remove_noise_translate: bool = False,
        align_X0: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the SDE using a numerical integration scheme.

        This function samples from the stochastic differential equation (SDE) defined
        by the model using a numerical integration scheme such as Euler-Maruyama or
        huen. The SDE can be either in the forward or reverse direction. The function
        also supports optional conditioning on external variables and adding Langevin
        noise to the SDE dynamics.

        Args:
            X0_func (Callable): A denoising function that maps `(X, C, t)` to `X0`.
            C (torch.LongTensor): Conditioner tensor with shape `(num_batch,
                num_residues)`.
            X_init (torch.Tensor, optional): Initial state tensor with shape `(num_batch
                , num_residues, 4 ,3)` or None.
                If None, a zero tensor will be used as the initial state.
            conditioner (Callable, optional): A function that transforms X, C, U, t.
                If None, no conditioning will be applied.
            N (int): Number of integration steps.
            tspan (Tuple[float,float]): Time span for integration.
            inverse_temperature (float): Inverse temperature parameter for SDE.
            langevin_factor (float): Langevin factor for adding noise to SDE.
            langevin_isothermal (bool): Whether to use isothermal or adiabatic Langevin
                 dynamics.
            sde_func (str): Which SDE function to use ('reverse_sde', 'langevin' or 'ode').
            integrate_func (str): Which integration function to use ('euler_maruyama'
                 or 'heun').
            initialize_noise (bool): Whether to initialize the state with noise.
            remap_time (bool): Whether to remap the time grid according to the noise
                 schedule.
            remove_drift_translate (bool): Whether to remove the net translational
                 component from the drift term.
            remove_noise_translate (bool): Whether to remove the net translational
                 component from the noise term.
            align_X0 (bool): Whether to Kabsch align X0 with X before computing SDE terms.

        Returns:
            outputs (Dict[str, torch.Tensor]): A dictionary of output tensors with the
            following keys:
                - 'C': The conditioned tensor with shape `(num_batch,num_residues)`.
                - 'X_sample': The final sampled state tensor with shape `(num_batch,
                    num_residues ,4 ,3)`.
                - 'X_trajectory': A list of state tensors along the trajectory with
                    shape `(num_batch,num_residues ,4 ,3)` each.
                - 'Xhat_trajectory': A list of transformed state tensors along the
                    trajectory with shape `(num_batch,num_residues ,4 ,3)` each.
                - 'Xunc_trajectory': A list of unconstrained state tensors along the
                    trajectory with shape `(num_batch,num_residues ,4 ,3)` each.
        """

        # Setup SDE integration
        integrate_func = self.integrate_funcs[integrate_func]
        sde_func = self.sde_funcs[sde_func]
        T_grid = (
            self.noise_schedule.linear_logsnr_grid(N=N, tspan=tspan).to(C.device)
            if remap_time
            else torch.linspace(tspan[0], tspan[1], N + 1).to(C.device)
        )

        # Intercept the X0 function for tracking Xt and Xhat
        Xhat_trajectory = []
        Xt_trajectory = []
        U_trajectory = []

        def _X0_func(_X, _C, t):
            _X0 = X0_func(_X, _C, t)
            Xt_trajectory.append(_X.detach())
            Xhat_trajectory.append(_X0.detach())
            return _X0

        def sdefun(_t, _X):
            f, gZ = sde_func(
                _X,
                _X0_func,
                C,
                _t,
                conditioner=conditioner,
                inverse_temperature=inverse_temperature,
                langevin_factor=langevin_factor,
                langevin_isothermal=langevin_isothermal,
                align_X0=align_X0,
            )
            # Remove net translational component
            if remove_drift_translate:
                f = backbone.center_X(f, C)
            if remove_noise_translate:
                gZ = backbone.center_X(gZ, C)
            return f, gZ

        # Initialization
        if initialize_noise and X_init is not None:
            X_init = self.forward(X_init, C, t=tspan[0]).detach()
        elif X_init is None:
            X_init = torch.zeros(list(C.shape) + [4, 3], device=C.device)
            X_init = self.forward(X_init, C, t=tspan[0]).detach()

        # Determine output shape via a test forward pass
        if conditioner:
            with torch.enable_grad():
                X_init_test = X_init.clone()
                X_init_test.requires_grad = True
                S_test = torch.zeros(C.shape, device=X_init.device).long()
                O_test = F.one_hot(S_test, len(AA20)).float()
                U_test = 0.0
                t_test = torch.tensor([0.0], device=X_init.device)
                _, Ct, _, _, _ = conditioner(X_init_test, C, O_test, U_test, t_test)
        else:
            Ct = C

        # Integrate
        X_trajectory = integrate_func(sdefun, X_init, tspan, N=N, T_grid=T_grid)

        # Return constrained coordinates
        outputs = {
            "C": Ct,
            "X_sample": Xt_trajectory[-1],
            "X_trajectory": [Xt_trajectory[-1]] + Xt_trajectory,
            "Xhat_trajectory": Xhat_trajectory,
            "Xunc_trajectory": X_trajectory,
        }
        return outputs

    @torch.no_grad()
    def estimate_pseudoelbo_X(
        self,
        X0_func,
        X,
        C,
        num_samples=50,
        deterministic_seed=0,
        return_elbo_t=False,
        noise=True,
    ):
        with torch.random.fork_rng():
            torch.random.manual_seed(deterministic_seed)

            mask = (C > 0).float()
            mask_atoms = mask.reshape(list(mask.shape) + [1, 1]).expand([-1, -1, 4, 1])

            elbo = []
            T = np.linspace(1e-4, 1.0, num_samples)
            for t in tqdm(T.tolist()):
                X_noise = self.forward(X, C, t=t) if noise else X
                X_denoise = X0_func(X_noise, C, t)

                elbo_t = -self.noise_schedule.SNR_derivative(t).to(X.device) * (
                    ((mask_atoms * (X_denoise - X) / 10.0) ** 2).sum([1, 2, 3])
                    / mask_atoms.sum([1, 2, 3])
                )
                elbo.append(elbo_t)
            elbo = torch.stack(elbo, 0)
            if not return_elbo_t:
                elbo = elbo.mean(0)
        return elbo

    def _score_direct(
        self, Xt, X0_func, C, t, align_X0=True,
    ):
        X0 = X0_func(Xt, C, t)

        """Compute the score function directly. (Sometimes numerically unstable)"""

        alpha = self.noise_schedule.alpha(t).to(Xt.device)
        sigma = self.noise_schedule.sigma(t).to(Xt.device)

        # Impute sensibly behaved values in masked regions for numerical stability
        # X0 = backbone.impute_masked_X(X0, C)
        Xt = backbone.impute_masked_X(Xt, C)

        if align_X0:
            X0, _ = self.loss_rmsd.align(X0, Xt, C, align_unmasked=True)

        # Compute mean
        X_mu = self._mean(X0, C, alpha)
        X_mu = backbone.impute_masked_X(X_mu, C)
        dX = Xt - X_mu

        Ci_dX = self.base_gaussian.multiply_inverse_covariance(dX, C)
        score = -Ci_dX / sigma.pow(2)[:, None, None, None]

        # Mask
        score = score.masked_fill((C <= 0)[..., None, None], 0.0)

        return score

    def estimate_logp(
        self,
        X0_func: Callable,
        X_sample: torch.Tensor,
        C: torch.LongTensor,
        N: int,
        return_trace_t: bool = False,
    ):
        """Estimate the model logP for given protein backboones
            (num_batch, num_residues, 4, 3) by the Continuous Normalizing Flow formalism

            Reference:
                https://arxiv.org/abs/1810.01367
                https://arxiv.org/abs/1806.07366

        Args:
            X0_func (Callable): A function that returns the initial protein backboone
                 (num) features given a condition.
            X_sample (torch.Tensor): A tensor of protein backboone (num) features with
            shape
                `(batch_size, num_residues, 4, 3)`.
            C (torch.Tensor): A tensor of condition features with shape `(batch_size,
                 num_residues)`.
            N (int, optional): number of ode integration steps
            return_trace_t (bool, optional): A flag that indicates whether to return the
            log |df / dx| for each time step for the integrated log Jacobian trance.
              Default is False.

        Returns:
            elbo (torch.Tensor): A tensor of logP value
            if return_elbo_t is False, or `(N)` if return_elbo_t
            is True.
        """

        def divergence(fn, x, t):
            """Calculate Divergance with Stochastic Trace Estimator"""
            vec_eps = torch.randn_like(x)
            fn_out, eps_J_prod = torch.autograd.functional.vjp(
                fn, (t, x), vec_eps, create_graph=False
            )
            eps_J_eps = (
                (eps_J_prod[1] * vec_eps).reshape(x.shape[0], -1).sum(-1).unsqueeze(-1)
            )
            return fn_out, eps_J_eps

        def flow_gradient(
            X, X0_func, C, t,
        ):
            """Compute the time gradient from the probability flow ODE."""

            _, _, beta, g, _, _ = self._schedule_coefficients(t)
            score = self._score_direct(X, X0_func, C, t)
            dXdt = (-1 / 2) * beta * X - 0.5 * g.pow(2) * score

            return dXdt

        def odefun(_t, _X):
            _t = _t.detach()
            f = flow_gradient(_X, X0_func, C, _t,)
            return f

        # foward integration to noise
        X_sample = backbone.center_X(X_sample, C)
        X_sample = backbone.impute_masked_X(X_sample, C)
        C = C.abs()

        out = self.sample_sde(
            X0_func=X0_func,
            C=C,
            X_init=X_sample,
            N=N,
            sde_func="ode",
            tspan=(0, 1.0),
            inverse_temperature=1.0,
            langevin_factor=0.0,
            initialize_noise=False,
            align_X0=False,
        )

        X_flow = out["X_trajectory"][1:]

        # get ode function
        ddlogp = []

        for i, t in enumerate(tqdm(torch.linspace(1e-2, 1.0, len(X_flow)))):
            with torch.enable_grad():
                dlogP = divergence(odefun, X_flow[i], t[None].to(C.device))[1]
            ddlogp.append(dlogP.item())

        logp_x1 = self.base_gaussian.log_prob(X_flow[-1], C).item()

        if return_trace_t:
            return np.array(ddlogp) / ((C > 0).float().sum().item() * 4)
        else:
            return (logp_x1 + np.array(ddlogp).mean()) / (
                (C > 0).float().sum().item() * 4
            )

    @torch.no_grad()
    @validate_XC(all_atom=False)
    def estimate_elbo(
        self,
        X0_func: Callable,
        X: torch.Tensor,
        C: torch.LongTensor,
        num_samples: int = 50,
        deterministic_seed: int = 0,
        return_elbo_t: bool = False,
        grad_logprob_Y_func: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Estimate the evidence lower bound (ELBO) for given protein backboones
            (num_batch, num_residues, 4, 3) and condition.

        Args:
            X0_func (Callable): A function that returns the initial protein backboone
                 (num) features given a condition.
            X (torch.Tensor): A tensor of protein backboone (num) features with shape
                `(batch_size, num_residues, 4, 3)`.
            C (torch.Tensor): A tensor of condition features with shape `(batch_size,
                 num_residues)`.
            num_samples (int, optional): The number of time steps to sample for
                estimating the ELBO. Default is 50.
            deterministic_seed (int, optional): The seed for generating random noise.
                 Default is 0.
            return_elbo_t (bool, optional): A flag that indicates whether to return the
            ELBO for each time step or the average ELBO. Default is False.
            grad_logprob_Y_func (Optional[Callable], optional): A function that returns
            the gradient of the log probability of the observed protein backboone (num)
            given a time step and a noisy image. Default is None.

        Returns:
            elbo (torch.Tensor): A tensor of ELBO values with shape `(batch_size,)`
            if return_elbo_t is False, or `(num_samples, batch_size)` if return_elbo_t
            is True.
        """
        X = backbone.impute_masked_X(X, C)

        with torch.random.fork_rng():
            torch.random.manual_seed(deterministic_seed)
            mask = (C > 0).float()
            mask_atoms = mask.reshape(list(mask.shape) + [1, 1]).expand([-1, -1, 4, 1])

            elbo = []
            T = np.linspace(1e-4, 1.0, num_samples)
            for t in tqdm(T.tolist()):
                X_noise = self.forward(X, C, t=t)
                X_denoise = X0_func(X_noise, C, t)

                # Adjust X-hat estimate with aux-grad
                if grad_logprob_Y_func is not None:
                    with torch.random.fork_rng():
                        grad = grad_logprob_Y_func(t, X_noise)
                        sigma_square = (
                            self.noise_schedule.sigma(t).square().to(X.device)
                        )
                        dXhat = sigma_square * self.base_gaussian.multiply_covariance(
                            grad, C
                        )
                        dXhat = backbone.center_X(dXhat, C)
                        X_denoise = X_denoise + dXhat

                elbo_t, _ = self.elbo(X_denoise, X, C, t)

                elbo.append(elbo_t)

            elbo_t = torch.stack(elbo, 0)

        if return_elbo_t:
            return elbo_t
        else:
            return elbo_t.mean(0)

    def conditional_X0(
        self, X0: torch.Tensor, score: torch.Tensor, C: torch.tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Use Bayes theorem and Tweedie formula to obtain a conditional X0 given
        prior X0 and a conditional score \nabla_x p( y | x)
        X0 <- X0 + \frac{sigma_t**2}{alpha_t} \Sigma score
        Args:
            X0 (torch.Tensor): backbone coordinates of size (batch, num_residues, 4, 3)
            score (torch.Tensor): of size (batch, num_residues, 4, 3)
            C (torch.Tensor): of size (batch, num_residues)
            t (torch.Tensor): of size (batch,)

        Returns:
            X0 (torch.Tensor): updated conditional X0 of size (batch, num_residues, 4, 3)
        """
        alpha, sigma, _, _, _, _ = self._schedule_coefficients(t)
        X_update = sigma.pow(2).div(alpha)[
            ..., None
        ] * self.base_gaussian.multiply_covariance(score, C)
        return X0 + X_update

    def _mean(self, X, C, alpha):
        """Build the diffusion kernel mean given alpha"""
        # Compute the MVN mean
        X_mu = backbone.scale_around_mean(X, C, alpha)
        return X_mu

    def _X_to_Z(self, X_sample, X, C, alpha, sigma):
        """Convert from output space to standardized space"""

        # Impute missing data with conditional means
        X = backbone.impute_masked_X(X, C)
        X_sample = backbone.impute_masked_X(X_sample, C)

        # sigma = self.noise_schedule.sigma(t).to(X.device)

        # Step 4. [Inverse] Add mean
        X_mu = self._mean(X, C, alpha)
        X_mu = backbone.impute_masked_X(X_mu, C)
        X_noise = (X_sample - X_mu).reshape(X.shape[0], -1, 3)

        # Step 3. [Inverse] Scale noise by sigma
        X_noise = X_noise / sigma[:, None, None]

        # Step 1 & 2. Multiply Z by inverse square root of covariance
        Z = self.base_gaussian._multiply_R_inverse(X_noise, C)

        return Z

    def _Z_to_X(self, Z, X, C, alpha, sigma):
        """Convert from standardized space to output space"""

        # Step 1 & 2. Multiply Z by square root of covariance
        dX = self.base_gaussian._multiply_R(Z, C)

        # Step 3. Scale noise by alpha
        dX = sigma[:, None, None, None] * dX.reshape(X.shape)

        # Step 4. Add mean
        X_mu = self._mean(X, C, alpha)
        X_sample = X_mu + dX

        return X_sample

    def sample_conditional(
        self, X: torch.Tensor, C: torch.LongTensor, t: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from the forward process q(x_{t} | x_{s}) for t > s.
        See appendix A.1 in [https://arxiv.org/pdf/2107.00630.pdf]. `forward` does this for s = 0.
        Args:
            X (torch.Tensor): Input coordinates with shape `(batch_size, num_residues,
                4, 3)` at time `t0`.
            C (torch.Tensor): Chain tensor with shape `(batch_size, num_residues)`.
            t (torch.Tensor): Time index with shape `(batch_size,)`.
            s (torch.Tensor): Time index with shape `(batch_size,)`.

        Returns:
            X_sample (torch.Tensor): Sampled coordinates from the forward diffusion
                marginals with shape `(batch_size, num_residues, 4, 3)`.
        """
        assert (t > s).all()
        X = backbone.impute_masked_X(X, C)
        # Do we need this?
        X = backbone.center_X(X, C)
        alpha_ts = self.noise_schedule.alpha(t) / self.noise_schedule.alpha(s)
        sigma_ts = (
            self.noise_schedule.sigma(t).pow(2)
            - alpha_ts.pow(2) * self.noise_schedule.sigma(s).pow(2)
        ).sqrt()

        X_sample = alpha_ts * X + sigma_ts * self.base_gaussian.sample(C)
        # Do we need this?
        X_sample = backbone.center_X(X_sample - X, C) + X
        return X_sample

    @validate_XC(all_atom=False)
    def forward(
        self, X: torch.Tensor, C: torch.LongTensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the forwards diffusion marginals at time t

        Inputs:
            X (torch.Tensor): Input coordinates with shape `(batch_size, num_residues,
                4, 3)`.
            C (torch.LongTensor): Chain tensor with shape `(batch_size, num_residues)`.
            t (torch.Tensor, optional): Time index with shape `(batch_size,)`. If not
                given, a random time index will be sampled. Defaults to None.

        Outputs:
            X_sample (torch.Tensor): Sampled coordinates from the forward diffusion
                marginals with shape `(batch_size, num_residues, 4, 3)`.
            t (torch.Tensor, optional): Time index with shape `(batch_size,)`. Only
                returned if t is not given as input.
        """

        # Draw a sample from the prior
        X_prior = self.base_gaussian.sample(C)

        # Sample time if not given
        t_input = t
        t = self.sample_t(C, t)

        alpha = self.noise_schedule.alpha(t)[:, None, None, None].to(X.device)
        sigma = self.noise_schedule.sigma(t)[:, None, None, None].to(X.device)

        X_sample = alpha * X + sigma * X_prior
        X_sample = backbone.center_X(X_sample - X, C) + X

        if t_input is None:
            return X_sample, t
        else:
            return X_sample


class ReconstructionLosses(nn.Module):
    """Compute diffusion reconstruction losses for protein backbones.

    Args:
        diffusion (DiffusionChainCov): Diffusion object parameterizing a
            forwards diffusion over protein backbones.
        loss_scale (float): Length scale parameter used for setting loss error
            scaling in units of Angstroms. Default is 10, which corresponds to
            using units of nanometers.
        rmsd_method (str): Method used for computing RMSD superpositions. Can
            be "symeig" (default) or "power" for power iteration.

    Inputs:
        X0_pred (torch.Tensor): Denoised coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        X (torch.Tensor): Unperturbed coordinates with shape
            `(num_batch, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        t (torch.Tensor): Diffusion time with shape `(batch_size,)`.
            Should be on [0,1].

    Outputs:
        losses (dict): Dictionary of reconstructions computed across different
            metrics. Metrics prefixed with `batch_` will be batch-averaged scalars
            while other metrics should be per batch member with shape
            `(num_batch, ...)`.
    """

    def __init__(
        self,
        diffusion: DiffusionChainCov,
        loss_scale: float = 10.0,
        rmsd_method: str = "symeig",
    ):
        super().__init__()
        self.noise_perturb = diffusion
        self.loss_scale = loss_scale
        self._loss_eps = 1e-5

        # Auxiliary losses
        self.loss_rmsd = rmsd.BackboneRMSD(method=rmsd_method)
        self.loss_fragment = rmsd.LossFragmentRMSD(method=rmsd_method)
        self.loss_fragment_pair = rmsd.LossFragmentPairRMSD(method=rmsd_method)
        self.loss_neighborhood = rmsd.LossNeighborhoodRMSD(method=rmsd_method)
        self.loss_hbond = hbonds.LossBackboneHBonds()
        self.loss_distance = backbone.LossBackboneResidueDistance()

        self.loss_functions = {
            "elbo": self._loss_elbo,
            "rmsd": self._loss_rmsd,
            "pseudoelbo": self._loss_pseudoelbo,
            "fragment": self._loss_fragment,
            "pair": self._loss_pair,
            "neighborhood": self._loss_neighborhood,
            "distance": self._loss_distance,
            "hbonds": self._loss_hbonds,
        }

    def _batch_average(self, loss, C):
        weights = (C > 0).float().sum(-1)
        return (weights * loss).sum() / (weights.sum() + self._loss_eps)

    def _loss_elbo(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        losses["elbo"], losses["batch_elbo"] = self.noise_perturb.elbo(X0_pred, X, C, t)

    def _loss_rmsd(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        _, rmsd_denoise = self.loss_rmsd.align(X, X0_pred, C)
        _, rmsd_noise = self.loss_rmsd.align(X, X_t_2, C)
        rmsd_ratio_per_item = w * rmsd_denoise / (rmsd_noise + self._loss_eps)
        global_mse_normalized = (
            w
            * self.loss_scale
            * rmsd_denoise.square()
            / (rmsd_noise.square() + self._loss_eps)
        )
        losses["rmsd_ratio"] = self._batch_average(rmsd_ratio_per_item, C)
        losses["global_mse"] = global_mse_normalized
        losses["batch_global_mse"] = self._batch_average(global_mse_normalized, C)

    def _loss_pseudoelbo(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        # Unaligned residual pseudoELBO
        unaligned_mse = ((X - X0_pred) / self.loss_scale).square().sum(-1).mean(-1)
        losses["elbo_X"], losses["batch_pseudoelbo_X"] = self.noise_perturb.pseudoelbo(
            unaligned_mse, C, t
        )

    def _loss_fragment(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        # Aligned Fragment MSE loss
        mask = (C > 0).float()
        rmsd_fragment = self.loss_fragment(X0_pred, X, C)
        rmsd_fragment_noise = self.loss_fragment(X_t_2, X, C)
        fragment_mse_normalized = (
            self.loss_scale
            * w
            * (
                (mask * rmsd_fragment.square()).sum(1)
                / ((mask * rmsd_fragment_noise.square()).sum(1) + self._loss_eps)
            )
        )
        losses["fragment_mse"] = fragment_mse_normalized
        losses["batch_fragment_mse"] = self._batch_average(fragment_mse_normalized, C)

    def _loss_pair(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        # Aligned Pair MSE loss
        rmsd_pair, mask_ij_pair = self.loss_fragment_pair(X0_pred, X, C)
        rmsd_pair_noise, mask_ij_pair = self.loss_fragment_pair(X_t_2, X, C)
        pair_mse_normalized = (
            self.loss_scale
            * w
            * (
                (mask_ij_pair * rmsd_pair.square()).sum([1, 2])
                / (
                    (mask_ij_pair * rmsd_pair_noise.square()).sum([1, 2])
                    + self._loss_eps
                )
            )
        )
        losses["pair_mse"] = pair_mse_normalized
        losses["batch_pair_mse"] = self._batch_average(pair_mse_normalized, C)

    def _loss_neighborhood(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        # Neighborhood MSE
        rmsd_neighborhood, mask = self.loss_neighborhood(X0_pred, X, C)
        rmsd_neighborhood_noise, mask = self.loss_neighborhood(X_t_2, X, C)
        neighborhood_mse_normalized = (
            self.loss_scale
            * w
            * (
                (mask * rmsd_neighborhood.square()).sum(1)
                / ((mask * rmsd_neighborhood_noise.square()).sum(1) + self._loss_eps)
            )
        )
        losses["neighborhood_mse"] = neighborhood_mse_normalized
        losses["batch_neighborhood_mse"] = self._batch_average(
            neighborhood_mse_normalized, C
        )

    def _loss_distance(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        # Distance MSE
        mask = (C > 0).float()
        distance_mse = self.loss_distance(X0_pred, X, C)
        distance_mse_noise = self.loss_distance(X_t_2, X, C)
        distance_mse_normalized = self.loss_scale * (
            w
            * (mask * distance_mse).sum(1)
            / ((mask * distance_mse_noise).sum(1) + self._loss_eps)
        )
        losses["distance_mse"] = distance_mse_normalized
        losses["batch_distance_mse"] = self._batch_average(distance_mse_normalized, C)

    def _loss_hbonds(self, losses, X0_pred, X, C, t, w=None, X_t_2=None):
        # HBond recovery
        outs = self.loss_hbond(X0_pred, X, C)
        hb_local, hb_nonlocal, error_co = [w * o for o in outs]

        losses["batch_hb_local"] = self._batch_average(hb_local, C)
        losses["hb_local"] = hb_local
        losses["batch_hb_nonlocal"] = self._batch_average(hb_nonlocal, C)
        losses["hb_nonlocal"] = hb_nonlocal
        losses["batch_hb_contact_order"] = self._batch_average(error_co, C)

    @torch.no_grad()
    @validate_XC(all_atom=False)
    def estimate_metrics(
        self,
        X0_func: Callable,
        X: torch.Tensor,
        C: torch.LongTensor,
        num_samples: int = 50,
        deterministic_seed: int = 0,
        use_noise: bool = True,
        return_samples: bool = False,
        tspan: Tuple[float] = (1e-4, 1.0),
    ):
        """Estimate time-averaged reconstruction losses of protein backbones.

        Args:
            X0_func (Callable): A denoising function that maps `(X, C, t)` to `X0`.
            X (torch.Tensor): A tensor of protein backboone (num) features with shape
                `(batch_size, num_residues, 4, 3)`.
            C (torch.Tensor): A tensor of condition features with shape `(batch_size,
                num_residues)`.
            num_samples (int, optional): The number of time steps to sample for
            estimating the ELBO. Default is 50.
            use_noise (bool): If True, add noise to each structure before denoising.
                Default is True. When False this can be used for estimating if
                if structures are fixed points of the denoiser across time.
            deterministic_seed (int, optional): The seed for generating random noise.
                Default is 0.
            return_samples (bool): If True, include intermediate sampled
                values for each metric. Default is false.
            tspan (tuple[float]): Tuple of floats indicating the diffusion
                times between which to integrate.

        Returns:
            metrics (dict): A dictionary of reconstruction metrics averaged over
                time.
            metrics_samples (dict, optional): A dictionary of in metrics
                averaged over time.
        """
        #
        X = backbone.impute_masked_X(X, C)
        with torch.random.fork_rng():
            torch.random.manual_seed(deterministic_seed)
            T = np.linspace(1e-4, 1.0, num_samples)
            losses = []
            for t in tqdm(T.tolist(), desc="Integrating diffusion metrics"):
                X_noise = self.noise_perturb(X, C, t=t) if use_noise else X
                X_denoise = X0_func(X_noise, C, t)
                losses_t = self.forward(X_denoise, X, C, t)

                # Discard batch estimated objects
                losses_t = {
                    k: v
                    for k, v in losses_t.items()
                    if not k.startswith("batch_") and k != "rmsd_ratio"
                }
                losses.append(losses_t)

            # Transpose list of dicts to a dict of lists
            metrics_samples = {k: [d[k] for d in losses] for k in losses[0].keys()}

            # Average final metrics across time
            metrics = {
                k: torch.stack(v, 0).mean(0)
                for k, v in metrics_samples.items()
                if isinstance(v[0], torch.Tensor)
            }
        if return_samples:
            return metrics, metrics_samples
        else:
            return metrics

    @validate_XC()
    def forward(
        self,
        X0_pred: torch.Tensor,
        X: torch.Tensor,
        C: torch.LongTensor,
        t: torch.Tensor,
    ):
        # Collect all losses and tensors for metric tracking
        losses = {"t": t, "X": X, "X0_pred": X0_pred}
        X_t_2 = self.noise_perturb(X, C, t=t)

        # Per complex weights
        ssnr = self.noise_perturb.noise_schedule.SSNR(t).to(X.device)
        prob_ssnr = self.noise_perturb.noise_schedule.prob_SSNR(ssnr)
        importance_weights = 1 / prob_ssnr

        for _loss in self.loss_functions.values():
            _loss(losses, X0_pred, X, C, t, w=importance_weights, X_t_2=X_t_2)
        return losses


def _debug_viz_gradients(
    pml_file, X_list, dX_list, C, S, arrow_length=2.0, name="gradient", color="red"
):
    """ """
    lines = [
        "from pymol.cgo import *",
        "from pymol import cmd",
        f'color_1 = list(pymol.cmd.get_color_tuple("{color}"))',
        'color_2 = list(pymol.cmd.get_color_tuple("blue"))',
    ]

    with open(pml_file, "w") as f:
        for model_ix, X in enumerate(X_list):
            print(model_ix)
            lines = lines + ["obj_1 = []"]

            dX = dX_list[model_ix]
            scale = dX.norm(dim=-1).mean().item()
            X_i = X
            X_j = X + arrow_length * dX / scale

            for a_ix in range(4):
                for i in range(X.size(1)):
                    x_i = X_i[0, i, a_ix, :].tolist()
                    x_j = X_j[0, i, a_ix, :].tolist()
                    lines = lines + [
                        f"obj_1 = obj_1 + [CYLINDER] + {x_i} + {x_j} + [0.15]"
                        " + color_1 + color_1"
                    ]
            lines = lines + [f'cmd.load_cgo(obj_1, "{name}", {model_ix+1})']
            f.write("\n" + "\n".join(lines))
            lines = []


def _debug_viz_XZC(X, Z, C, rgb=True):
    from matplotlib import pyplot as plt

    if len(X.shape) > 3:
        X = X.reshape(X.shape[0], -1, 3)
    if len(Z.shape) > 3:
        Z = Z.reshape(Z.shape[0], -1, 3)
    if C.shape[1] != X.shape[1]:
        C_expand = C.unsqueeze(-1).expand(-1, -1, 4)
        C = C_expand.reshape(C.shape[0], -1)

    # C_mask = expand_chain_map(torch.abs(C))
    # X_expand = torch.einsum('nix,nic->nicx', X, C_mask)
    # plt.plot(X_expand[0,:,:,0].data.numpy())
    N = X.shape[1]
    Ymax = torch.max(X[0, :, 0]).item()
    plt.figure(figsize=[12, 4])
    plt.subplot(2, 1, 1)

    plt.bar(
        np.arange(0, N),
        (C[0, :].data.numpy() < 0) * Ymax,
        width=1.0,
        edgecolor=None,
        color="lightgrey",
    )
    if rgb:
        plt.plot(X[0, :, 0].data.numpy(), "r", linewidth=0.5)
        plt.plot(X[0, :, 1].data.numpy(), "g", linewidth=0.5)
        plt.plot(X[0, :, 2].data.numpy(), "b", linewidth=0.5)
        plt.xlim([0, N])
        plt.grid()
        plt.title("X")
        plt.xticks([])
        plt.subplot(2, 1, 2)
        plt.plot(Z[0, :, 0].data.numpy(), "r", linewidth=0.5)
        plt.plot(Z[0, :, 1].data.numpy(), "g", linewidth=0.5)
        plt.plot(Z[0, :, 2].data.numpy(), "b", linewidth=0.5)
        plt.plot(C[0, :].data.numpy(), "orange")
        plt.xlim([0, N])
        plt.grid()
        plt.title("RInverse @ [X]")
        plt.xticks([])
        plt.savefig("xzc.pdf")
    else:
        plt.plot(X[0, :, 0].data.numpy(), "k", linewidth=0.5)
        plt.xlim([0, N])
        plt.grid()
        plt.title("X")
        plt.xticks([])
        plt.subplot(2, 1, 2)
        plt.plot(Z[0, :, 0].data.numpy(), "k", linewidth=0.5)
        plt.plot(C[0, :].data.numpy(), "orange")
        plt.xlim([0, N])
        plt.grid()
        plt.title("Inverse[X]")
        plt.xticks([])
        plt.savefig("xzc.pdf")
    exit()
