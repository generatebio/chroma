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

"""Layers for multivariate normal models of protein structure.

This module contains pytorch layers for perturbing protein structure with noise,
which can be useful both for data augmentation, benchmarking, or denoising based
training.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from chroma.layers import conv
from chroma.layers.structure import backbone


class BackboneMVNGlobular(torch.nn.Module):
    """
    Gaussian model for protein backbones.
    """

    def __init__(
        self,
        covariance_model="brownian",
        complex_scaling=False,
        sigma_translation=1.0,
        **kwargs,
    ):
        super().__init__()

        # These constant was derived from fitting uniform phi,psi chains
        self._scale = 1.5587407701549267

        # These parameterize the scaling law, *per xyz dimension*
        # Rg = Rg0 * N_atoms ^ nu
        self._nu = 2.0 / 5.0
        self._rg_0_1D = 2.0 / 3.0

        # Exact solution for Rg0, agrees with above to 2 decimals.
        # We divide the literature prefactor for a per-residue
        # scaling law (John J Tanner 2016) by two terms:
        #   1. a conversion factor from residues to atoms (4^nu)
        #   2. the sqrt(3) to account for isotropic dimensional (xyz) Rg contributions
        # self._rg_0_1D = 2.0 / (4 ** self._nu * np.sqrt(3))

        self.covariance_model = covariance_model
        self.complex_scaling = complex_scaling
        self.sigma_translation = sigma_translation

    def _atomic_mean(self, X_flat, mask):
        """Compute the mean across all 4 atom types by mask expansion"""
        mask_expand = mask.unsqueeze(-1).expand(-1, -1, 4)
        mask_atomic = mask_expand.reshape(mask.shape[0], -1).unsqueeze(-1)
        X_mean = torch.sum(mask_atomic * X_flat, 1, keepdims=True) / (
            torch.sum(mask_atomic, 1, keepdims=True)
        )
        return X_mean, mask_atomic

    def _C_atomic(self, C):
        # Expand chain map into atomic level masking
        C_expand = C.unsqueeze(-1).expand(-1, -1, 4)
        C_atomic = C_expand.reshape(C.shape[0], -1)
        return C_atomic

    def _globular_parameters(self, C_mask_all, translation_inflation=None):
        """Compute parameters for enforcing Rg scaling"""
        # Rg scaling constants
        nu = self._nu
        a = self._scale
        r = self._rg_0_1D

        # C_mask_all is ()
        # Monomer and complex sizes (batch, {chains})
        C_mask = C_mask_all.squeeze(-1)
        N_per_chain = C_mask.sum(1)
        N_per_complex = C_mask.sum([1, 2])

        # Compute expected Rg^2 values per complex
        Rg2_complex = (r ** 2) * N_per_complex ** (2.0 * nu)
        Rg2_chain = (r ** 2) * N_per_chain ** (2.0 * nu)

        # Compute OU process parameters
        N_per_chain = torch.clip(N_per_chain, 1, 1e6)

        # Decay parameter B is related to global spring coefficient
        # as k = (1-B)^2
        B = (3.0 / N_per_chain) + N_per_chain ** (-nu) * torch.sqrt(
            N_per_chain ** (2 * (nu - 1)) * (N_per_chain ** 2 + 9) - (a / r) ** 2
        )
        B = torch.clip(B, 1e-4, 1.0 - 1e-4)

        # OU process equilibrium standard deviation warm-starts process
        x_init_std = torch.sqrt(1.0 / (1.0 - B ** 2))

        # Compute size-weighted average Rg^2 per chain
        Rg2_chain_avg = (N_per_chain * Rg2_chain).sum(1) / (N_per_chain.sum(1) + 1e-5)
        Rg2_centers_of_mass = torch.clip(Rg2_complex - Rg2_chain_avg, min=1)
        Rg_centers_of_mass = torch.sqrt(Rg2_centers_of_mass)

        # Mean scaling parameter deflates equilibrium variance to unity and
        # optionally re-inflates the per center of mass variance to implement
        # complex scaling
        if translation_inflation is not None:
            # This argument overrides default per-chain translational scaling
            # to [translation_inflation * Complex Rg]
            marginal_COM_std = translation_inflation * Rg2_complex.sqrt()
            mean_shift_scale = (x_init_std - marginal_COM_std[..., None]) / (x_init_std)
        elif self.complex_scaling:
            N_chains_per_complex = (C_mask.sum(1) > 0).sum(1)
            # Correct for the fact that we are sampling chains IID (not
            # centered) but want to control centered Rg
            std_correction = torch.sqrt(
                N_chains_per_complex / (N_chains_per_complex - 1).clamp(min=1)
            )
            marginal_COM_std = std_correction * Rg_centers_of_mass
            mean_shift_scale = (x_init_std - marginal_COM_std[..., None]) / (x_init_std)
        else:
            mean_shift_scale = (x_init_std - 1.0) / (x_init_std)

        return B[..., None], x_init_std[..., None], mean_shift_scale[..., None]

    def _expand_masks(self, C):
        C_atomic = self._C_atomic(C)
        C_mask_all = backbone.expand_chain_map(torch.abs(C_atomic))[..., None]
        C_mask_present = backbone.expand_chain_map(C_atomic)[..., None]
        return C_mask_all, C_mask_present

    def _expand_per_chain(self, Z, C):
        """Build augmented [num_batch, 4*num_residues, num_chains, 3] system"""
        # Build masks and augmented [B,4N,C,3] system
        C_mask_all, C_mask_present = self._expand_masks(C)
        Z_expand = C_mask_all * Z[..., None, :]
        return C_mask_all, C_mask_present, Z_expand

    def _shift_means(self, X_expand, C_mask_mean, C_mask_apply, scale, shift=None):
        """Inflate or deflate per-chain means by a scale factor."""
        X_chain_mean = (C_mask_mean * X_expand).sum(1, keepdims=True) / (
            C_mask_mean.sum(1, keepdims=True) + 1e-5
        )
        shift = shift if shift is not None else 0
        shift = shift + scale * X_chain_mean
        X_expand = C_mask_apply * (X_expand + shift)
        return X_expand

    def _translate_by_x1(self, X_expand, C_mask, scale_mean, scale_x1):
        """Shift mean to mean <- mean + scale_mean * mean + scale_x1 * x1"""
        X_1 = self._gather_chain_init(X_expand, C_mask)
        X_expand = self._shift_means(
            X_expand, C_mask, C_mask, scale=scale_mean, shift=X_1 * scale_x1
        )
        return X_expand

    def _translate_by_x1_transpose(self, X_expand, C_mask, scale_mean, scale_x1):
        """Transpose of _translate_by_x1."""

        # Shift mean (Symmetric under transpose)
        X_chain_sum = (C_mask * X_expand).sum(1, keepdims=True)
        X_chain_mean = X_chain_sum / (C_mask.sum(1, keepdims=True) + 1e-5)
        X_expand = C_mask * (X_expand + scale_mean * X_chain_mean)

        # Update to X_init
        # The transpose of updating all by X_init is updating X_init by all
        first_index = torch.max(C_mask, 1, keepdim=True)[1]
        first_index_expand = first_index.expand(-1, -1, -1, 3)
        X_init = torch.gather(X_expand, 1, first_index_expand)
        X_init_update = X_init + scale_x1 * X_chain_sum
        X_expand = X_expand.scatter(1, first_index_expand, X_init_update)
        return X_expand

    def _gather_chain_init(self, X_expand, C_mask):
        """Extract first coordinates, per chain"""
        first_index = torch.max(C_mask, 1, keepdim=True)[1]
        first_index_expand = first_index.expand(-1, -1, -1, 3)
        X_init = torch.gather(X_expand, 1, first_index_expand)
        return X_init

    def _multiply_R(self, Z, C):
        """Multiply by the square root of the covariance matrix"""
        if Z.dim() == 4:
            Z = Z.reshape(Z.shape[0], -1, 3)

        C_mask_all, C_mask_present, Z_expand = self._expand_per_chain(Z, C)

        if self.covariance_model == "brownian":
            # Step 1. Scaled cumsum along each chain (including missing residues)
            # [B,4N,3] -> [B,4N,C,3]
            R_Z_expand = C_mask_all * torch.cumsum(Z_expand, 1) * self._scale

            # Step 2. Translate by rescaled X_1
            R_Z_expand = self._translate_by_x1(
                R_Z_expand, C_mask_all, scale_mean=-1, scale_x1=self.sigma_translation
            )
        elif self.covariance_model == "globular":
            # Build coefficients per chain as as [B,C,1]
            B, x_init_std, mean_shift_scale = self._globular_parameters(C_mask_all)

            # Step 1. R_init
            # Scale z_1 to have equilibrium variance
            # z_1 will be the position where (1 - mask_{i-1}) = mask_i
            C_mask_prev = F.pad(C_mask_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            mask_init = (1.0 - C_mask_prev) * C_mask_all
            # Inflate z_1 by the equilibrium variance
            Z_expand = (1.0 - mask_init) * Z_expand + mask_init * x_init_std[
                :, None, ...
            ] * Z_expand

            # Step 2. R_sum
            # Apply linear recurrence `x_i = z_i + b * x_{i-1}`
            # Repack all independent signals and coeffs (B,C,3) in 1D
            num_B, num_N, num_C, _ = Z_expand.shape
            # [B,4N,C,3] => [B,C,3,4N] => [BC3, 4N]
            Z_1D = Z_expand.permute([0, 2, 3, 1]).reshape([-1, num_N])
            # [B,C,1] => [BC,1] => [BC,3] => [BC3]
            B_1D = B.reshape([-1, 1]).expand([-1, 3]).reshape([-1])
            R_Z_1D = self._scale * conv.filter1D_linear_decay(Z_1D, B_1D)
            # [BC3,4N] -> [B,C,3,4N] -> [B,4N,C,3]
            R_Z_expand = R_Z_1D.reshape([num_B, num_C, 3, num_N]).permute([0, 3, 1, 2])
            R_Z_expand = C_mask_all * R_Z_expand

            # Step 3. R_center
            # Rescale translational variance
            scale = -mean_shift_scale[:, None, ...]
            R_Z_expand = self._shift_means(
                R_Z_expand, C_mask_all, C_mask_all, scale=scale
            )

        # Collapse out chain dimension
        R_Z = R_Z_expand.sum(2).reshape(Z.shape[0], -1, 4, 3)
        return R_Z

    def _multiply_R_transpose(self, Z, C):
        """Multiply by the square root of the covariance matrix (transpose)"""
        if Z.dim() == 4:
            Z = Z.reshape(Z.shape[0], -1, 3)

        # Inflate chain dimension [B,4N,C,3]
        C_mask_all, C_mask_present, Z_expand = self._expand_per_chain(Z, C)

        if self.covariance_model == "brownian":
            # Step 2. [Transpose of] Translate by rescaled X_1
            Z_expand = self._translate_by_x1_transpose(
                Z_expand, C_mask_all, scale_mean=-1, scale_x1=self.sigma_translation
            )

            # Step 1. [Transpose of] Scaled cumsum along each chain
            Rt_Z_expand = torch.flip(torch.cumsum(torch.flip(Z_expand, [1]), 1), [1])
            Rt_Z_expand = C_mask_all * Rt_Z_expand * self._scale

        elif self.covariance_model == "globular":
            # Build coefficients per chain as as [B,C,1]
            B, x_init_std, mean_shift_scale = self._globular_parameters(C_mask_all)
            Rt_Z_expand = Z_expand

            # Step 3. R_center_transpose = R_center (by symmetry)
            scale = -mean_shift_scale[:, None, ...]
            Rt_Z_expand = self._shift_means(
                Rt_Z_expand, C_mask_all, C_mask_all, scale=scale
            )

            # Step 2. R_sum_transpose = R_sum @ R_flip
            # Apply linear recurrence `x_i = z_i + b * x_{i-1}`
            # Repack all independent signals and coeffs (B,C,3) in 1D
            num_B, num_N, num_C, _ = Rt_Z_expand.shape
            # [B,4N,C,3] => [B,C,3,4N] => [BC3, 4N]
            Z_1D = Rt_Z_expand.permute([0, 2, 3, 1]).reshape([-1, num_N])
            Z_1D_reverse = torch.flip(Z_1D, [1])
            # [B,C,1] => [BC,1] => [BC,3] => [BC3]
            B_1D = B.reshape([-1, 1]).expand([-1, 3]).reshape([-1])
            Rt_Z_1D_reverse = self._scale * conv.filter1D_linear_decay(
                Z_1D_reverse, B_1D
            )
            Rt_Z_1D = torch.flip(Rt_Z_1D_reverse, [1])
            # [BC3,4N] -> [B,C,3,4N] -> [B,4N,C,3]
            Rt_Z_expand = Rt_Z_1D.reshape([num_B, num_C, 3, num_N]).permute(
                [0, 3, 1, 2]
            )
            Rt_Z_expand = C_mask_all * Rt_Z_expand

            # Step 1. R_init_transpose = R_init (by symmetry)
            # Scale z_1 to have equilibrium variance
            # z_1 will be the position where (1 - mask_{i-1}) = mask_i
            C_mask_prev = F.pad(C_mask_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            mask_init = (1.0 - C_mask_prev) * C_mask_all
            # Inflate z_1 by the equilibrium variance
            Rt_Z_expand = (1.0 - mask_init) * Rt_Z_expand + mask_init * x_init_std[
                :, None, ...
            ] * Rt_Z_expand

        # Collapse out chain dimension
        Rt_Z = Rt_Z_expand.sum(2).reshape(Z.shape[0], -1, 4, 3)
        return Rt_Z

    def _multiply_R_inverse(self, X, C):
        """Multiply by the inverse of the square root of the covariance matrix"""
        if X.dim() == 4:
            X = X.reshape(X.shape[0], -1, 3)

        # Inflate chain dimension [B,4N,C,3]
        C_mask_all, C_mask_present, X_expand = self._expand_per_chain(X, C)

        if self.covariance_model == "brownian":
            # Step 2. [Inverse of] Translate by rescaled X_1
            X_expand = self._translate_by_x1(
                X_expand, C_mask_all, scale_mean=1 / self.sigma_translation, scale_x1=-1
            )

            # Step 1. [Inverse of] Scaled cumsum per chain [X_i - X_(i-1)]
            Ri_X_expand = X_expand - F.pad(X_expand[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            Ri_X_expand = C_mask_all * Ri_X_expand / self._scale

        elif self.covariance_model == "globular":
            # Build coefficients per chain as as [B,C,1]
            B, x_init_std, mean_shift_scale = self._globular_parameters(C_mask_all)

            # Step 3. R_center_inverse
            # Rescale translational variance
            mean_shift_scale_inverse = mean_shift_scale / (1 - mean_shift_scale)
            scale = mean_shift_scale_inverse[:, None, ...]
            X_expand = self._shift_means(X_expand, C_mask_all, C_mask_all, scale=scale)

            # Step 2. R_sum_inverse
            # Apply linear recurrence `x_i = z_i + b * x_{i-1}`
            X_prev = F.pad(X_expand[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            Ri_X_expand = (
                C_mask_all * (X_expand - B[:, None, ...] * X_prev) / self._scale
            )

            # Step 1. R_init_inverse
            # Scale z_1 to have equilibrium variance
            # z_1 will be the position where (1 - mask_{i-1}) = mask_i
            C_mask_prev = F.pad(C_mask_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            mask_init = (1.0 - C_mask_prev) * C_mask_all
            Ri_X_expand = (
                1.0 - mask_init
            ) * Ri_X_expand + mask_init * Ri_X_expand / x_init_std[:, None, ...]

        # Collapse out chain dimension
        Ri_X = Ri_X_expand.sum(2).reshape(X.shape[0], -1, 4, 3)
        return Ri_X

    def _multiply_R_inverse_transpose(self, X, C):
        """Multiply by the inverse trasnpose of the square root of the
        covariance matrix
        """
        if X.dim() == 4:
            X = X.reshape(X.shape[0], -1, 3)

        C_mask_all, C_mask_present, X_expand = self._expand_per_chain(X, C)

        if self.covariance_model == "brownian":
            # Step 1. [Inverse transpose of] Scaled cumsum per chain [X_i - X_(i+1)]
            Rit_X_expand = X_expand - F.pad(X_expand[:, 1:, :, :], (0, 0, 0, 0, 0, 1))
            Rit_X_expand = C_mask_all * Rit_X_expand / self._scale

            # Step 2. [Inverse transpose of] Translate by rescaled X_1
            Rit_X_expand = self._translate_by_x1_transpose(
                Rit_X_expand,
                C_mask_all,
                scale_mean=1 / self.sigma_translation,
                scale_x1=-1,
            )
        elif self.covariance_model == "globular":
            # Build coefficients per chain as as [B,C,1]
            B, x_init_std, mean_shift_scale = self._globular_parameters(C_mask_all)
            Rit_X_expand = X_expand

            # Step 1. R_init_inverse_transpose = R_init_inverse (by symmetry)
            # Scale z_1 to have equilibrium variance
            # z_1 will be the position where (1 - mask_{i-1}) = mask_i
            C_mask_prev = F.pad(C_mask_all[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
            mask_init = (1.0 - C_mask_prev) * C_mask_all
            Rit_X_expand = (
                1.0 - mask_init
            ) * Rit_X_expand + mask_init * Rit_X_expand / x_init_std[:, None, ...]

            # Step 2. R_sum_inverse_transpose
            # Apply linear recurrence `x_i = z_i + b * x_{i-1}`
            X_future = F.pad(Rit_X_expand[:, 1:, :, :], (0, 0, 0, 0, 0, 1))
            Rit_X_expand = (
                C_mask_all * (Rit_X_expand - B[:, None, ...] * X_future) / self._scale
            )

            # Step 3. R_center_inverse_transpose = R_center_inverse (by symmetry)
            # Rescale translational variance
            mean_shift_scale_inverse = mean_shift_scale / (1 - mean_shift_scale)
            scale = mean_shift_scale_inverse[:, None, ...]
            Rit_X_expand = self._shift_means(
                Rit_X_expand, C_mask_all, C_mask_all, scale=scale
            )

        Rit_X = Rit_X_expand.sum(2).reshape(X.shape[0], -1, 4, 3)
        return Rit_X

    def multiply_covariance(self, dX, C):
        """Multiply by the covariance matrix.

        Args:
            dX (Tensor): Backbone tensor with dimensions
                `(num_batch, num_residues, 4, 3)`.
            (Note: this will typically be a gradient or direction vector,
            such as the score function. Not absolute coordinates).
            C (Tensor): Chain map with dimensions

        returns:
            C_dX (Tensor): The matrix-vector product resulting from
        left-multiplying by the covariance matrix.
        """
        # Covariance C = G @ G.T
        dX_flat = dX.reshape([dX.shape[0], -1, 3])
        Rt_dX = self._multiply_R_transpose(dX_flat, C)
        C_dX = self._multiply_R(Rt_dX, C)
        C_dX = C_dX.reshape(dX.shape)
        return C_dX

    def multiply_inverse_covariance(self, dX, C):
        """Multiply by the inverse covariance matrix.

        Args:
            dX (Tensor): Backbone tensor with dimensions
                `(num_batch, num_residues, 4, 3)`.
            C (Tensor): Chain map with dimensions

        returns:
            Ci_dX (Tensor): The matrix-vector product resulting from
        left-multiplying by the inverse covariance matrix.
        """
        # Covariance C = G @ G.T
        dX_flat = dX.reshape([dX.shape[0], -1, 3])
        Ri_dX = self._multiply_R_inverse(dX_flat, C)
        Ci_dX = self._multiply_R_inverse_transpose(Ri_dX, C)
        Ci_dX = Ci_dX.reshape(dX.shape)
        return Ci_dX

    def log_determinant(self, C):
        """Compute log determinant of the covariance matrix"""
        C_mask_all, C_mask_present = self._expand_masks(C)

        B, x_init_std, xi = self._globular_parameters(C_mask_all)
        a = self._scale
        B = B[..., 0]
        xi = xi[..., 0]

        # Compute determinants per chain
        N_chain = C_mask_all.sum([1, 3])
        logdet_chain = (
            N_chain * np.log(a) + torch.log(1.0 - xi) - 0.5 * torch.log(1.0 - B ** 2)
        )

        # We pick up one determinant per chain per spatial dimension (xyz)
        logdet = 3.0 * logdet_chain.sum(-1)
        return logdet

    def log_prob(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for Backbone MVN as follows:

        term1 = -n/2 log(2π)
        term2 = -1/2 log|Σ|
        term3 = -1/2 ∑_{i=1}^{n} (x_i - μ)^T Σ^-1 (x_i - μ)
        logP = term1 + term2 + term3

        Args:
            X (torch.Tensor): of size (batch, num_residues, 4, 3)
            C (torch.Tensor): of size (batch, num_residues)

        Returns:
            logp (torch.Tensor): of size (batch,)
        """
        term1 = -(C.shape[1] * 4 * 3) / 2 * np.log(2 * np.pi)
        term2 = -1 / 2 * self.log_determinant(C)
        term3 = -1 / 2 * (X * self.multiply_inverse_covariance(X, C)).sum([1, 2, 3])
        logp = term1 + term2 + term3

        return logp

    def sample(
        self,
        C: torch.Tensor,
        ddX: Optional[torch.Tensor] = None,
        Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Draws samples from the MVN.

        Args:
            C: (torch.Tensor): specifying the shape of the samples.
            ddX (torch.Tensor, optional): Optionally can specify a shift ddX which will be transformed (R^t * ddX) and used to shift Z.
            Z (torch.Tensor, optional): Optionally can specify random normal samples to transform into samples from the backbone MVN.

        Returns:
            X (torch.Tensor): of size (C.size(0), C.size(1), 4, 3) samples from the MVN.
        """
        num_batch = C.shape[0]
        num_residues = C.shape[1]

        if Z is None:
            Z = torch.randn([num_batch, num_residues * 4, 3], device=C.device)
        if ddX is None:
            X_flat = self._multiply_R(Z, C)
        else:
            RtddX = self._multiply_R_transpose(ddX, C)
            X_flat = self._multiply_R(Z + RtddX, C)

        X = X_flat.reshape([num_batch, num_residues, 4, 3])
        return X


class ConditionalBackboneMVNGlobular(BackboneMVNGlobular):
    """
    A conditional MVN distribution where some subset of the atomic coordinates are known.
    Args:
        covariance_model (str): Specifying the covariance_model of the base distribution (which respect to which we are conditioning).
        complex_scaling (bool): Specifying the complex_scaling of the base distribution (which respect to which we are conditioning).
        sigma_translation (float): Specifying the sigma_translation of the base distribution (which respect to which we are conditioning).
        X (torch.Tensor): of size (1, num_residues, 4, 3) containing atomic coordinates
        C (torch.Tensor): of size (1, num_residues) specifying chain specification
        D (torch.Tensor): of size (1, num_residues) containing 1s or 0s, (castable as Byte or Bool) where 1 indicates a residue's structural information
                          is to be conditioned on.
        gamma (float): This inflates variance of the center of mass of the samples generated by the CMVN.
    """

    def __init__(
        self,
        covariance_model: str = "brownian",
        complex_scaling: bool = False,
        sigma_translation: float = 1.0,
        X: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(covariance_model, complex_scaling, sigma_translation, **kwargs)
        assert D.shape[0] == 1
        assert C.shape[0] == 1
        assert X.shape[0] == 1
        self.gamma = gamma
        self.register_buffer("X", X)
        self.register_buffer("C", C)
        self.register_buffer("D", D.float())

        self._check_C(self.C.abs())
        R, RRt = self._materialize_RRt(self.C)
        self.register_buffer("R", R)
        self.register_buffer("RRt", RRt)

        R_clamp, RRt_clamp = self._condition_RRt(self.RRt, self.D)
        self.register_buffer("R_clamp", R_clamp)
        self.register_buffer("RRt_clamp", RRt_clamp)

        R_clamp_inverse = torch.linalg.pinv(self.R_clamp)
        self.register_buffer("R_clamp_inverse", R_clamp_inverse)
        self.register_buffer("mu_sample", self.mu(X))

    def _center_of_mass(self, X, C):
        mask_expand = (
            (C > 0).float().reshape(list(C.shape) + [1, 1]).expand([-1, -1, 4, -1])
        )
        X_mean = (mask_expand * X).sum([1, 2], keepdims=True) / (
            mask_expand.sum([1, 2], keepdims=True)
        )
        return X_mean

    def mu(self, X: torch.Tensor):
        """
        Returns the mean of the conditional distribution obtained by conditioning on atomic coordinates specified in `X` at
        the residues specified in `self.D`.
        Args:
            X (torch.Tensor): of size (B, num_residues, 4, 3)

        Returns:
            X_mu (torch.Tensor): of size (B, num_residues, 4, 3)
        """
        B, _, _, _ = X.size()
        loc = self._center_of_mass(X, self.D).squeeze().reshape(B, 1, 3)
        m = (self.D_atom[..., None] > 0).repeat(B, 1, 3)
        X_flat = X.reshape(B, -1, 3)
        X_restricted = X_flat[m].reshape(B, -1, 3)
        mu = loc + (self.S12 @ torch.linalg.pinv(self.S22) @ (X_restricted - loc))
        X_mu = X_flat.scatter(1, self.zero_indices[None, ..., None].repeat(B, 1, 3), mu)
        return X_mu.reshape(B, -1, 4, 3)

    def sample(
        self,
        num_batch: int = 1,
        Z: Optional[torch.Tensor] = None,
        mu_X: Optional[torch.Tensor] = None,
    ):
        """
        Draws samples from the conditional MVN with mean `mu_X`.

        Args:
            num_batch (int): Number of samples to draw.
            Z (torch.Tensor, optional): of size (batch, num_residues, 4, 3) random standard normal samples can be specified (that are transformed into samples from the CMVN)
            mu_X (torch.Tensor, optional): of size (batch, num_residues, 4, 3) optionally take the mean with respect to a different `X` tensor than was used to instantiate the class.

        Returns:
            samples (torch.Tensor): of size (num_batch, num_residues, 4, 3)
        """
        if Z is not None:
            num_batch = Z.shape[0]
        C_expand = self.C.repeat(num_batch, 1, 1, 1)
        mu = self.mu_sample.repeat(num_batch, 1, 1, 1)
        if mu_X is not None:
            mu = self.mu(mu_X)
        if Z is None:
            Z = torch.randn_like(mu)
        return mu + self._multiply_R(Z, C_expand)

    def _scatter(self, A, index, source):
        J = torch.zeros_like(A)
        J[index[:, None], index[None, :]] = source
        return J

    def _materialize_RRt(self, C):
        """As in C.4 of `https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1.full.pdf`"""
        a = self._scale
        bs, sl = C.size()
        Z = torch.randn(bs, 4 * sl, 3).to(C.device)
        C_mask_all, C_mask_present, Z_expand = self._expand_per_chain(Z, C)

        gamma = self.gamma if self.gamma is None or self.gamma > 0.0 else None
        b, x_init_std, xi = self._globular_parameters(
            C_mask_all, translation_inflation=gamma
        )

        C_atom = self._C_atomic(C.abs())
        R_center = self._build_R_center(C_atom, xi)
        R_sum = self._build_R_sum(C_atom, b)
        R_init = self._build_R_init(C_atom, b)

        R = a * R_center @ R_sum @ R_init
        RRt = R @ R.t()
        return R, RRt

    def _check_C(self, C):
        _C = C[0][:-1] - C[0][1:]
        if (_C > 0).any():
            raise ValueError("Chain map needs to be increasing in this class!")

    def _build_R_center(self, C_atom, xi):
        chain_indices = C_atom.unique()
        blocks = []
        for chain_index, _xi in zip(chain_indices, xi[0]):
            N = C_atom[C_atom == chain_index].numel()
            blocks.append(
                (
                    torch.eye(N, device=_xi.device)
                    - (_xi / N) * torch.ones(N, N, device=_xi.device)
                )
            )
        return torch.block_diag(*blocks)

    def _build_R_sum(self, C_atom, b):
        chain_indices = C_atom.unique()
        blocks = []
        for chain_index, _b in zip(chain_indices, b[0]):
            N = C_atom[C_atom == chain_index].numel()
            blocks.append(
                (
                    _b
                    ** (
                        torch.arange(N, device=_b.device).unsqueeze(0)
                        - torch.arange(N, device=_b.device).unsqueeze(-1)
                    )
                    .tril()
                    .abs()
                ).tril()
            )
        return torch.block_diag(*blocks)

    def _build_R_init(self, C_atom, b):
        indices = [(C_atom == k).float().argmax(1).item() for k in C_atom.unique()]
        N = C_atom.numel()
        P3 = torch.eye(N).to(C_atom.device)
        for index, _b in zip(indices, b[0]):
            P3.diagonal().data[index] = 1 / math.sqrt(1 - _b ** 2)
        return P3

    def _condition_RRt(self, RRt, D):
        """
        Args:
            RRt (torch.tensor): of size (N x N) the original full covariance
            D (torch.tensor): of dtype float and size (1xN) containing 1.0 for known indices else 0.0.
        """
        self.register_buffer("D_atom", self._C_atomic(D))
        self.register_buffer("zero_indices", torch.nonzero((1 - self.D_atom[0]))[:, 0])
        self.register_buffer("nonzero_indices", torch.nonzero(self.D_atom[0])[:, 0])

        self.register_buffer("S11", RRt[self.zero_indices][:, self.zero_indices])
        self.register_buffer("S12", RRt[self.zero_indices][:, self.nonzero_indices])
        self.register_buffer("S21", RRt[self.nonzero_indices][:, self.zero_indices])
        self.register_buffer("S22", RRt[self.nonzero_indices][:, self.nonzero_indices])

        S_clamp = self.S11 - ((self.S12 @ torch.linalg.pinv(self.S22) @ self.S21))
        R_clamp = torch.linalg.cholesky(S_clamp)
        self.register_buffer("RRt_clamp_restricted", R_clamp @ R_clamp.t())
        RRt_clamp = self._scatter(
            torch.zeros_like(RRt), self.zero_indices, self.RRt_clamp_restricted
        )
        R_clamp = self._scatter(torch.zeros_like(RRt), self.zero_indices, R_clamp)
        return R_clamp, RRt_clamp

    def _multiply_R(self, Z, C):
        Z_flat = Z.reshape([Z.shape[0], -1, 3])
        return (self.R_clamp @ Z_flat).reshape(Z.shape)

    def _multiply_R_transpose(self, Z, C):
        Z_flat = Z.reshape([Z.shape[0], -1, 3])
        return (self.R_clamp.t() @ Z_flat).reshape(Z.shape)

    def _multiply_R_inverse(self, X, C):
        X_flat = X.reshape([X.shape[0], -1, 3])
        return (self.R_clamp_inverse @ X_flat).reshape(X.shape)

    def _multiply_R_inverse_transpose(self, X, C):
        X_flat = X.reshape([X.shape[0], -1, 3])
        return (self.R_clamp_inverse.t() @ X_flat).reshape(X.shape)

    def multiply_covariance(self, dX, C):
        dX_flat = dX.reshape([dX.shape[0], -1, 3])
        return (self.RRt_clamp @ dX_flat).reshape(dX.shape)

    def multiply_inverse_covariance(self, dX, C):
        dX_flat = dX.reshape([dX.shape[0], -1, 3])
        return (self.RRt_clamp_inverse @ dX_flat).reshape(dX.shape)


class BackboneMVNResidueGas(torch.nn.Module):
    """
    Gaussian model for protein backbones.
    """

    def __init__(self, stddev_CA=10.0, stddev_atoms=1.0, **kwargs):
        super().__init__()
        self.stddev_CA = stddev_CA
        self.stddev_atoms = stddev_atoms

        # The full R matrix factorizes into a block diagonal of 4x4 matrices
        s1 = stddev_CA
        s2 = stddev_atoms
        # Atoms are N-CA-C=O
        R_local = torch.tensor(
            [[s2, s1, 0, 0], [0, s1, 0, 0], [0, s1, s2, 0], [0, s1, 0, s2]]
        ).float()
        self.register_buffer("R_local", R_local)
        self.register_buffer("Ri_local", torch.linalg.inv(R_local).detach())

    def _unflatten(self, Z):
        if len(Z.shape) == 3:
            num_batch, num_atoms, _ = Z.shape
            num_residues = num_atoms // 4
            Z_unflat = Z.reshape([num_batch, num_residues, 4, 3])
            return Z_unflat
        else:
            return Z

    def _multiply_R(self, Z, C):
        """Multiply by the square root of the covariance matrix"""
        Z_unflat = self._unflatten(Z)
        R_Z_unflat = torch.einsum("biax,ca->bicx", Z_unflat, self.R_local)
        R_Z = R_Z_unflat.reshape(Z.shape)
        return R_Z

    def _multiply_R_transpose(self, Z, C):
        """Multiply by the square root of the covariance matrix (transpose)"""
        Z_unflat = self._unflatten(Z)
        Rt_Z_unflat = torch.einsum("biax,ac->bicx", Z_unflat, self.R_local)
        Rt_Z = Rt_Z_unflat.reshape(Z.shape)
        return Rt_Z

    def _multiply_R_inverse(self, X, C):
        """Multiply by the inverse of the square root of the covariance matrix"""
        X_unflat = self._unflatten(X)
        Ri_X = torch.einsum("biax,ca->bicx", X_unflat, self.Ri_local)
        return Ri_X.reshape(X.shape)

    def _multiply_R_inverse_transpose(self, X, C):
        """Multiply by the inverse trasnpose of the square root of the
        covariance matrix
        """
        X_unflat = self._unflatten(X)
        Rit_X = torch.einsum("biax,ac->bicx", X_unflat, self.Ri_local)
        return Rit_X.reshape(X.shape)

    def multiply_covariance(self, dX, C):
        """Multiply by the covariance matrix.

        Args:
            dX (Tensor): Backbone tensor with dimensions
                `(num_batch, num_residues, 4, 3)`.
            (Note: this will typically be a gradient or direction vector,
            such as the score function. Not absolute coordinates).
            C (Tensor): Chain map with dimensions

        returns:
            C_dX (Tensor): The matrix-vector product resulting from
        left-multiplying by the covariance matrix.
        """
        # Covariance C = G @ G.T
        dX_flat = dX.reshape([dX.shape[0], -1, 3])
        Rt_dX = self._multiply_R_transpose(dX_flat, C)
        C_dX = self._multiply_R(Rt_dX, C)
        C_dX = C_dX.reshape(dX.shape)
        return C_dX

    def multiply_inverse_covariance(self, dX, C):
        """Multiply by the inverse covariance matrix.

        Args:
            dX (Tensor): Backbone tensor with dimensions
                `(num_batch, num_residues, 4, 3)`.
            C (Tensor): Chain map with dimensions

        returns:
            Ci_dX (Tensor): The matrix-vector product resulting from
        left-multiplying by the inverse covariance matrix.
        """
        # Covariance C = G @ G.T
        dX_flat = dX.reshape([dX.shape[0], -1, 3])
        Ri_dX = self._multiply_R_inverse(dX_flat, C)
        Ci_dX = self._multiply_R_inverse_transpose(Ri_dX, C)
        Ci_dX = Ci_dX.reshape(dX.shape)
        return Ci_dX

    def log_determinant(self, C):
        """Compute log determinant of the covariance matrix"""
        log_s1 = np.log(self.stddev_CA)
        log_s2 = np.log(self.stddev_atoms)
        num_residues = C.ne(0).float().sum(1)
        """ We have
                det([s2,s1, 0, 0],
                    [0, s1, 0, 0],
                    [0, s1,s2, 0],
                    [0, s1, 0,s2])
                =
                det([s1, 0, 0, 0],
                    [s1, s2, 0, 0],
                    [s1, 0,s2, 0],
                    [s1, 0, 0,s2])
                = s1 * s2^3
            And we pick up one determinant per residue per xyz dimension
        """
        logdet = 3 * num_residues * (log_s1 + 3.0 * log_s2)
        return logdet

    def log_prob(
        self, X: torch.Tensor, C: torch.Tensor, *, normalized: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample(
        self,
        C: torch.Tensor,
        ddX: Optional[torch.Tensor] = None,
        Z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the Gaussian."""
        num_batch = C.shape[0]
        num_residues = C.shape[1]

        if Z is None:
            Z = torch.randn([num_batch, num_residues * 4, 3], device=C.device)
        if ddX is None:
            X_flat = self._multiply_R(Z, C)
        else:
            RtddX = self._multiply_R_transpose(ddX, C)
            X_flat = self._multiply_R(Z + RtddX, C)

        X = X_flat.reshape([num_batch, num_residues, 4, 3])
        return X
