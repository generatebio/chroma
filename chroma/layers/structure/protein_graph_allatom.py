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

"""Layers for building graph representations of protein structure, all-atom.

This module contains pytorch layers for representing protein structure as a
graph with node and edge features based on geometric information. The graph
features are differentiable with respect to input coordinates and can be used
for building protein scoring functions and optimizing protein geometries
natively in pytorch.
"""


import numpy as np
import torch
import torch.nn as nn

from chroma.layers import graph
from chroma.layers.structure import geometry, sidechain


class NodeChiRBF(nn.Module):
    """Layers for featurizing chi angles with a smooth binning

    Args:
        num_chi_bins (int): Number of bins for discretizing chi angles.
        num_chi (int): Number of chi angles.
        dim_out (int): Number of output feature dimensions.
        bin_scale (float, optional): Scaling parameter that sets bin smoothing.

    Input:
        chi (Tensor): Chi angles with shape `(num_batch, num_residues, num_chi)`.

    Output:
        h_chi (Tensor): Chi angle features with shape
            `(num_batch, num_residues, num_chi * num_chi_bins)`.
    """

    def __init__(self, dim_out, num_chi, num_chi_bins, bin_scale=2.0):
        super(NodeChiRBF, self).__init__()
        self.dim_out = dim_out
        self.num_chi = num_chi
        self.num_chi_bins = num_chi_bins
        self.bin_scale = bin_scale

        self.embed = nn.Linear(self.num_chi * self.num_chi_bins, dim_out)

    def _featurize(self, chi, mask_chi=None):
        num_batch, num_residues, _ = chi.shape

        chi_bin_center = (
            torch.arange(0, self.num_chi_bins, device=chi.device)
            * 2.0
            * np.pi
            / self.num_chi_bins
        )
        chi_bin_center = chi_bin_center.reshape([1, 1, 1, -1])

        # Set smoothing length scale based on ratio beteen adjacent bin centers
        # bin_i / bin_i+1 = 1 / scale
        delta_adjacent = np.cos(0.0) - np.cos(2.0 * np.pi / self.num_chi_bins)
        cosine = torch.cos(chi.unsqueeze(-1) - chi_bin_center)
        chi_features = torch.exp((cosine - 1.0) * self.bin_scale / delta_adjacent)
        if mask_chi is not None:
            chi_features = mask_chi.unsqueeze(-1) * chi_features
        chi_features = chi_features.reshape(
            [num_batch, num_residues, self.num_chi * self.num_chi_bins]
        )
        return chi_features

    def forward(self, chi, mask_chi=None):
        chi_features = self._featurize(chi, mask_chi=mask_chi)
        h_chi = self.embed(chi_features)
        return h_chi


class EdgeSidechainsDirect(nn.Module):
    """Layers for direct encoding of side chain geometries.

    Args:
        dim_out (int): Number of output hidden dimensions.
        max_D (float, optional): Maximum distance cutoff for encoding
            of edges.

    Input:
        X (Tensor): All atom coordinates with shape
                `(num_batch, num_residues, 14, 3)`.
        C (LongTensor): Chain map with shape `(num_batch, num_residues)`.
        S (LongTensor): Sequence tensor with shape
            `(num_batch, num_residues)`.
        edge_idx (Tensor): Graph indices for expansion with shape
            `(num_batch, num_residues_out, num_neighbors)`. The dimension
            of output variables `num_residues_out` must either equal
            `num_residues` or 1, the latter of which can be useful for sequential
            decoding.

    Output:
        h (Tensor): Features with shape
            `(num_batch, num_residues_out, num_neighbors, num_hidden)`.
    """

    def __init__(
        self,
        dim_out,
        length_scale=7.5,
        distance_eps=0.1,
        num_fourier=30,
        fourier_order=2,
        basis_type="rff",
    ):
        super(EdgeSidechainsDirect, self).__init__()
        self.dim_out = dim_out
        self.length_scale = length_scale
        self.distance_eps = distance_eps

        # self.embed = nn.Linear(14 * 3 , dim_out)
        self.num_fourier = num_fourier
        self.rff = torch.nn.Parameter(
            2.0 * np.pi / self.length_scale * torch.randn((3, self.num_fourier))
        )
        self.basis_type = basis_type
        if self.basis_type == "rff":
            self.embed = nn.Linear(14 * self.num_fourier * 2, dim_out)
        elif self.basis_type == "spherical":
            self.fourier_order = fourier_order
            self.embed = nn.Linear(14 * (self.fourier_order * 2) ** 3, dim_out)

    def _local_coordinates(self, X, C, S, edge_idx):
        num_batch, num_residues, num_neighbors = edge_idx.shape

        # Mask and transform into features
        mask_atoms = sidechain.atom_mask(C, S)
        mask_atoms_j = graph.collect_neighbors(mask_atoms, edge_idx)
        mask_i = (C > 0).float().reshape([num_batch, num_residues, 1, 1])
        mask_atoms_ij = mask_i * mask_atoms_j

        # Build conditioning mask
        R_i, CA = geometry.frames_from_backbone(X[:, :, :4, :])

        # Transform neighbor X coordinates into local frames
        X_flat = X.reshape([num_batch, num_residues, -1])
        X_j_flat = graph.collect_neighbors(X_flat, edge_idx)
        X_j = X_j_flat.reshape([num_batch, num_residues, num_neighbors, 14, 3])
        dX_ij = X_j - CA.reshape([num_batch, num_residues, 1, 1, 3])
        U_ij = torch.einsum("niab,nijma->nijmb", R_i, dX_ij)
        return U_ij, mask_atoms_ij

    def _local_coordinates_t(self, t, X, C, S, edge_idx_t):
        num_batch, _, num_neighbors = edge_idx_t.shape
        num_residues = X.shape[1]

        # Make a mask that
        C_i = C[:, t].unsqueeze(1)
        # S_i = S[:,t].unsqueeze(1)
        # mask_atoms_i = sidechain.atom_mask(C_i, S_i)
        C_j = graph.collect_neighbors(C.unsqueeze(-1), edge_idx_t).reshape(
            [num_batch, num_neighbors]
        )
        S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx_t).reshape(
            [num_batch, num_neighbors]
        )
        mask_atoms_j = sidechain.atom_mask(C_j, S_j).unsqueeze(1)
        mask_i = (C_i > 0).float().reshape([num_batch, 1, 1, 1])
        mask_atoms_ij = mask_i * mask_atoms_j

        # Build conditioning mask
        X_bb_i = X[:, t, :4, :].unsqueeze(1)
        R_i, CA = geometry.frames_from_backbone(X_bb_i)

        # Transform neighbor X coordinates into local frames
        X_flat = X.reshape([num_batch, num_residues, -1])
        X_j_flat = graph.collect_neighbors(X_flat, edge_idx_t)
        X_j = X_j_flat.reshape([num_batch, 1, num_neighbors, 14, 3])
        dX_ij = X_j - CA.reshape([num_batch, 1, 1, 1, 3])
        U_ij = torch.einsum("niab,nijma->nijmb", R_i, dX_ij)
        return U_ij, mask_atoms_ij

    def _fourier_expand(self, h, order):
        k = torch.arange(order, device=h.device)
        k = k.reshape([1 for i in h.shape] + [-1])
        return torch.cat(
            [torch.sin(h.unsqueeze(-1) * (k + 1)), torch.cos(h.unsqueeze(-1) * k)],
            dim=-1,
        )

    def _featurize(self, U_ij, mask_atoms_ij):
        if self.basis_type == "rff":
            # Random fourier features
            U_ij = mask_atoms_ij.unsqueeze(-1) * U_ij
            U_ff = torch.einsum("nijax,xy->nijay", U_ij, self.rff)
            U_ff = torch.concat([torch.cos(U_ff), torch.sin(U_ff)], -1)

            # Gaussian RBF envelope
            D_ij = torch.sqrt((U_ij ** 2).sum(-1) + self.distance_eps)
            magnitude = torch.exp(-D_ij * D_ij / (2 * self.length_scale ** 2))
            U_ff = magnitude.unsqueeze(-1) * U_ff

            U_ff = U_ff.reshape(list(D_ij.shape)[:3] + [-1])
            h = mask_atoms_ij[:, :, :, 0].unsqueeze(-1) * self.embed(U_ff)

        elif self.basis_type == "spherical":
            # Convert to spherical coordinates
            r_ij = torch.sqrt((U_ij ** 2).sum(-1) + self.distance_eps)
            r_ij_scale = r_ij * 2.0 * np.pi / self.length_scale
            x, y, z = U_ij.unbind(-1)
            theta_ij = torch.acos(z / r_ij)
            phi_ij = torch.atan2(y, x)

            # Build Fourier expansions of each coordinate
            r_ff, theta_ff, phi_ff = [
                self._fourier_expand(h, self.fourier_order)
                for h in [r_ij_scale, theta_ij, phi_ij]
            ]
            # Radial envelope function
            r_envelope = mask_atoms_ij * torch.exp(
                -r_ij * r_ij / (2 * self.length_scale ** 2)
            )

            # Tensor outer product
            bf_ij = torch.einsum(
                "bika,bikar,bikat,bikap->bikartp", r_envelope, r_ff, theta_ff, phi_ff
            ).reshape(list(r_ij.shape)[:3] + [-1])

            h = mask_atoms_ij[:, :, :, 0].unsqueeze(-1) * self.embed(bf_ij)

        return h

    def forward(self, X, C, S, edge_idx):
        U_ij, mask_atoms_ij = self._local_coordinates(X, C, S, edge_idx)
        h = self._featurize(U_ij, mask_atoms_ij)
        return h

    def step(self, t, X, C, S, edge_idx_t):
        U_ij, mask_atoms_ij = self._local_coordinates_t(t, X, C, S, edge_idx_t)
        h = self._featurize(U_ij, mask_atoms_ij)
        return h
