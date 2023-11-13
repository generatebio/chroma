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

from __future__ import print_function

import numpy as np
import torch
import torch.linalg
import torch.nn as nn

from chroma.layers import graph
from chroma.layers.linalg import eig_leading
from chroma.layers.structure import geometry, protein_graph


class CrossRMSD(nn.Module):
    """Compute optimal RMSDs between two sets of structures.

    This module uses the quaternion-based approach for calculating RMSDs as
    described in `Using Quaternions to Calculate RMSD`, 2004, by Coutsias,
    Seok, and Dill. The minimal RMSD and associated rotation are computed in
    terms of the most positive eigenvalue and associated eigvector of a special
    4x4 matrix.

    Args:
        method (str, optional): Method for calculating the most postive
            eigenvalue. Can be `power` or `symeig`. If `symeig`, this will use
            `torch.symeig`, which is the most accurate method but tends to be
            very slow on GPU for large batches of RMSDs. If `power`, then use
            power iteration to estimate leading eigenvalues. Default is `power`.
        method_iter (int, optional): When the method is `power`, this argument
            sets the number of power iterations used for approximation.
            The default is 50, which has tended to produce estimates of optimal
            RMSD with sub-angstrom accuracy on test problems. Note: Convergence
            rates of power iteration can be highly variable dependening on the
            system. If accuracy is important, it is recommended to compare
            outputs with `symeig`-based RMSDs.

    Inputs:
        X_mobile (Tensor): Mobile coordinates, i.e. the "mobile" coordinates,
            with shape `(num_source, num_atoms, 3)`.
        X_target (Tensor): Target coordinates with shape
            `(num_target, num_atoms, 3)`.

    Outputs:
        RMSD (Tensors): RMSDs after optimal superposition for all pairs of
            source and target structures with shape `(num_source, num_target)`.
            While `forward` returns the Cartesian product of all possible
            alignments, i.e. (`num_source * num_target` alignments), the
            `pairedRMSD` will do the same calculation for zipped batches, i.e.
            `num_source` total alignments.
    """

    def __init__(self, method="power", method_iter=50, dither=True):
        super(CrossRMSD, self).__init__()

        self.method = method
        self.method_iter = method_iter
        self._eps = 1e-5
        self.dither = dither

        # R_to_F converts xyz cross-covariance matrices (3x3) to the (4x4) F
        # matrix of Coutsias et al. This F matrix encodes the optimal RMSD in
        # its spectra; namely, the eigenvector associated with the most
        # positive eigenvalue of F is the quaternion encoding the optimal
        # 3D rotation for superposition.
        # fmt: off
        R_to_F = np.zeros((9, 16)).astype("f")
        F_nonzero = [
        [(0,0,1.),(1,1,1.),(2,2,1.)],            [(1,2,1.),(2,1,-1.)],            [(2,0,1.),(0,2,-1.)],            [(0,1,1.),(1,0,-1.)],
                [(1,2,1.),(2,1,-1.)],  [(0,0,1.),(1,1,-1.),(2,2,-1.)],             [(0,1,1.),(1,0,1.)],             [(0,2,1.),(2,0,1.)],
                [(2,0,1.),(0,2,-1.)],             [(0,1,1.),(1,0,1.)],  [(0,0,-1.),(1,1,1.),(2,2,-1.)],             [(1,2,1.),(2,1,1.)],
                [(0,1,1.),(1,0,-1.)],             [(0,2,1.),(2,0,1.)],             [(1,2,1.),(2,1,1.)],  [(0,0,-1.),(1,1,-1.),(2,2,1.)]
        ]
        # fmt: on

        for F_ij, nonzero in enumerate(F_nonzero):
            for R_i, R_j, sign in nonzero:
                R_to_F[R_i * 3 + R_j, F_ij] = sign
        self.register_buffer("R_to_F", torch.tensor(R_to_F))

    def forward(self, X_mobile, X_target):
        num_source = X_mobile.size(0)
        num_target = X_target.size(0)
        num_atoms = X_mobile.size(1)

        # Center coordinates
        X_mobile = X_mobile - X_mobile.mean(dim=1, keepdim=True)
        X_target = X_target - X_target.mean(dim=1, keepdim=True)

        # CrossCov matrices contract over atoms
        R = torch.einsum("sai,taj->stij", [X_mobile, X_target])

        # F Matrix has leading eigenvector as optimal quaternion
        R_flat = R.reshape(num_source, num_target, 9)
        F = torch.matmul(R_flat, self.R_to_F).reshape(num_source, num_target, 4, 4)

        # Compute optimal quaternion by extracting leading eigenvector
        if self.method == "symeig":
            top_eig = torch.linalg.eigvalsh(F)[:, :, 3]
        elif self.method == "power":
            top_eig, vec = eig_leading(F, num_iterations=self.method_iter)
        else:
            raise NotImplementedError

        # Compute RMSD in terms of RMSD using the scheme of Coutsias et al
        norms = (X_mobile ** 2).sum(dim=[-1, -2]).unsqueeze(1) + (X_target ** 2).sum(
            dim=[-1, -2]
        ).unsqueeze(0)
        sqRMSD = torch.relu((norms - 2 * top_eig) / (num_atoms + self._eps))
        RMSD = torch.sqrt(sqRMSD)
        return RMSD

    def pairedRMSD(
        self,
        X_mobile,
        X_target,
        mask=None,
        compute_alignment=False,
        align_unmasked=False,
    ):
        """Compute optimal RMSDs between each corresponding batch members.

        Args:
            X_mobile (Tensor): Mobile coordinates with shape
                `(..., num_atoms, 3)`.
            X_target (Tensor): Target coordinates with shape
                `(..., num_atoms, 3)`.
            mask (Tensor, optional): Binary mask tensor for missing atoms with
                shape `(..., num_atoms)`.
            compute_alignment (boolean, optional): If True, also return the
                superposed coordinates.

        Returns:
            RMSD (Tensors): Optimal RMSDs after superposition for all pairs of
                input structures with shape `(...)`.
            X_mobile_transform (Tensor, optional): Superposed coordinates with
                shape `(..., num_atoms, 3)`. Requires
                `compute_alignment` = True`.
        """
        # Collapse all leading batch dimensions
        num_atoms = X_mobile.size(-2)
        batch_dims = list(X_mobile.shape)[:-2]
        X_mobile = X_mobile.reshape([-1, num_atoms, 3])
        X_target = X_target.reshape([-1, num_atoms, 3])
        num_batch = X_mobile.size(0)
        if mask is not None:
            mask = mask.reshape([-1, num_atoms])

        # Center coordinates
        if mask is None:
            X_mobile_mean = X_mobile.mean(dim=1, keepdim=True)
            X_target_mean = X_target.mean(dim=1, keepdim=True)
        else:
            mask_expand = mask.unsqueeze(-1)
            X_mobile_mean = torch.sum(mask_expand * X_mobile, 1, keepdim=True) / (
                torch.sum(mask_expand, 1, keepdim=True) + self._eps
            )
            X_target_mean = torch.sum(mask_expand * X_target, 1, keepdim=True) / (
                torch.sum(mask_expand, 1, keepdim=True) + self._eps
            )

        X_mobile_center = X_mobile - X_mobile_mean
        X_target_center = X_target - X_target_mean

        if mask is not None:
            X_mobile_center = mask_expand * X_mobile_center
            X_target_center = mask_expand * X_target_center

        # Cross-covariance matrices contract over atoms
        R = torch.einsum("sai,saj->sij", [X_mobile_center, X_target_center])

        # F Matrix has leading eigenvector as optimal quaternion
        R_flat = R.reshape(num_batch, 9)
        R_to_F = self.R_to_F.type(R_flat.dtype)
        F = torch.matmul(R_flat, R_to_F).reshape(num_batch, 4, 4)
        if self.dither:
            F = F + 1e-5 * torch.randn_like(F)

        # Compute optimal quaternion by extracting leading eigenvector
        if self.method == "symeig":
            L, V = torch.linalg.eigh(F)
            top_eig = L[:, 3]
            vec = V[:, :, 3]
        elif self.method == "power":
            top_eig, vec = eig_leading(F, num_iterations=self.method_iter)
        else:
            raise NotImplementedError

        # Compute RMSD using top eigenvalue
        norms = (X_mobile_center ** 2).sum(dim=[-1, -2]) + (X_target_center ** 2).sum(
            dim=[-1, -2]
        )
        sqRMSD = torch.relu((norms - 2 * top_eig) / (num_atoms + self._eps))
        rmsd = torch.sqrt(sqRMSD)

        if not compute_alignment:
            # Unpack leading batch dimensions
            rmsd = rmsd.reshape(batch_dims)
            return rmsd
        else:
            R = geometry.rotations_from_quaternions(vec, normalize=False)

            X_mobile_transform = torch.einsum("bxr,bir->bix", R, X_mobile_center)
            X_mobile_transform = X_mobile_transform + X_target_mean

            if mask is not None:
                X_mobile_transform = mask_expand * X_mobile_transform

            # Return the RMSD of the transformed coordinates
            rmsd_direct = rmsd_unaligned(X_mobile_transform, X_target, mask)

            # Unpack leading batch dimensions
            rmsd_direct = rmsd_direct.reshape(batch_dims)
            X_mobile_transform = X_mobile_transform.reshape(batch_dims + [num_atoms, 3])
            if align_unmasked:
                X_mobile_transform = X_mobile - X_mobile_mean
                X_mobile_transform = torch.einsum(
                    "bxr, bir -> bix",
                    R,
                    X_mobile_transform.view(X_mobile.size(0), -1, 3),
                )
                X_mobile_transform = X_mobile_transform + X_target_mean

            return rmsd_direct, X_mobile_transform


class BackboneRMSD(nn.Module):
    """Compute optimal RMSDs between two sets of backbones.

    This wraps `CrossRMSD` for use with XCS-formatted protein data.

    Args:
        method (str, optional): Method for calculating the most postive
            eigenvalue. Can be `power` or `symeig`. Default is `power`.
        method_iter (int, optional): Number of power iterations for eigenvalue
            approximation. Requires `method=power`. Default is 50.

    Inputs:
        X_mobile (Tensor): Mobile coordinates with shape
            `(num_source, num_atoms, 4, 3)`.
        X_target (Tensor): Target coordinates with shape
            `(num_target, num_atoms, 4, 3)`.
        C (Tensor): Chain map with shape `(num_batch, num_residues)`.

    Outputs:
        X_aligned (Tensor, optional): Superposed `X_mobile` with shape
            `(num_batch, num_atoms, 3)`.
        rmsd (Tensors): Optimal RMSDs after superposition with shape
            `(num_batch)`.
    """

    def __init__(self, method="symeig"):
        super(BackboneRMSD, self).__init__()
        self.rmsd = CrossRMSD(method=method)

    def align(self, X_mobile, X_target, C, align_unmasked=False):
        mask = (C > 0).type(torch.float32)
        mask_flat = mask.unsqueeze(-1).expand(-1, -1, 4).reshape(mask.shape[0], -1)

        X_mobile_flat = X_mobile.reshape(X_mobile.size(0), -1, 3)
        X_target_flat = X_target.reshape(X_target.size(0), -1, 3)
        rmsd, X_aligned = self.rmsd.pairedRMSD(
            X_mobile_flat,
            X_target_flat,
            mask=mask_flat,
            compute_alignment=True,
            align_unmasked=align_unmasked,
        )
        X_aligned = X_aligned.reshape(X_mobile.size()).contiguous()
        return X_aligned, rmsd


class LossFragmentRMSD(nn.Module):
    """Compute optimal fragment-pair RMSDs between two sets of backbones.

    Args:
        fragment_k (int, option): Fram
        method (str, optional): Method for calculating the most postive
            eigenvalue. Can be `power` or `symeig`. Default is `power`.
        method_iter (int, optional): Number of power iterations for eigenvalue
            approximation. Requires `method=power`. Default is 50.

    Inputs:
        X_mobile (Tensor): Mobile coordinates with shape
            `(num_source, num_atoms, 4, 3)`.
        X_target (Tensor): Target coordinates with shape
            `(num_target, num_atoms, 4, 3)`.
        edge_idx
        C (Tensor): Chain map with shape `(num_batch, num_residues)`.

    Outputs:
        rmsd (Tensor, optional): Per-site fragment RMSDs with shape
            `(num_batch)`.
    """

    def __init__(self, k=7, method="symeig", method_iter=50):
        super(LossFragmentRMSD, self).__init__()
        self.k = k
        self.rmsd = CrossRMSD(method=method, method_iter=method_iter)

    def forward(self, X_mobile, X_target, C, return_coords=False):
        # Discard potential sidechain coordinates
        X_mobile = X_mobile[:, :, :4, :]
        X_target = X_target[:, :, :4, :]

        # Build graph and pair fragments

        X_fragment_mobile, C_fragment_mobile = _collect_X_fragments(X_mobile, C, self.k)
        X_fragment_target, C_fragment_target = _collect_X_fragments(X_target, C, self.k)
        shape = list(C.shape) + [-1, 3]
        X_fragment_mobile = X_fragment_mobile.reshape(shape)
        X_fragment_target = X_fragment_target.reshape(shape)

        mask = (C_fragment_mobile > 0).float()
        rmsd, X_fragment_mobile_align = self.rmsd.pairedRMSD(
            X_fragment_mobile, X_fragment_target, mask, compute_alignment=True
        )
        if return_coords:
            return rmsd, X_fragment_target, X_fragment_mobile, X_fragment_mobile_align
        else:
            return rmsd


class LossFragmentPairRMSD(nn.Module):
    """Compute optimal fragment-pair RMSDs between two sets of backbones.

    Args:
        fragment_k (int, option): Fram
        method (str, optional): Method for calculating the most postive
            eigenvalue. Can be `power` or `symeig`. Default is `power`.
        method_iter (int, optional): Number of power iterations for eigenvalue
            approximation. Requires `method=power`. Default is 50.

    Inputs:
        X_mobile (Tensor): Mobile coordinates with shape
            `(num_source, num_atoms, 4, 3)`.
        X_target (Tensor): Target coordinates with shape
            `(num_target, num_atoms, 4, 3)`.
        edge_idx
        C (Tensor): Chain map with shape `(num_batch, num_residues)`.

    Outputs:
        rmsd (Tensor, optional): Per-site fragment RMSDs with shape
            `(num_batch)`.
    """

    def __init__(self, k=7, method="symeig", method_iter=50, graph_num_neighbors=30):
        super(LossFragmentPairRMSD, self).__init__()
        self.k = k
        self.rmsd = CrossRMSD(method=method, method_iter=method_iter)
        self.graph_builder = protein_graph.ProteinGraph(
            num_neighbors=graph_num_neighbors
        )

    def _stack_neighbor(self, node_h, edge_idx):
        neighbor_h = graph.collect_neighbors(node_h, edge_idx)
        node_h = node_h[:, :, None, :].expand(neighbor_h.shape)
        edge_h = torch.cat([neighbor_h, node_h], dim=-1)
        return edge_h

    def _collect_X_fragment_pairs(self, X, C, edge_idx):
        X_kmer, C_kmer = _collect_X_fragments(X, C, self.k)
        X_pair = self._stack_neighbor(X_kmer, edge_idx)
        C_pair = self._stack_neighbor(C_kmer, edge_idx)
        X_pair = X_pair.reshape(list(X_pair.shape)[:-1] + [-1, 3])
        return X_pair, C_pair

    def forward(self, X_mobile, X_target, C, return_coords=False):
        # Discard potential sidechain coordinates
        X_mobile = X_mobile[:, :, :4, :]
        X_target = X_target[:, :, :4, :]

        # Build graph and pair fragments
        edge_idx, mask_ij = self.graph_builder(X_target, C)
        X_pair_mobile, C_pair_mobile = self._collect_X_fragment_pairs(
            X_mobile, C, edge_idx
        )
        X_pair_target, C_pair_target = self._collect_X_fragment_pairs(
            X_target, C, edge_idx
        )

        mask = (C_pair_mobile > 0).float()

        rmsd, X_pair_mobile_align = self.rmsd.pairedRMSD(
            X_pair_mobile, X_pair_target, mask, compute_alignment=True
        )
        if return_coords:
            return rmsd, mask_ij, X_pair_target, X_pair_mobile, X_pair_mobile_align
        else:
            return rmsd, mask_ij


class LossNeighborhoodRMSD(nn.Module):
    """Compute optimal fragment-pair RMSDs between two sets of backbones.

    Args:
        fragment_k (int, option): Fram
        method (str, optional): Method for calculating the most postive
            eigenvalue. Can be `power` or `symeig`. Default is `power`.
        method_iter (int, optional): Number of power iterations for eigenvalue
            approximation. Requires `method=power`. Default is 50.

    Inputs:
        X_mobile (Tensor): Mobile coordinates with shape
            `(num_source, num_atoms, 4, 3)`.
        X_target (Tensor): Target coordinates with shape
            `(num_target, num_atoms, 4, 3)`.
        edge_idx
        C (Tensor): Chain map with shape `(num_batch, num_residues)`.

    Outputs:
        rmsd (Tensor, optional): Per-site fragment RMSDs with shape
            `(num_batch)`.
    """

    def __init__(self, method="symeig", method_iter=50, graph_num_neighbors=30):
        super(LossNeighborhoodRMSD, self).__init__()
        self.rmsd = CrossRMSD(method=method, method_iter=method_iter)
        self.graph_builder = protein_graph.ProteinGraph(
            num_neighbors=graph_num_neighbors
        )

    def _collect_X_neighborhood(self, X, C, edge_idx):
        num_batch, num_nodes, num_atoms, _ = X.shape
        shape_flat = [num_batch, num_nodes, -1]
        X_flat = X.reshape(shape_flat)
        C_flat = C[..., None].expand([-1, -1, num_atoms])
        X_neighborhood = graph.collect_neighbors(X_flat, edge_idx).reshape(
            [num_batch, num_nodes, -1, 3]
        )
        C_neighborhood = graph.collect_neighbors(C_flat, edge_idx).reshape(
            [num_batch, num_nodes, -1]
        )
        return X_neighborhood, C_neighborhood

    def forward(self, X_mobile, X_target, C, return_coords=False):
        # Discard potential sidechain coordinates
        X_mobile = X_mobile[:, :, :4, :]
        X_target = X_target[:, :, :4, :]

        # Build graph and pair fragments
        edge_idx, mask_ij = self.graph_builder(X_target, C)
        X_neighborhood_mobile, C_neighborhood_mobile = self._collect_X_neighborhood(
            X_mobile, C, edge_idx
        )
        X_neighborhood_target, C_neighborhood_target = self._collect_X_neighborhood(
            X_target, C, edge_idx
        )
        mask = (C_neighborhood_mobile > 0).float()

        rmsd, X_neighborhood_mobile_align = self.rmsd.pairedRMSD(
            X_neighborhood_mobile, X_neighborhood_target, mask, compute_alignment=True
        )
        mask = (mask.sum(-1) > 0).float()
        if return_coords:
            return (
                rmsd,
                mask,
                X_neighborhood_target,
                X_neighborhood_mobile,
                X_neighborhood_mobile_align,
            )
        else:
            return rmsd, mask


def rmsd_unaligned(X_a, X_b, mask=None, eps=1e-5, _min_rmsd=1e-8):
    """Compute RMSD between two coordinate sets without alignment.

    Args:
        X_a (Tensor): Coordinate set 1 with shape `(..., num_points, 3)`.
        X_b (Tensor): Coordinate set 2 with shape `(..., num_points, 3)`.
        mask (Tensor, optional): Mask with shape `(..., num_points)`.
        eps (float, optional): Small number to prevent division by zero.
            default is 1E-5.

    Returns:
        rmsd (Tensor): Root mean squared deviations (raw) with shape `(...)`.
    """
    squared_dev = ((X_a - X_b) ** 2).sum(-1)
    if mask is None:
        rmsd = torch.sqrt(squared_dev.mean(-1).clamp(min=_min_rmsd))
    else:
        rmsd = torch.sqrt(
            (mask * squared_dev).sum(-1).clamp(min=_min_rmsd) / (mask.sum(-1) + eps)
        )
    return rmsd


def _collect_X_fragments(X, C, k):
    num_batch, num_nodes, num_atoms, _ = X.shape
    shape_flat = [num_batch, num_nodes, -1]
    X_flat = X.reshape(shape_flat)
    C_flat = C[..., None].expand([-1, -1, num_atoms])

    # Grab local kmers
    X_kmer = _collect_kmers(X_flat, k).reshape(shape_flat)
    C_kmer = _collect_kmers(C_flat, k).reshape(shape_flat)

    # Treat noncontiguous atoms as missing
    C_kmer = torch.where(C[..., None].eq(C_kmer), C_kmer, -C_kmer.abs())
    return X_kmer, C_kmer


def _collect_kmers(node_h, k):
    """Gather `(B,I,H) => (B,I,K,H)`"""
    device = node_h.device
    num_batch, num_nodes, _ = node_h.shape

    # Build indices
    k_idx = torch.arange(k, device=device) - (k - 1) // 2
    node_idx = torch.arange(node_h.shape[1], device=device)
    kmer_idx = node_idx[None, :, None] - k_idx[None, None, :]
    kmer_idx = kmer_idx.clamp(min=0, max=num_nodes - 1).long()
    kmer_idx = kmer_idx.expand([num_batch, -1, k])

    # Collect neighbors
    kmer_h = graph.collect_neighbors(node_h, kmer_idx)
    return kmer_h
