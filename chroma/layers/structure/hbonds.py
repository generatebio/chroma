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

"""Layers for annotating hydrogen bonds in protein structures.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chroma.layers.graph import collect_neighbors
from chroma.layers.structure import protein_graph
from chroma.layers.structure.geometry import normed_vec


class BackboneHBonds(nn.Module):
    """Compute hydrogen bonds from protein backbones.

    We use the simple electrostatic model for calling hydrogen
    bonds of DSSP, which is described at
    https://en.wikipedia.org/wiki/DSSP_(algorithm). After
    placing virtual hydrogens on all backbone nitrogens,
    we consider potential hydrogen bonds with carbonyl groups
    on the backbone with residue distance |i-j| > 2. The
    picture is:

       -0.20e    +0.20e    -0.42e    +0.42e
        [N_i]-----[H_i] ::: [O_j]=====[C_j]

    Args:
        cutoff_energy (float, optional): Cutoff energy with
            default value -0.5 (DSSP).
        cutoff_distance (float, optional): Max distance
            between `N_i` and `O_j` with default value 3.6 angstroms.
        cutoff_gap (float, optional): Minimum tolerated residue
            distance, i.e. `|i-j| >= cutoff_gap`.
            Default value of 3.

    Inputs:
        X (Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_atom_types, 3)`.
        C (LongTensor): Chain map tensor with shape `(num_batch, num_residues)`.
        edge_idx (LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask_ij (Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.

    Outputs:
        hbonds (Tensor): Binary matrix annotating backbone hydrogen bonds
            with shape `(num_batch, num_nodes, num_neighbors)`.
        mask_hb_ij (Tensor): Hydrogen bond mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
        H_i (Tensor): Virtual hydrogen coordinates with shape
            `(num_batch, num_nodes, 3)`.
    """

    def __init__(
        self,
        cutoff_energy: float = -0.5,
        cutoff_distance: float = 3.6,
        cutoff_gap: float = 3,
        distance_eps: float = 1e-3,
    ) -> None:
        super(BackboneHBonds, self).__init__()
        self.cutoff_energy = cutoff_energy
        self.cutoff_distance = cutoff_distance
        self.cutoff_gap = cutoff_gap
        self._coefficient = 0.42 * 0.2 * 332
        self._eps = distance_eps

        # Lishan Yao et al. JACS 2008, NMR data
        self._length_NH = 1.015
        return

    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        edge_idx: torch.LongTensor,
        mask_ij: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_batch, num_residues, _, _ = X.shape
        # Collect coordinates at i and j
        X_flat = X.reshape([num_batch, num_residues, -1])
        X_j_flat = collect_neighbors(X_flat, edge_idx)
        X_j = X_j_flat.reshape([num_batch, num_residues, -1, 4, 3])

        # Get amide [N-H] atoms at i by
        # by placing virtual H from C_{i-1}-N-Ca neg bisector
        X_prev = F.pad(X, [0, 0, 0, 0, 1, 0], mode="replicate")[:, :-1, :, :]
        C_prev_i = X_prev[:, :, 2, :]
        N_i = X[:, :, 0, :]
        Ca_i = X[:, :, 1, :]
        u_CprevN_i = normed_vec(N_i - C_prev_i)
        u_CaN_i = normed_vec(N_i - Ca_i)
        u_NH_i = normed_vec(u_CprevN_i + u_CaN_i)
        H_i = N_i + self._length_NH * u_NH_i
        # Add broadcasting dimensions
        N_i = N_i[:, :, None, :]
        H_i = H_i[:, :, None, :]

        # Get carbonyl [C=O] atoms at j
        O_j = X_j[:, :, :, 3, :]
        C_j = X_j[:, :, :, 2, :]

        _invD = (
            lambda Xi, Xj: (Xi - Xj).square().sum(-1).add(self._eps).sqrt().reciprocal()
        )
        U_ij = self._coefficient * (
            _invD(N_i, O_j) - _invD(N_i, C_j) + _invD(H_i, C_j) - _invD(H_i, O_j)
        )

        # Mask any bonds exceeding donor/acceptor cutoff distance
        D_nonhydrogen = (N_i - O_j).square().sum(-1).add(self._eps).sqrt()
        mask_ij_cutoff_D = (D_nonhydrogen < self.cutoff_distance).float()

        # Mask hbonds on same chain with |i-j| < gap_cutoff
        mask_ij_nonlocal = 1.0 - _locality_mask(C, edge_idx, cutoff=self.cutoff_gap)

        # Ignore N terminal hydrogen bonding because of ambiguous hydrogen placement
        C_prev = F.pad(C, [1, 0], "constant")[:, 1:]
        mask_i = ((C > 0) * (C == C_prev)).float()
        mask_j = collect_neighbors(C[..., None], edge_idx)[..., 0]
        mask_ij_internal = mask_i[..., None] * (mask_j > 0).float()

        mask_hb_ij = mask_ij * mask_ij_nonlocal * mask_ij_cutoff_D * mask_ij_internal

        # Call hydrogen bonds
        hbonds = mask_hb_ij * (U_ij < self.cutoff_energy).float()
        return hbonds, mask_hb_ij, H_i


class LossBackboneHBonds(nn.Module):
    """Score hydrogen bond recovery from protein backbones.

    Args:
        See `BackboneHBonds`.

    Inputs:
        X (Tensor): Backbone coordinates to score with shape
            `(num_batch, num_residues, 4, 3)`.
        X_target (Tensor): Reference coordinates to compare to with shape
            `(num_batch, num_residues, 4, 3)`.
        C (LongTensor): Chain map tensor with shape `(num_batch, num_residues)`.

    Outputs:
        recovery_local (Tensor): Local hydrogen bond recovery with shape
            `(num_batch)`.
        recovery_nonlocal (Tensor): Nonlocal hydrogen bond recovery with shape
            `(num_batch)`.
        error_co (Tensor): Absolute error in terms of contact order recovery
    """

    def __init__(
        self,
        cutoff_local: float = 8,
        cutoff_energy: float = -0.5,
        cutoff_distance: float = 3.6,
        cutoff_gap: float = 3,
        distance_eps: float = 1e-3,
        num_neighbors: int = 30,
    ) -> None:
        super(LossBackboneHBonds, self).__init__()
        self.cutoff_local = cutoff_local
        self.cutoff_energy = cutoff_energy
        self.cutoff_distance = cutoff_distance
        self.cutoff_gap = cutoff_gap
        self._eps = 1e-3

        self.graph_builder = protein_graph.ProteinGraph(num_neighbors=num_neighbors)
        self.hbonds = BackboneHBonds(
            cutoff_energy=cutoff_energy,
            cutoff_distance=cutoff_distance,
            cutoff_gap=cutoff_gap,
        )

    def forward(
        self, X: torch.Tensor, X_target: torch.Tensor, C: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build Graph
        edge_idx, mask_ij = self.graph_builder(X_target, C)
        hb_target, mask_hb, H_i = self.hbonds(X_target, C, edge_idx, mask_ij)
        hb_current, _, _ = self.hbonds(X, C, edge_idx, mask_ij)

        # Split into local and long range hbonds
        mask_local = _locality_mask(C, edge_idx, cutoff=self.cutoff_local)
        hb_target_local = mask_local * hb_target
        hb_target_nonlocal = (1 - mask_local) * hb_target

        # Compute per complex
        recovery_local = (hb_current * hb_target_local).sum([1, 2]) / (
            hb_target_local.sum([1, 2]) + self._eps
        )
        recovery_nonlocal = (hb_current * hb_target_nonlocal).sum([1, 2]) / (
            hb_target_nonlocal.sum([1, 2]) + self._eps
        )

        # Compute contact order
        co_target = _contact_order(hb_target, C, edge_idx)
        co_current = _contact_order(hb_current, C, edge_idx)

        error_co = (co_target - co_current).abs()
        return recovery_local, recovery_nonlocal, error_co


def _ij_distance(
    C: torch.LongTensor, edge_idx: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    C_i = C[..., None]
    C_j = collect_neighbors(C_i, edge_idx)[..., 0]
    ix = torch.arange(C.shape[1], device=C.device)[None, :, None].expand(
        C.shape[0], -1, -1
    )
    jx = collect_neighbors(ix, edge_idx)[..., 0]
    dij = (jx - ix).abs()
    mask_same_chain = C_i.eq(C_j).float()
    return dij, mask_same_chain


def _contact_order(
    contacts: torch.Tensor,
    C: torch.LongTensor,
    edge_idx: torch.LongTensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute contact order"""
    dij, mask_same_chain = _ij_distance(C, edge_idx)
    mask_ij = mask_same_chain * contacts
    CO = (mask_ij * dij).sum([1, 2]) / (mask_ij + eps).sum([1, 2])
    return CO


def _locality_mask(
    C: torch.LongTensor, edge_idx: torch.LongTensor, cutoff: float,
) -> torch.Tensor:
    dij, mask_same_chain = _ij_distance(C, edge_idx)
    mask_ij_local = ((dij < cutoff) * mask_same_chain).float()
    return mask_ij_local
