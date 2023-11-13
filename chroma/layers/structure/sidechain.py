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

"""Layers for modeling protein side chain conformations.

This module contains layers for building, measuring, and scoring protein side 
chain conformations in a differentiable way. These can be used for tasks such
as building differentiable all-atom structures from chi-angles, computing chi
angles from existing structures, and scoring or optimizing side chains using 
symmetry-aware rmsds.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chroma import constants
from chroma.layers import graph
from chroma.layers.structure import protein_graph
from chroma.layers.structure.geometry import (
    dihedrals,
    extend_atoms,
    frames_from_backbone,
    quaternions_from_rotations,
    rotations_from_quaternions,
)


class SideChainBuilder(nn.Module):
    """Protein side chain builder from chi angles.

    When only partial information is given such as chi angles, this module
    will default to using the ideal geometries given in the CHARMM toppar
    topology files.

    `Optimization of the additive CHARMM all-atom protein force
    field targeting improved sampling of the backbone phi,
    psi and side-chain chi1 and chi2 dihedral angles`

    Inputs:
        X (tensor): Backbone coordinates with shape
            `(batch_size, num_residues, 4, 3)`.
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.
        chi (tensor): Backbone chi angles with shape
            `(batch_size, num_residues, 4)`.

    Outputs:
        X (tensor): All-atom coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        mask_X (tensor): Atomic mask with shape
            `(batch_size, num_residues, 14, 1)`
    """

    def __init__(self, distance_eps=1e-6):
        super(SideChainBuilder, self).__init__()
        self.num_atoms = 10
        self.num_chi = 4
        self.num_aa = len(constants.AA20)
        self.distance_eps = distance_eps

        self._init_maps()

    def _init_maps(self):
        """Build geometry and topology maps in tensor form."""

        shape = (3, self.num_atoms, self.num_aa)
        self.register_buffer("_Z", torch.zeros(shape, dtype=torch.float))
        self.register_buffer("_parents", torch.zeros(shape, dtype=torch.long))
        self.register_buffer(
            "_chi_ix", 10 * torch.ones((self.num_chi, self.num_aa), dtype=torch.long)
        )

        for i, aa in enumerate(constants.AA20_3):
            aa_dict = constants.AA_GEOMETRY[aa]
            atoms_parents = ["N", "CA", "C", "O"] + aa_dict["atoms"]
            for j, atom in enumerate(aa_dict["atoms"]):
                # Internal coordinates per atom
                self._Z[0, j, i] = aa_dict["z-lengths"][j]
                self._Z[1, j, i] = aa_dict["z-angles"][j]
                self._Z[2, j, i] = aa_dict["z-dihedrals"][j]

                # Parent indices per atom
                parents = [atoms_parents.index(p) for p in aa_dict["parents"][j]]
                self._parents[0, j, i] = parents[0]
                self._parents[1, j, i] = parents[1]
                self._parents[2, j, i] = parents[2]

            # Map which chi angles are flexible
            for j, parent_ix in enumerate(aa_dict["chi_indices"]):
                self._chi_ix[j, i] = parent_ix

        # Convert angles from degrees to radians
        self._Z[1:, :, :] = self._Z[1:, :, :] * np.pi / 180.0

        # Manually fix Arginine, for which CHARMM places NH1 in trans to CD
        self._Z[2, 5, constants.AA20.index("R")] = 0.0

    def forward(self, X, C, S, chi=None):
        num_batch, num_residues = list(S.shape)

        if X.shape[2] > 4:
            X = X[:, :, :4, :]

        # Expand sequence indexing tensors for gathering residue-specific info
        # (B,L) => (B,L,4)
        S_expand3 = S.unsqueeze(-1).expand(-1, -1, 4)
        # (B,L) => (B,AA,ATOM,L)
        S_expand4 = S.reshape([num_batch, 1, 1, num_residues]).expand(
            -1, 3, self.num_atoms, -1
        )

        def _gather(Z):
            Z_expand = Z.unsqueeze(0).expand([num_batch, -1, -1, -1])
            # (B,3,ATOM,AA) @ (B,3,ATOM,L) => (B,3,ATOM,L) => (B,L,3,ATOM)
            Z_i = torch.gather(Z_expand, -1, S_expand4).permute([0, 3, 1, 2])
            return Z_i

        # Build ideal geometry length, angle, and dihedral tensors 3x(B,L,10)
        B, A, D = _gather(self._Z).unbind(-2)

        if chi is not None:
            # Scatter chi angles (B,L,4) onto their corresponding dihedrals (B,L,10)
            # (4,AA) => (B,AA,4)
            chi_ix_expand = (
                self._chi_ix.unsqueeze(0).expand([num_batch, -1, -1]).transpose(-2, -1)
            )
            # (B,AA,4) @ (B,L,4) => (B,L,4)
            chi_ix_i = torch.gather(chi_ix_expand, -2, S_expand3)

            # Scatter extra chi angles into an extra pad dimension & re-slice
            # (B,L,10) <- (B,L,4),(B,L,4) => (B,L,10)
            D_pad = F.pad(D, (0, 1))
            D_pad = torch.scatter(D_pad, -1, chi_ix_i, chi)
            D = D_pad[:, :, : self.num_atoms]

        # Build indices of parent atoms (B,L,3,10)
        X_full = F.pad(X, (0, 0, 0, self.num_atoms))
        parents = _gather(self._parents)

        # Build atom i given current frame
        for i in range(self.num_atoms):
            # Gather parents (B,L,A,3) => (B,L,3,3)
            parents_expand = parents[:, :, :, i].unsqueeze(-1).expand(-1, -1, -1, 3)
            # (B,L,A,3) @ (B,L,3,3) => (B,L,3,3)
            X1, X2, X3 = torch.gather(X_full, -2, parents_expand).unbind(-2)

            # Extend atom i
            X4 = extend_atoms(
                X1,
                X2,
                X3,
                B[:, :, i],
                A[:, :, i],
                D[:, :, i],
                degrees=False,
                distance_eps=self.distance_eps,
            )

            # Scatter
            # X[:,:,i+4,:] = X4
            # scatter_ix = (i+4) * torch.ones(
            #     (num_batch,num_residues,1,3), dtype=torch.long
            # )
            # print(X_full.shape, X4.shape, scatter_ix.shape, i+4)
            # print(scatter_ix)
            # X_full.scatter_(-2, scatter_ix, X4.unsqueeze(-2))
            # X_full = torch.scatter(X_full, -2, scatter_ix, X4)
            # X_full = X_full + 0.1*X4.mean()

            # For some reason direct scatter causes autograd issues
            X4_expand = F.pad(X4.unsqueeze(-2), (0, 0, 4 + i, 9 - i))
            X_full = X_full + X4_expand

            # DEBUG: TEST
            if False:
                D_reconstruct = dihedrals(X1, X2, X3, X4)
                D_error = (
                    (torch.cos(D[:, :, i]) - torch.cos(D_reconstruct)) ** 2
                    + (torch.sin(D[:, :, i]) - torch.sin(D_reconstruct)) ** 2
                ).mean()
                print(D_error)

        mask_X = atom_mask(C, S).unsqueeze(-1)
        X_full = mask_X * X_full
        return X_full, mask_X


class ChiAngles(nn.Module):
    """Computes Chi-angles from an all-atom protein structure.

    Inputs:
        X (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        chi (tensor): Backbone chi angles with shape
            `(batch_size, num_residues, 4)`.
        mask_chi (tensor): Chi angle mask with shape
            `(batch_size, num_residues, 4)`.
    """

    def __init__(self, distance_eps=1e-6):
        super(ChiAngles, self).__init__()
        self.num_atoms = 10
        self.num_chi = 4
        self.num_aa = len(constants.AA20)

        self.distance_eps = distance_eps

        self._init_maps()

    def _init_maps(self):
        """Build geometry and topology maps in tensor form."""

        self.register_buffer(
            "_chi_atom_sets",
            torch.zeros((self.num_aa, self.num_chi, 4), dtype=torch.long),
        )

        for i, aa in enumerate(constants.AA20_3):
            aa_dict = constants.AA_GEOMETRY[aa]
            atoms_names = ["N", "CA", "C", "O"] + aa_dict["atoms"]

            # Map which chi angles are flexible
            for j, parent_ix in enumerate(aa_dict["chi_indices"]):
                atom_quartet = aa_dict["parents"][parent_ix] + [
                    aa_dict["atoms"][parent_ix]
                ]
                for k, atom in enumerate(atom_quartet):
                    self._chi_atom_sets[i, j, k] = atoms_names.index(atom)

    def forward(self, X, C, S):
        num_batch, num_residues = list(S.shape)
        # (B,L) => (B,L,16)
        S_expand = S.unsqueeze(-1).expand([-1, -1, 16])

        # (AA,CHI,ATOM) => (AA,16) => (B,AA,16)
        chi_indices_per_aa = self._chi_atom_sets.reshape([1, self.num_aa, 16])
        chi_indices_per_aa = chi_indices_per_aa.expand([num_batch, -1, -1])

        # (B,AA,16) @ (B,L,16) => (B,L,16) => (B,L,16)
        chi_indices = torch.gather(chi_indices_per_aa, -2, S_expand)
        chi_indices = chi_indices.unsqueeze(-1).expand([-1, -1, -1, 3])

        # (B,L,14,3) @ (B,L,16,3)  => (B,L,16,3) => (B,L,4,4,3) => (B,L,4)
        X_chi = torch.gather(X, -2, chi_indices)
        X_1, X_2, X_3, X_4 = X_chi.reshape([num_batch, num_residues, 4, 4, 3]).unbind(
            -2
        )

        chi = dihedrals(X_1, X_2, X_3, X_4, distance_eps=self.distance_eps)

        mask_chi = chi_mask(C, S)
        chi = chi * mask_chi
        return chi, mask_chi


class SideChainSymmetryRenamer(nn.Module):
    """Rename atom to their 180-degree symmetry alternatives via permutation.

    Inputs:
        X (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        X_renamed (tensor): Renamed atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
    """

    def __init__(self):
        super(SideChainSymmetryRenamer, self).__init__()
        self.num_atoms = 10
        self.num_aa = len(constants.AA20)

        # Build symmetry indices give alternative atom labelings
        self.register_buffer(
            "_symmetry_indices",
            torch.arange(self.num_atoms).unsqueeze(0).repeat(self.num_aa, 1),
        )
        for i, aa in enumerate(constants.AA20_3):
            if aa in constants.ATOM_SYMMETRIES:
                for aa_1, aa_2 in constants.ATOM_SYMMETRIES[aa]:
                    atom_names = constants.AA_GEOMETRY[aa]["atoms"]
                    ix_1 = atom_names.index(aa_1)
                    ix_2 = atom_names.index(aa_2)
                    self._symmetry_indices[i, ix_1] = ix_2
                    self._symmetry_indices[i, ix_2] = ix_1

    def _gather_per_residue(self, AA_table, S):
        num_batch, num_residues = list(S.shape)

        # (B,L) => (B,L,ATOM)
        S_expand = S.unsqueeze(-1).expand([-1, -1, self.num_atoms])

        # (AA,ATOM) => (B,AA,ATOM)
        value_per_aa = AA_table.unsqueeze(0).expand([num_batch, -1, -1])

        # (B,AA,ATOM) @ (B,L,ATOM) => (B,L,ATOM)
        value_per_residue = torch.gather(value_per_aa, -2, S_expand)
        return value_per_residue

    def forward(self, X, S):
        alt_indices = self._gather_per_residue(self._symmetry_indices, S)
        alt_indices = alt_indices.unsqueeze(-1).expand(-1, -1, -1, 3)

        X_bb, X_sc = X[:, :, :4, :], X[:, :, 4:, :]
        X_sc_alternate = torch.gather(X_sc, -2, alt_indices)
        X_alternate = torch.cat([X_bb, X_sc_alternate], dim=-2)
        return X_alternate


class AllAtomFrameBuilder(nn.Module):
    """Build all-atom protein structure from oriented C-alphas and chi angles.

    Inputs:
        x (Tensor): C-alpha coordinates with shape `(num_batch, num_residues, 3)`.
        q (Tensor): Quaternions representing C-alpha orientiations with shape
            with shape `(num_batch, num_residues, 4)`.
        chi (tensor): Backbone chi angles with shape
            `(num_batch, num_residues, 4)`.
        C (tensor): Chain map with shape `(num_batch, num_residues)`.
        S (tensor): Sequence tokens with shape `(num_batch, num_residues)`.

    Outputs:
        X (Tensor): All-atom protein coordinates with shape
            `(num_batch, num_residues, 14, 3)`
    """

    def __init__(self):
        super(AllAtomFrameBuilder, self).__init__()
        self.sidechain_builder = SideChainBuilder()
        self.chi_angles = ChiAngles()

        # Build idealized backbone fragment
        # IC +N   CA   *C   O     1.3558 116.8400  180.0000 122.5200  1.2297
        dX = torch.tensor(
            [
                [1.459, 0.0, 0.0],  # N-C via Engh & Huber is 1.459
                [0.0, 0.0, 0.0],  # CA is origin
                [-0.547, 0.0, -1.424],  # C is placed 1.525 A @ 111 degrees from N
            ],
            dtype=torch.float32,
        )
        self.register_buffer("_dX_local", dX)

    def forward(self, x, q, chi, C, S):
        # Build backbone
        R = rotations_from_quaternions(q, normalize=True)
        dX = torch.einsum("ay,nixy->niax", self._dX_local, R)
        X_chain = x.unsqueeze(-2) + dX

        # Build carboxyl groups
        X_N, X_CA, X_C = X_chain.unbind(-2)

        # TODO: fix this behavior for termini
        mask_next = (C > 0).float()[:, 1:].unsqueeze(-1)
        X_N_next = F.pad(mask_next * X_N[:, 1:,], (0, 0, 0, 1),)

        num_batch, num_residues = C.shape
        ones = torch.ones(list(C.shape), dtype=torch.float32, device=C.device)
        X_O = extend_atoms(
            X_N_next,
            X_CA,
            X_C,
            1.2297 * ones,
            122.5200 * ones,
            180 * ones,
            degrees=True,
        )
        X_bb = torch.stack([X_N, X_CA, X_C, X_O], dim=-2)

        # Build sidechains
        X, mask_atoms = self.sidechain_builder(X_bb, C, S, chi)
        return X, mask_atoms

    def inverse(self, X, C, S):
        X_bb = X[:, :, :4, :]
        R, x = frames_from_backbone(X_bb)
        q = quaternions_from_rotations(R)
        chi, mask_chi = self.chi_angles(X, C, S)
        return x, q, chi


class LossSideChainRMSD(nn.Module):
    """Compute side chain RMSDs per residues from an all-atom protein structure.

    Inputs:
        X (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        X_target (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        chi (tensor): Backbone chi angles with shape
            `(batch_size, num_residues, 4)`.
    """

    def __init__(self, rmsd_eps=1e-2):
        super(LossSideChainRMSD, self).__init__()
        self.num_atoms = 10
        self.num_aa = len(constants.AA20)

        self.rmsd_eps = rmsd_eps
        self.renamer = SideChainSymmetryRenamer()

    def _rmsd(self, X, X_target, atom_mask):
        sd = atom_mask * ((X - X_target) ** 2).sum(-1)
        rmsd = torch.sqrt(
            sd.sum(-1) / (atom_mask.sum(-1) + self.rmsd_eps) + self.rmsd_eps
        )
        return rmsd

    def forward(self, X, X_target, C, S, include_symmetry=True):
        mask_atoms = atom_mask(C, S)

        X_alt = self.renamer(X, S)[:, :, 4:, :]
        X = X[:, :, 4:, :]
        X_target = X_target[:, :, 4:, :]
        mask_atoms = mask_atoms[:, :, 4:]

        rmsd = self._rmsd(X, X_target, mask_atoms)
        if include_symmetry:
            rmsd_alternate = self._rmsd(X_alt, X_target, mask_atoms)

            # rmsd = torch.minimum(rmsd, rmsd_alternate)
            rmsd = torch.stack([rmsd, rmsd_alternate], -1).min(-1)[0]
        rmsd = (C > 0).float() * rmsd
        return rmsd


class LossFrameAlignedGraph(nn.Module):
    """Compute the frame-aligned loss on a nearest neighbors graph.

    Args:
        num_neighbors (int): Number of neighbors to build in the graph. Default
            is 30.

    Inputs:
        X (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        X_target (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        D (tensor): Per-residue losses with shape `(batch_size, num_residues)`.
    """

    def __init__(
        self,
        num_neighbors=30,
        distance_eps=1e-2,
        distance_scale=10.0,
        interface_only=False,
    ):
        super(LossFrameAlignedGraph, self).__init__()
        self.distance_eps = distance_eps
        self.distance_scale = distance_scale

        self.renamer = SideChainSymmetryRenamer()
        self.graph_builder = protein_graph.ProteinGraph(num_neighbors)
        self.interface_only = interface_only

    def _frame_ij(self, X, edge_idx):
        # Build local frames
        num_batch, num_residues, num_atoms, _ = X.shape

        # Build frames at neighbor j (B,L,K,3,3), (B,L,K,3)
        X_bb_flat = X[:, :, :4, :].reshape([num_batch, num_residues, -1])
        X_j_flat = graph.collect_neighbors(X_bb_flat, edge_idx)
        X_j = X_j_flat.reshape([num_batch, num_residues, -1, 4, 3])
        R_j, X_j_CA = frames_from_backbone(X_j, distance_eps=self.distance_eps)

        # (B,L,1,A,3) - (B,L,K,1,3) => (B,L,K,A,3)
        X_ij = X.unsqueeze(-3) - X_j_CA.unsqueeze(-2)

        # Rotate displacements into local frames
        r_ij = torch.einsum("nijax,nijxy->nijay", X_ij, R_j)
        return r_ij

    def _dist(self, r_ij_1, r_ij_2):
        D_sq = (r_ij_1 - r_ij_2) ** 2
        D = torch.sqrt(D_sq.sum(-1) + self.distance_eps)
        return D

    def forward(self, X, X_target, C, S):
        if X_target.size(2) == 14:
            mask_atoms = atom_mask(C, S)
            X_alt = self.renamer(X, S)
        elif X_target.size(2) == 4:
            mask_atoms = (C > 0).float().unsqueeze(-1).expand([-1, -1, 4])
            X_alt = X
        else:
            raise Exception(
                "Size of atom dimension must be 4 (backbone) or 14 (all-atom)."
            )

        # Build the union graph
        custom_mask_2D = None
        if self.interface_only:
            custom_mask_2D = torch.ne(C.unsqueeze(1), C.unsqueeze(2)).float()
        edge_idx_model, _ = self.graph_builder(
            X[:, :, :4, :], C, custom_mask_2D=custom_mask_2D
        )
        edge_idx_target, _ = self.graph_builder(
            X_target[:, :, :4, :], C, custom_mask_2D=custom_mask_2D
        )
        edge_idx = torch.cat([edge_idx_model, edge_idx_target], 2)

        # Build frame-aligned displacement vectors (B,N,K,A,3)
        r_ij = self._frame_ij(X, edge_idx)
        r_ij_alt = self._frame_ij(X_alt, edge_idx)
        r_ij_target = self._frame_ij(X_target, edge_idx)

        # Build 2D masks (B,N,K,A)
        num_batch, num_residues, num_atoms, _ = X.shape
        mask_residues = (C > 0).float()
        # (B,N,1,A)
        mask_i = mask_atoms.reshape([num_batch, num_residues, 1, num_atoms])
        # (B,N,K,1)
        mask_j = graph.collect_neighbors(mask_residues.unsqueeze(-1), edge_idx)
        mask_ij = mask_i * mask_j

        # Build frame-aligned displacement vectors (B,N,N,A)
        D = mask_ij * self._dist(r_ij, r_ij_target)
        D_alt = mask_ij * self._dist(r_ij_alt, r_ij_target)

        # Which definition of atom j gives a better score? (B,N)
        mask_reduce = mask_ij.sum([-2, -1])
        D_j = D.sum([-2, -1]) / (mask_reduce + self.distance_eps)
        D_j_alt = D_alt.sum([-2, -1]) / (mask_reduce + self.distance_eps)
        D_j_min = torch.stack([D_j, D_j_alt], -1).min(-1)[0]

        # Return as a per-residue loss
        return D_j_min


class LossAllAtomDistances(nn.Module):
    """Compute the interatomic distance loss on a nearest neighbors graph.

    Args:
        num_neighbors (int): Number of neighbors to build in the graph. Default
            is 30.

    Inputs:
        X (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        X_target (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        D (tensor): Per-residue losses with shape `(batch_size, num_residues)`.
    """

    def __init__(self, num_neighbors=30, distance_eps=1e-2):
        super(LossAllAtomDistances, self).__init__()
        self.distance_eps = distance_eps

        self.graph_builder = protein_graph.ProteinGraph(num_neighbors)

    def _dist_ij(self, X, edge_idx):
        # Build local frames
        num_batch, num_residues, num_atoms, _ = X.shape

        # Build frames at neighbor j (B,L,K,), (B,L,K,A,3)
        X_flat = X.reshape([num_batch, num_residues, -1])
        X_j_flat = graph.collect_neighbors(X_flat, edge_idx)
        X_j = X_j_flat.reshape([num_batch, num_residues, -1, num_atoms, 3])
        X_i = X.unsqueeze(2).expand([-1, -1, X_j.shape[2], -1, -1])

        X_ij = torch.cat([X_i, X_j], -2)
        D_ij = torch.sqrt(
            ((X_ij.unsqueeze(-2) - X_ij.unsqueeze(-3)) ** 2).sum(-1) + self.distance_eps
        )
        return D_ij

    def _mask_ij(self, C, S, edge_idx):
        # (B,L,A)
        mask_atoms = atom_mask(C, S)

        mask_j = graph.collect_neighbors(mask_atoms, edge_idx)
        mask_i = mask_atoms.unsqueeze(2).expand([-1, -1, edge_idx.shape[2], -1])
        mask_ij = torch.cat([mask_i, mask_j], -1)

        mask_D = mask_ij.unsqueeze(-1) * mask_ij.unsqueeze(-2)
        return mask_D

    def forward(self, X, X_target, C, S):
        # Build the union graph
        edge_idx_model, _ = self.graph_builder(X[:, :, :4, :], C)
        edge_idx_target, _ = self.graph_builder(X_target[:, :, :4, :], C)
        edge_idx = torch.cat([edge_idx_model, edge_idx_target], 2)

        mask_ij = self._mask_ij(C, S, edge_idx)
        D_model = self._dist_ij(X, edge_idx)
        D_target = self._dist_ij(X_target, edge_idx)

        loss = torch.sqrt((D_model - D_target) ** 2 + self.distance_eps)
        loss_i = (mask_ij * loss).sum([2, 3, 4]) / (
            mask_ij.sum([2, 3, 4]) + self.distance_eps
        )
        return loss_i


class LossSidechainClashes(nn.Module):
    """Count sidechain clashes in a structure using a nearest neighbors graph.

    This uses the Van der Waals radii based definition of bonding
    in pymol as described at https://pymolwiki.org/index.php/Connect_cutoff.

    Args:
        num_neighbors (int, optional): Number of neighbors to
            build in the graph. Default is 30.
        connect_cutoff (float, optional): Bonding cutoff used in formula
            `D_clash_cutoff = D_vdw / 2. + self.connect_cutoff`. Default is
            0.35.
        use_smooth_cutoff (bool, optional): If True, use a differentiable
            definition of clashes by replacing `D < cutoff` with
            `sigmoid(smooth_cutoff_alpha * (cutoff - D))`. Default is False.
        smooth_cutoff_alpha (float, optional): Steepness parameter for
            differentiable clashes, as `alpha -> infinity` it will behave as
            discrete cutoff. Default is 1.0.

    Inputs:
        X (tensor): Atomic coordinates with shape
            `(batch_size, num_residues, 14, 3)`.
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.
        mask_j (tensor, optional): Binary mask encoding which side chains
            should be tested for clashing.

    Outputs:
        clashes (tensor): Per-residue number of clashes with shape
            `(batch_size, num_residues)`.
    """

    def __init__(
        self,
        num_neighbors=30,
        distance_eps=1e-3,
        connect_cutoff=0.35,
        use_smooth_cutoff=False,
        smooth_cutoff_alpha=1.0,
    ):
        super(LossSidechainClashes, self).__init__()
        self.distance_eps = distance_eps
        self.graph_builder = protein_graph.ProteinGraph(num_neighbors)
        self.connect_cutoff = connect_cutoff
        self.use_smooth_cutoff = use_smooth_cutoff
        self.smooth_cutoff_alpha = smooth_cutoff_alpha

    def _dist_ij(self, X, edge_idx):
        num_batch, num_residues, num_atoms, _ = X.shape

        # Build frames at neighbor j (B,L,K,), (B,L,K,A,3)
        X_flat = X.reshape([num_batch, num_residues, -1])
        X_j_flat = graph.collect_neighbors(X_flat, edge_idx)
        X_j = X_j_flat.reshape([num_batch, num_residues, -1, num_atoms, 3])
        X_i = X.unsqueeze(2).expand([-1, -1, X_j.shape[2], -1, -1])

        D_ij = torch.sqrt(
            ((X_i.unsqueeze(-2) - X_j.unsqueeze(-3)) ** 2).sum(-1) + self.distance_eps
        )
        return D_ij

    def _mask_ij(self, C, S, edge_idx, mask_j=None):
        # (B,L,A)
        mask_atoms = atom_mask(C, S)

        # Mask only present atoms
        mask_atoms_j = mask_atoms
        if mask_j is not None:
            mask_atoms_j = mask_atoms_j * mask_j.unsqueeze(-1)
        mask_j = graph.collect_neighbors(mask_atoms_j, edge_idx)
        mask_i = mask_atoms.unsqueeze(2).expand([-1, -1, edge_idx.shape[2], -1])
        mask_D = mask_i.unsqueeze(-1) * mask_j.unsqueeze(-2)

        # Mask self interactions
        node_idx = torch.arange(C.shape[1], device=C.device).reshape([1, -1, 1])
        mask_ne = torch.ne(edge_idx, node_idx)
        mask_D = mask_D * mask_ne.reshape(list(mask_ne.shape) + [1, 1])
        return mask_D

    def _gather_vdw_radii(self, C, S):
        vdw_radii = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}

        # Van der waal radii per atom per residue [AA,ATOM]
        R = torch.zeros([20, 14], device=C.device)
        for i, aa in enumerate(constants.AA20_3):
            atoms = constants.ATOMS_BB + constants.AA_GEOMETRY[aa]["atoms"]
            for j, atom_name in enumerate(atoms):
                R[i, j] = vdw_radii[atom_name[0]]

        # (B, AA, ATOM) @ (B, L, ATOM) => (B, L, ATOM)
        R = R.reshape([1, 20, 14]).expand([C.shape[0], -1, -1])
        S = S.unsqueeze(-1).expand([-1, -1, 14])
        atom_radii = torch.gather(R, 1, S)
        return atom_radii

    def _gather_vdw_diameters(self, C, S, edge_idx):
        num_batch, num_residues, num_neighbors = edge_idx.shape

        # Gather van der Waals radii
        radii_i = self._gather_vdw_radii(C, S)
        radii_j = graph.collect_neighbors(radii_i, edge_idx)
        radii_i = radii_i.reshape([num_batch, num_residues, 1, -1, 1])
        radii_j = radii_j.reshape([num_batch, num_residues, num_neighbors, 1, -1])

        D_vdw = radii_i + radii_j
        return D_vdw

    def forward(self, X, C, S, edge_idx=None, mask_j=None, mask_ij=None):
        # Compute sidechain interatomic distances
        if edge_idx is None:
            edge_idx, mask_ij = self.graph_builder(X[:, :, :4, :], C)

        # Distance with shape [B,L,K,AI,AJ]
        mask_clash_ij = self._mask_ij(C, S, edge_idx, mask_j)
        if mask_ij is not None:
            mask_clash_ij = mask_clash_ij * mask_ij.reshape(
                list(mask_ij.shape) + [1, 1]
            )
        D = self._dist_ij(X, edge_idx)
        D_vdw = self._gather_vdw_diameters(C, S, edge_idx)
        D_clash_cutoff = D_vdw / 2.0 + self.connect_cutoff

        # Optionally use a smooth definition of clashes that is differentiable
        if self.use_smooth_cutoff:
            bond_clash = mask_clash_ij * torch.sigmoid(
                self.smooth_cutoff_alpha * (D_clash_cutoff - D)
            )
        else:
            bond_clash = mask_clash_ij * (D < D_clash_cutoff).float()

        # Only cound outgoing clashes from sidechain atoms at i
        bond_clash = bond_clash[:, :, :, 4:, :]

        clashes = bond_clash.sum([2, 3, 4])
        return clashes


def _gather_atom_mask(C, S, atoms_per_aa, num_atoms):
    device = S.device
    atoms_per_aa = torch.tensor(atoms_per_aa, dtype=torch.long)
    atoms_per_aa = atoms_per_aa.to(device).unsqueeze(0).expand(S.shape[0], -1)

    # (B,A) @ (B,L)  => (B,L)
    atoms_per_residue = torch.gather(atoms_per_aa, -1, S)
    atoms_per_residue = (C > 0).float() * atoms_per_residue

    ix_expand = torch.arange(num_atoms, device=device).reshape([1, 1, -1])
    mask_atoms = ix_expand < atoms_per_residue.unsqueeze(-1)
    mask_atoms = mask_atoms.float()
    return mask_atoms


def atom_mask(C, S):
    """Constructs a all-atom coordinate mask from a sequence and chain map.

    Inputs:
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        mask_atoms (tensor): Atomic mask with shape
            `(batch_size, num_residues, 14)`.
    """
    return _gather_atom_mask(C, S, constants.AA20_NUM_ATOMS, 14)


def chi_mask(C, S):
    """Constructs a all-atom coordinate mask from a sequence and chain map.

    Inputs:
        C (tensor): Chain map with shape `(batch_size, num_residues)`.
        S (tensor): Sequence tokens with shape `(batch_size, num_residues)`.

    Outputs:
        mask_atoms (tensor): Chi angle mask with shape
            `(batch_size, num_residues, 4)`.
    """
    return _gather_atom_mask(C, S, constants.AA20_NUM_CHI, 4)
