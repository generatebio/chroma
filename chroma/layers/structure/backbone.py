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

"""Layers for generating protein structure.

This module contains pytorch layers for parametrically generating and
manipulating protein backbones. These can be used in tandem with loss functions
to generate and optimize protein structure (e.g. folding from predictions) or
used as an intermediate layer in a learned structure generation model.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chroma.layers.structure import geometry, transforms


class ProteinBackbone(nn.Module):
    """Protein backbone layer with optimizable geometry (batch form).

    This layer stores the parameters for a protein backbone, which can be based
    on either internal coordinate or Cartesian parameterizations.
    It outputs coordinates in Cartesian form as 4D tensors with indices
    `[batch, position, atom_type, xyz]`. The `atom_type` index runs over the
    heavy atoms of a protein backbone in PDB order, i.e. `[N, CA, C, O]`.
    The resulting coordinates can be directly optimized with pytorch optimizers.

    Args:
        num_residues (int): Number of residues.
        num_batch (int): Batch size.
        init_state (str): Initialization state. Can be ['alpha', 'beta', '']
        use_internal_coords (Boolean): Use a phi,psi parameterization.
            Default is True.
        X_init (torch.Tensor, optional): Initialize with pre-specified coordinates.
            Requires that use_internal_coords=False.

    Outputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(batch_size, num_residues, 4, 3)`.
    """

    def __init__(
        self,
        num_residues: int,
        num_batch: int = 1,
        init_state: str = "alpha",
        use_internal_coords: bool = True,
        X_init: Optional[torch.Tensor] = None,
    ):
        super(ProteinBackbone, self).__init__()

        # Dimensions
        self.num_batch = num_batch
        self.num_residues = num_residues

        # Rigid body translation and rotation
        self.transform = RigidTransform(num_batch=num_batch, keep_centered=True)

        self.use_internal_coords = use_internal_coords
        if self.use_internal_coords:
            # Internal coordinate parameterization
            self.phi = nn.Parameter(torch.zeros(num_batch, num_residues))
            self.psi = nn.Parameter(torch.zeros(num_batch, num_residues))

            # Initializer
            phi_psi = {
                "alpha": (np.radians(-60.0), np.radians(-45.0)),
                "beta": (np.radians(-140.0), np.radians(135.0)),
            }
            if init_state in phi_psi:
                torch.nn.init.constant_(self.phi, phi_psi[init_state][0])
                torch.nn.init.constant_(self.psi, phi_psi[init_state][1])
            else:
                torch.nn.init.uniform_(self.phi, a=-np.pi, b=np.pi)
                torch.nn.init.uniform_(self.psi, a=-np.pi, b=np.pi)

            self.backbone_geometry = BackboneBuilder()
        else:
            # Use a Cartesian parameterization
            if X_init is not None:
                assert not use_internal_coords
            else:
                X_init = ProteinBackbone(
                    num_residues=num_residues,
                    num_batch=num_batch,
                    init_state=init_state,
                    use_internal_coords=True,
                )()
            self.X = nn.Parameter(X_init)

    def forward(self) -> torch.Tensor:
        if self.use_internal_coords:
            X = self.backbone_geometry(self.phi, self.psi)
        else:
            X = self.X

        # Apply rotation and translation
        X = self.transform(X)
        return X


class RigidTransform(nn.Module):
    """Rigid-body rotation and translation (batch form).

    This layer stores the parameters for a rigid body rotation and translation.
    It can be composed with other generative geometry layers to optimize over
    poses.

    Args:
        num_batch (int): Number of poses to store parameters for.
        keep_centered (Boolean): If True, center the input coordinates by
            default.
        scale_dX (float): Scale factor which affects the rate of change of
            translation.
        scale_q (float): Scale factor which affects the rate of change of
            rotation.

    Inputs:
        X (torch.Tensor): Input coordinates with shape `(batch_size, ..., 3)`.

    Outputs:
        X_t (torch.Tensor): Transformed coordinates with shape:
            `(batch_size, ..., 3)`.
    """

    def __init__(
        self,
        num_batch: int = 1,
        keep_centered: bool = False,
        scale_dX: float = 1.0,
        scale_q: float = 1.0,
    ):
        super(RigidTransform, self).__init__()
        self.num_batch = num_batch

        # Cartesian offset initialized to 0
        self.dX = nn.Parameter(torch.zeros(self.num_batch, 3))
        self.scale_dX = scale_dX

        # Unconstrained quaternion initialized to identity
        self.scale_q = scale_q
        q_init = np.asarray([[1.0, 0, 0, 0]] * self.num_batch)
        q_init = torch.tensor(q_init, dtype=torch.float32) / self.scale_q
        self.q_unc = nn.Parameter(q_init)

        self.rigid_transform = RigidTransformer(keep_centered=keep_centered)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dX = self.scale_dX * self.dX
        q_unc = self.scale_q * self.q_unc
        X_t = self.rigid_transform(X, dX, q_unc)
        return X_t


class RigidTransformer(nn.Module):
    """Rigid-body rotation and translation (batch form).

    This layer applies a rigid body rotation and translation,
    and can be composed with other generative geometry layers to modify poses.

    Internally, the coordinates are centered before rotation and translation.
    The rotation itself is parameterized as a quaternion to prevent
    Gimbal lock (https://en.wikipedia.org/wiki/Gimbal_lock).

    Args:
        center_intput (Boolean): Center the input coordinates (default: True)
            default.

    Inputs:
        X (torch.Tensor): Input coordinates with shape `(batch_size, ..., 3)`.
        dX (torch.Tensor): Translation vector with shape `(batch_size, 3)`.
        q (torch.Tensor): Rotation vector (quaternion) with shape `(batch_size, 4)`.
            It can be any 4-element real vector, but will internally be
            normalized to a unit quaternion.
        mask (tensor,optional): Mask tensor with shape `(batch_size, ..., 3)`.

    Outputs:
        X_t (torch.Tensor): Transformed coordinates with shape `(batch_size, ..., 3)`.
    """

    def __init__(self, center_rotation: bool = True, keep_centered: bool = False):
        super(RigidTransformer, self).__init__()
        self.center_rotation = center_rotation
        self.keep_centered = keep_centered
        self.dist_eps = 1e-5

    def _rotation_matrix(self, q_unc: torch.Tensor) -> torch.Tensor:
        """Build rotation matrix from quaternion parameters.

        See en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for further
        details on converting between quaternions and rotation matrices.

        Args:
            q_unc (torch.Tensor): Unnormalized quaternion representing rotation with
                shape `(batch_size, 3)`.

        Returns:
            R (torch.Tensor): Rotation matrix with shape `(batch_size, 3)`.
        """
        num_batch = q_unc.shape[0]
        q = F.normalize(q_unc, dim=-1)

        # fmt: off
        a,b,c,d = q.unbind(-1)
        a2,b2,c2,d2 = a**2, b**2, c**2, d**2
        R = torch.stack([
            a2 + b2 - c2 - d2,      2*b*c - 2*a*d,      2*b*d + 2*a*c,
                2*b*c + 2*a*d,  a2 - b2 + c2 - d2,      2*c*d - 2*a*b,
                2*b*d - 2*a*c,      2*c*d + 2*a*b,  a2 - b2 - c2 + d2
        ], dim=-1)
        # fmt: on

        R = R.view([num_batch, 3, 3])
        return R

    def forward(
        self,
        X: torch.Tensor,
        dX: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_batch = X.shape[0]
        X_flat = X.reshape([num_batch, -1, 3])

        # Flatten mask
        if mask is not None:
            shape_mask = list(mask.size())
            shape_X = list(X.size())
            shape_mask_expand = shape_mask + [
                1 for i in range(len(shape_X) - 1 - len(shape_mask))
            ]
            shape_mask_flat = list(X_flat.size())[:-1] + [1]

            mask_flat = mask.reshape(shape_mask_expand).expand(shape_X[:-1])
            mask_flat = mask_flat.reshape(shape_mask_flat)

            # Compute center
            X_mean = torch.sum(mask_flat * X_flat, 1, keepdims=True) / (
                torch.sum(mask_flat, 1, keepdims=True) + self.dist_eps
            )
        else:
            X_mean = torch.mean(X_flat, 1, keepdims=True)

        # Rotate around center of mass
        if self.center_rotation:
            X_centered = X_flat - X_mean
        else:
            X_centered = X_flat
        R = self._rotation_matrix(q)
        X_rotate = torch.einsum("bxr,bir->bix", R, X_centered)

        # Optionally preserve original centering
        if self.center_rotation and not self.keep_centered:
            X_rotate = X_rotate + X_mean

        # Translate
        X_transform = X_rotate + dX.unsqueeze(1)

        if mask is not None:
            X_transform = mask_flat * X_transform + (1 - mask_flat) * X_flat

        X_transform = X_transform.view(X.shape)
        return X_transform


class BackboneBuilder(nn.Module):
    """Protein backbone builder from dihedral angles (batch form).

    See ProteinBackbone() for further explanation of output coordinates.

    When only partial information is given such as phi & psi angles, this module
    will fall default to using the ideal geometries given in
        Engh & Huber, International Tables for Crystallography (2001).
        https://doi.org/10.1107/97809553602060000857

    Todo:
        * Add shifting and padding logic to associate phis and psis with their
            'natural' residue indices rather than the child atoms that they
            create during NERF recurrence
        * Add control over the bond lengths and angles for Oxygen

    Inputs:
        phi (torch.Tensor): Phi dihedral angles with shape `(batch_size, length)`.
        psi (torch.Tensor): Psi dihedral angles with shape `(batch_size, length)`.
        omega (torch.Tensor, optional): Omega dihedral angles with shape
            `(batch_size, length)`. Defaults to ideal geometry.
        angles (torch.Tensor, optional): Bond angles with shape
            `(batch_size, 3*length)` Defaults to ideal geometry.
        lengths (torch.Tensor, optional): Bond lengths with shape
            `(batch_size, 3*length)`. Defaults to ideal geometry.

    Outputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(batch_size, num_residues, 4, 3)`.
    """

    def __init__(self):
        super(BackboneBuilder, self).__init__()

        # From "Structure Quality and Target Parameters", Engh & Huber, 2001
        # fmt: off
        self.lengths = {
            'N_CA': 1.459,
            'CA_C': 1.525,
            'C_N': 1.336,
            'C_O': 1.229
        }
        angles = {
            'N_CA_C': 111.0,
            'CA_C_N': 117.2,
            'C_N_CA': 121.7,
            'omega': 179.3
        }
        self.angles = {
            k: v * np.pi / 180. for k,v in angles.items()
        }
        # fmt: on
        return

    def forward(
        self,
        phi: torch.Tensor,
        psi: torch.Tensor,
        omega: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        add_O: bool = True,
    ) -> torch.Tensor:
        N_batch, N_residues = phi.shape[0], phi.shape[1]
        linear_shape = [N_batch, N_residues]
        device = phi.device

        """
        This uses a similar (but not identical) approach as NERF:
            Parsons et al, Computational Chemistry (2005).
            https://doi.org/10.1002/jcc.20237
        See the reference for further explanation about converting from internal
        coordinates to Cartesian coordinates.
          ____________________________________________________________________
         |                 N-to-C backbone geometry for NERF                  |
         |      i.e. which internal coords create which Cartesian coords      |
         |               [% indicates preceding residue]                      |
         |______________________ _________________________________________ ___|
         |i-1                   |Residue i                                |i+1|
         |                      |                                         |   |
         |Atom:   [C%]--omega%--[N]----phi----[CA]----psi---[C]---omega---[N>]|
         |                       |             |             |                |
         |Parents                |             |             |                |
         |    Bond:           C%_N           N_CA         CA_C                |
         |   Angle:       CA%_C%_N        C%_N_CA       N_CA_C                |
         |Dihedral:    N%_CA%_C%_N    CA%_C%_N_CA    C%_N_CA_C                |
         |--------------------------------------------------------------------|
         |Bond:             [C_N]%          [N_CA]       [CA_C]               |
         |Dihedral:           psi%         omega%          phi                |
         |____________________________________________________________________|
        """

        if lengths is None:
            lengths = torch.tensor(
                [[self.lengths[key] for key in ["C_N", "N_CA", "CA_C"]]],
                dtype=torch.float32,
            ).to(device)
            lengths = lengths.repeat(N_batch, N_residues)

        if angles is None:
            angles = torch.tensor(
                [[self.angles[key] for key in ["CA_C_N", "C_N_CA", "N_CA_C"]]],
                dtype=torch.float32,
            ).to(device)
            angles = angles.repeat(N_batch, N_residues)

        if omega is None:
            omega = self.angles["omega"] * torch.ones(linear_shape).to(device)

        # Compute un-rotated Cartesian coordinates in batch
        dihedrals = torch.stack([psi, omega, phi], -1)
        dihedrals = dihedrals.view([N_batch, 3 * N_residues])
        angles_comp = np.pi - angles
        v = torch.stack(
            [
                torch.cos(angles_comp),
                torch.sin(angles_comp) * torch.cos(dihedrals),
                torch.sin(angles_comp) * torch.sin(dihedrals),
            ],
            -1,
        )

        # Lengths
        lengths_list = list(lengths.unsqueeze(-1).unbind(1))
        v_list = list(v.unbind(1))

        if add_O:
            # Build one extra appended residue
            lengths_list += lengths_list[-3:]
            v_list += v_list[-3:]

        def _build_x_i(v_i, l_i, x, u_minus_1, u_minus_2):
            """Recurrence relation for placing atoms (NERF)"""

            # Build matrix encoding local reference frame
            n_a_unnorm = torch.cross(u_minus_2, u_minus_1)
            n_a = F.normalize(n_a_unnorm, dim=-1)
            n_b = torch.cross(n_a, u_minus_1)

            # Matrix multiply version
            R = torch.stack([u_minus_1, n_b, n_a], 2)
            u_new = torch.matmul(R, v_i.unsqueeze(-1)).squeeze(-1)

            x_new = x + l_i * u_new
            return x_new, u_new, u_minus_1

        # Initialization
        x_i = torch.zeros([N_batch, 3]).to(device)
        u_i_minus_2 = torch.tensor([[1.0, 0, 0]] * N_batch, dtype=torch.float32).to(
            device
        )
        u_i_minus_1 = torch.tensor([[0, 1.0, 0]] * N_batch, dtype=torch.float32).to(
            device
        )

        # Build chain via NERF recurrence
        X = []
        for i, (v_i, l_i) in enumerate(zip(v_list, lengths_list)):
            x_i, u_i_minus_1, u_i_minus_2 = _build_x_i(
                v_i, l_i, x_i, u_i_minus_1, u_i_minus_2
            )
            X.append(x_i)
        X = torch.stack(X, 1)
        # [N,AL,3] => [N,L,A,3]
        X = X.view([N_batch, -1, 3, 3])

        if add_O:
            # Build the oxygen vector using symmetry
            u_1 = F.normalize(X[:, :-1, 2, :] - X[:, :-1, 1, :], dim=-1)  # CA->C
            u_2 = F.normalize(X[:, :-1, 2, :] - X[:, 1:, 0, :], dim=-1)  # C<-N*
            u = self.lengths["C_O"] * F.normalize(u_1 + u_2, dim=-1)
            X = X[:, :-1, :, :]
            X_O = X[:, :, 2, :] + u
            X = torch.cat([X, X_O.unsqueeze(2)], 2)

        X = X - X.mean([1, 2, 3], keepdim=True)
        return X


class FrameBuilder(nn.Module):
    """Build protein backbones from rigid residue poses.

    Inputs:
        R (torch.Tensor): Rotation of residue orientiations
            with shape `(num_batch, num_residues, 3, 3)`. If `None`,
            then `q` must be provided instead.
        t (torch.Tensor): Translation of residue orientiations
            with shape `(num_batch, num_residues, 3)`. This is the
            location of the C-alpha coordinates.
        C (torch.Tensor): Chain map with shape `(num_batch, num_residues)`.
        q (Tensor, optional): Quaternions representing residue orientiations
            with shape `(num_batch, num_residues, 4)`.

    Outputs:
        X (torch.Tensor): All-atom protein coordinates with shape
            `(num_batch, num_residues, 4, 3)`
    """

    def __init__(self, distance_eps: float = 1e-3):
        super().__init__()

        # Build idealized backbone fragment
        t = torch.tensor(
            [
                [1.459, 0.0, 0.0],  # N-C via Engh & Huber is 1.459
                [0.0, 0.0, 0.0],  # CA is origin
                [-0.547, 0.0, -1.424],  # C is placed 1.525 A @ 111 degrees from N
            ],
            dtype=torch.float32,
        ).reshape([1, 1, 3, 3])
        R = torch.eye(3).reshape([1, 1, 1, 3, 3])
        self.register_buffer("_t_atom", t)
        self.register_buffer("_R_atom", R)

        # Carbonyl geometry from CHARMM all36_prot ALA definition
        self._length_C_O = 1.2297
        self._angle_CA_C_O = 122.5200
        self._dihedral_Np_CA_C_O = 180
        self.distance_eps = distance_eps

    def _build_O(self, X_chain: torch.Tensor, C: torch.LongTensor):
        """Build backbone carbonyl oxygen."""
        # Build carboxyl groups
        X_N, X_CA, X_C = X_chain.unbind(-2)

        # TODO: fix this behavior for termini
        mask_next = (C > 0).float()[:, 1:].unsqueeze(-1)
        X_N_next = F.pad(mask_next * X_N[:, 1:,], (0, 0, 0, 1),)

        num_batch, num_residues = C.shape
        ones = torch.ones(list(C.shape), dtype=torch.float32, device=C.device)
        X_O = geometry.extend_atoms(
            X_N_next,
            X_CA,
            X_C,
            self._length_C_O * ones,
            self._angle_CA_C_O * ones,
            self._dihedral_Np_CA_C_O * ones,
            degrees=True,
        )
        mask = (C > 0).float().reshape(list(C.shape) + [1, 1])
        X = mask * torch.stack([X_N, X_CA, X_C, X_O], dim=-2)
        return X

    def forward(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        C: torch.LongTensor,
        q: Optional[torch.Tensor] = None,
    ):
        assert q is None or R is None

        if R is None:
            # (B,N,1,3,3) and (B,N,1,3)
            R = geometry.rotations_from_quaternions(
                q, normalize=True, eps=self.distance_eps
            )

        R = R.unsqueeze(-3)
        t_frame = t.unsqueeze(-2)
        X_chain = transforms.compose_translation(R, t_frame, self._t_atom)
        X = self._build_O(X_chain, C)
        return X

    def inverse(
        self, X: torch.Tensor, C: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct transformations from poses.

        Inputs:
            X (torch.Tensor): All-atom protein coordinates with shape
                `(num_batch, num_residues, 4, 3)`
            C (torch.Tensor): Chain map with shape `(num_batch, num_residues)`.

        Outputs:
            R (torch.Tensor): Rotation of residue orientiations
                with shape `(num_batch, num_residues, 3, 3)`.
            t (torch.Tensor): Translation of residue orientiations
                with shape `(num_batch, num_residues, 3)`. This is the
                location of the C-alpha coordinates.
            q (torch.Tensor): Quaternions representing residue orientiations
                with shape `(num_batch, num_residues, 4)`.
        """
        X_bb = X[:, :, :4, :]
        R, t = geometry.frames_from_backbone(X_bb, distance_eps=self.distance_eps)
        q = geometry.quaternions_from_rotations(R, eps=self.distance_eps)
        mask = (C > 0).float().unsqueeze(-1)
        R = mask.unsqueeze(-1) * R
        t = mask * t
        q = mask * q
        return R, t, q


class GraphBackboneUpdate(nn.Module):
    """Layer for updating backbone coordinates given graph embeddings.

    Args:
        dim_nodes (int): Node dimension of graph input.
        dim_edges (int): Edge dimension of graph input.
        distance_scale (float): Coordinate scaling factor in angstroms. Default
            is 10 angstroms per unit, which corresponds to nanometers.
        method (str): Method used for predicting coordinates. Options include
            * `local`: Node-based relative transformations.
            * `neighbor`: Inter-residue geometry.
            * `neighbor_global`: Inter-residue geometry with virtual global edge.
            * `neighbor_global_affine`: Inter-residue geometry with virtual
                global edge, parameterized as a residual update.
            * `none`: No transformation-based updates.
        iterations (int): Number of method iteractions.
        unconstrained (bool): If True, update sub-pose geometries beyond ideal
            coordinates.
        num_transform_weights (int): Number of uncertainty dimensions per residue
            for neighbor-based updates.
        black_hole_init (bool): If True, ignore initial geometry and initialize
            poses at the origin as in AlphaFold2 (Jumper et al 2020).

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_residues, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_residues)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.

    Outputs:
        X_update (torch.Tensor): Updated backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        distance_scale: float = 10.0,
        distance_eps: float = 1e-3,
        method: str = "neighbor",
        iterations: int = 1,
        unconstrained: bool = True,
        num_transform_weights: int = 1,
        black_hole_init: bool = False,
    ):
        super(GraphBackboneUpdate, self).__init__()
        self.distance_scale = distance_scale
        self.distance_eps = distance_eps
        self._eps = 1e-5

        self.frame_builder = FrameBuilder(distance_eps=distance_eps)
        self.method = method
        self.iterations = iterations
        self.unconstrained = unconstrained
        self.num_transform_weights = num_transform_weights
        self.black_hole_init = black_hole_init

        if self.method == "local":
            self.W_q = nn.Linear(dim_nodes, 4)
            self.W_t = nn.Linear(dim_nodes, 3)
        elif self.method == "neighbor":
            self.W_q = nn.Linear(dim_edges, 4)
            self.W_t = nn.Linear(dim_edges, 3)
            self.W_w = nn.Linear(dim_edges, self.num_transform_weights)
        elif self.method == "neighbor_global":
            self.W_q = nn.Linear(dim_edges, 4)
            self.W_t = nn.Linear(dim_edges, 3)
            self.W_w = nn.Linear(dim_edges, self.num_transform_weights)
            self.W_q_global = nn.Linear(dim_nodes, 4)
            self.W_t_global = nn.Linear(dim_nodes, 3)
            self.W_w_global = nn.Linear(dim_nodes, self.num_transform_weights)
        elif self.method == "neighbor_global_affine":
            self.W_s_node = nn.Linear(dim_nodes, 2)
            self.W_s_edge = nn.Linear(dim_edges, 2)
            self.W_q = nn.Linear(dim_edges, 4)
            self.W_t = nn.Linear(dim_edges, 3)
            self.W_w = nn.Linear(dim_edges, self.num_transform_weights)
            self.W_q_global = nn.Linear(dim_nodes, 4)
            self.W_t_global = nn.Linear(dim_nodes, 3)
            self.W_w_global = nn.Linear(dim_nodes, self.num_transform_weights)
        if self.method == "none":
            # None does no frame based updates
            assert self.unconstrained

        if self.unconstrained:
            self.W_t_local = nn.Linear(dim_nodes, 12)
        return

    def _init_black_hole(self, X):
        R = (
            torch.eye(3, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 3, 3)
            .repeat(X.size(0), X.size(1), 1, 1)
        )
        t = torch.zeros(X.size(0), X.size(1), 3, dtype=X.dtype, device=X.device)
        return R, t

    def _update_local_transform(self, X, C, node_h, edge_h, edge_idx, mask_i, mask_ij):
        """Update residue frames via transformation from self."""
        R_i, t_i, _ = self.frame_builder.inverse(X, C)
        if self.black_hole_init:
            R_i, t_i = self._init_black_hole(X)

        # Predict transforms
        R = geometry.rotations_from_quaternions(
            self.W_q(node_h), normalize=True, eps=self.distance_eps
        )
        t = self.distance_scale * self.W_t(node_h)

        # Apply transformations
        R_i_pred, t_i_pred = transforms.compose_transforms(R_i, t_i, R, t)
        X_update = self.frame_builder(R_i_pred, t_i_pred, C)
        return X_update, None, None, None

    def _update_neighbor_transform(
        self, X, C, node_h, edge_h, edge_idx, mask_i, mask_ij
    ):
        """Update residue frames via weighted average transformation from neighbors."""

        # Predict relative transformations from neighbors to self
        R_ji = geometry.rotations_from_quaternions(
            self.W_q(edge_h), normalize=True, eps=self.distance_eps
        )
        t_ji = self.distance_scale * self.W_t(edge_h)
        logit_ij = self.W_w(edge_h)

        # Compute predicted self locations from each neighbor
        R_i, t_i, _ = self.frame_builder.inverse(X, C)
        if self.black_hole_init:
            R_i, t_i = self._init_black_hole(X)

        R_i, t_i = transforms.equilibrate_transforms(
            R_i,
            t_i,
            R_ji,
            t_ji,
            logit_ij,
            mask_ij,
            edge_idx,
            iterations=self.iterations,
        )
        X_update = self.frame_builder(R_i, t_i, C)

        return X_update, R_ji, t_ji, None

    def _update_neighbor_global_transform(
        self, X, C, node_h, edge_h, edge_idx, mask_i, mask_ij
    ):
        """Update residue frames via weighted average transformation from neighbors."""

        # Predict relative transformations from neighbors to self
        R_ji = geometry.rotations_from_quaternions(
            self.W_q(edge_h), normalize=True, eps=self.distance_eps
        )
        t_ji = self.distance_scale * self.W_t(edge_h)
        logit_ji = self.W_w(edge_h)

        # Predict relative transformations to global frame
        R_global_i = geometry.rotations_from_quaternions(
            self.W_q_global(node_h), normalize=True, eps=self.distance_eps
        )
        t_global_i = self.distance_scale * self.W_t_global(node_h)
        logit_global_i = self.W_w_global(node_h)

        # Initialize global frame equivariantly
        R_i, t_i, _ = self.frame_builder.inverse(X, C)
        if self.black_hole_init:
            R_i, t_i = self._init_black_hole(X)

        R_global, t_global = transforms.average_transforms(
            R_i, t_i, mask_i[..., None], mask_i, dim=1, dither_eps=0.0
        )

        # Compute predicted self locations from each neighbor
        R_i, t_i = transforms.equilibrate_transforms(
            R_i,
            t_i,
            R_ji,
            t_ji,
            logit_ji,
            mask_ij,
            edge_idx,
            iterations=self.iterations,
            R_global=R_global,
            t_global=t_global,
            R_global_i=R_global_i,
            t_global_i=t_global_i,
            logit_global_i=logit_global_i,
        )
        X_update = self.frame_builder(R_i, t_i, C)

        return X_update, R_ji, t_ji, logit_ji

    def _update_neighbor_global_affine_transform(
        self, X, C, node_h, edge_h, edge_idx, mask_i, mask_ij
    ):
        """Update residue frames via weighted average transformation from neighbors."""

        # Compute interresidue geometries for current system
        R_i_init, t_i_init, _ = self.frame_builder.inverse(X, C)
        if self.black_hole_init:
            R_i_init, t_i_init = self._init_black_hole(X)

        R_j_init, t_j_init = transforms.collect_neighbor_transforms(
            R_i_init, t_i_init, edge_idx
        )
        R_global, t_global = transforms.average_transforms(
            R_i_init, t_i_init, mask_i[..., None], mask_i, dim=1, dither_eps=0.0
        )
        R_ji_init, t_ji_init = transforms.compose_inner_transforms(
            R_j_init, t_j_init, R_i_init.unsqueeze(-3), t_i_init.unsqueeze(-2)
        )
        R_gi_init, t_gi_init = transforms.compose_inner_transforms(
            R_global.unsqueeze(1), t_global.unsqueeze(1), R_i_init, t_i_init
        )
        q_ji_init = geometry.quaternions_from_rotations(R_ji_init)
        q_gi_init = geometry.quaternions_from_rotations(R_gi_init)

        # Scale factor
        s_node = torch.sigmoid(self.W_s_node(node_h)[..., None]).unbind(-2)
        s_edge = torch.sigmoid(self.W_s_edge(edge_h)[..., None]).unbind(-2)
        d_scale = self.distance_scale

        # Use edges to predict relative transformations from neighbors to self
        q_ji = s_edge[0] * q_ji_init + (1.0 - s_edge[0]) * self.W_q(edge_h)
        t_ji = s_edge[1] * t_ji_init + (1.0 - s_edge[1]) * d_scale * self.W_t(edge_h)
        logit_ji = self.W_w(edge_h)

        # Use nodes to predict relative transformations to global frame
        q_gi = s_node[0] * q_gi_init + (1.0 - s_node[0]) * self.W_q_global(node_h)
        t_gi = s_node[1] * t_gi_init + (1.0 - s_node[1]) * d_scale * self.W_t_global(
            node_h
        )
        logit_gi = self.W_w_global(node_h)

        R_ji = geometry.rotations_from_quaternions(
            q_ji, normalize=True, eps=self.distance_eps
        )
        R_gi = geometry.rotations_from_quaternions(
            q_gi, normalize=True, eps=self.distance_eps
        )

        # Compute predicted self locations from each neighbor
        R_i, t_i = transforms.equilibrate_transforms(
            R_i_init,
            t_i_init,
            R_ji,
            t_ji,
            logit_ji,
            mask_ij,
            edge_idx,
            iterations=self.iterations,
            R_global=R_global,
            t_global=t_global,
            R_global_i=R_gi,
            t_global_i=t_gi,
            logit_global_i=logit_gi,
        )
        X_update = self.frame_builder(R_i, t_i, C)
        return X_update, R_ji, t_ji, logit_ji

    def _inner_transforms(self, X, C, edge_idx):
        R_i, t_i, _ = self.frame_builder.inverse(X, C)
        R_ji, t_ji = transforms.collect_neighbor_inner_transforms(R_i, t_i, edge_idx)
        return R_ji, t_ji

    def _transform_loss(self, R_ij_predict, t_ij_predict, X, C, edge_idx, mask_ij):
        """Compute loss between transforms"""
        R_ij, t_ij = self._inner_transforms(X, C, edge_idx)
        R_ij_error = (R_ij_predict - R_ij).square().sum([-1, -2])
        t_ij_error = (t_ij_predict - t_ij).square().sum([-1])
        R_ij_mse = (mask_ij * R_ij_error).sum([1, 2]) / (
            mask_ij.sum([1, 2]) + self._eps
        )
        t_ij_mse = (mask_ij * t_ij_error).sum([1, 2]) / (
            mask_ij.sum([1, 2]) + self._eps
        )
        return R_ij_mse, t_ij_mse

    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        methods = {
            "local": self._update_local_transform,
            "neighbor": self._update_neighbor_transform,
            "neighbor_global": self._update_neighbor_global_transform,
            "neighbor_global_affine": self._update_neighbor_global_affine_transform,
            "none": lambda *args: args[0],
        }
        method = methods[self.method]

        # Update frames with ideal geometry
        X_update, R_ji, t_ji, logit_ji = method(
            X, C, node_h, edge_h, edge_idx, mask_i, mask_ij
        )

        if self.unconstrained:
            # Predict atomic updates as delta from ideal geometry
            # R_i, t_i, _ = self.frame_builder.inverse(X, C) # NOTE: Old models did this which was a typo
            R_i, t_i, _ = self.frame_builder.inverse(X_update, C)
            t_local = self.W_t_local(node_h).reshape(X.shape)

            # Rotate atomic updates into local frame
            R_i = R_i.unsqueeze(-3)
            t_i = torch.zeros_like(t_i).unsqueeze(-2)
            dX = transforms.compose_translation(R_i, t_i, t_local)

            if self.training:
                # Randomly swap between ideal coordinates at train time
                mask_drop = (
                    torch.randint(
                        low=0, high=2, size=[C.shape[0], 1, 1, 1], device=X.device
                    )
                    > 0
                ).float()
                dX = mask_drop * dX

            X_update = X_update + dX
        return X_update, R_ji, t_ji, logit_ji


class LossBackboneResidueDistance(nn.Module):
    """Compute losses for training denoising diffusion models.

    Inputs:
        X_mobile (torch.Tensor): Mobile coordinates with shape
            `(num_source, num_atoms, 4, 3)`.
        X_target (torch.Tensor): Target coordinates with shape
            `(num_target, num_atoms, 4, 3)`.
        C (torch.Tensor): Chain map with shape `(num_batch, num_residues)`.

    Outputs:
        D_error (Tensor, optional): Per-site average distance errors with shape
            `(num_batch)`.
    """

    def __init__(self, dist_eps: float = 1e-3):
        super(LossBackboneResidueDistance, self).__init__()
        self.dist_eps = dist_eps

    def _D(self, X):
        """Compute distance matrix between center of mass"""
        X_mean = X.mean(2)
        D = (
            (X_mean[:, :, None, :] - X_mean[:, None, :, :])
            .square()
            .sum(-1)
            .add(self.dist_eps)
            .sqrt()
        )
        return D

    def forward(
        self, X_mobile: torch.Tensor, X_target: torch.Tensor, C: torch.LongTensor
    ) -> torch.Tensor:
        mask = (C > 0).float()
        mask_2D = mask[:, :, None] * mask[:, None, :]
        D_error = (self._D(X_mobile) - self._D(X_target)).square()
        D_error = (mask_2D * D_error).sum(-1) / (mask_2D.sum(-1) + self.dist_eps)
        return D_error


def center_X(X: torch.Tensor, C: torch.LongTensor) -> torch.Tensor:
    """Center each protein system at the origin.

    Args:
        X (torch.Tensor): Backbone coordinates with shape
            `(batch_size, num_residues, num_atoms, 3)`.
        C (torch.LongTensor): Chain map with shape
            `(num_batch, num_residues)`.
    Returns:
        X_centered (torch.Tensor): Centered backbone coordinates with shape
            `(batch_size, num_residues, num_atoms, 3)`.
    """
    mask_expand = (
        (C > 0).float().reshape(list(C.shape) + [1, 1]).expand([-1, -1, 4, -1])
    )
    X_mean = (mask_expand * X).sum([1, 2], keepdims=True) / (
        mask_expand.sum([1, 2], keepdims=True)
    )
    X_centered = mask_expand * (X - X_mean)
    return X_centered


def atomic_mean(
    X_flat: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean across all 4 atom types.

    Args:
        X (torch.Tensor): Flattened backbone coordinates with shape
            `(batch_size, num_residues * num_atoms, 3)`.
        mask (torch.Tensor): Mask with shape `(num_batch, num_residues)`.
    Returns:
        X_mean (torch.Tensor): System centers with shape `(batch_size, 3)`.
        mask_atomic (torch.Tensor): Atomic mask with shape
            `(batch_size, num_residues * num_atoms)`.
    """
    mask_expand = mask.unsqueeze(-1).expand(-1, -1, 4)
    mask_atomic = mask_expand.reshape(mask.shape[0], -1).unsqueeze(-1)
    X_mean = torch.sum(mask_atomic * X_flat, 1, keepdims=True) / (
        torch.sum(mask_atomic, 1, keepdims=True)
    )
    return X_mean, mask_atomic


def scale_around_mean(
    X: torch.Tensor, C: torch.LongTensor, scale: float
) -> torch.Tensor:
    """Scale coordinates around mean.

    Args:
        X (torch.Tensor): Backbone coordinates with shape
            `(batch_size, num_residues, num_atoms, 3)`.
        C (LongTensor): Chain map with shape
            `(num_batch, num_residues)`.
        scale (float): Scalar factor by which to rescale
            the coordinates.

    Returns:
        X_scaled (torch.Tensor): Backbone coordinates with shape
            `(batch_size, num_residues, num_atoms, 3)`.
    """
    num_atoms = X.size(2)
    mask_expand = (
        (C > 0).float().reshape(list(C.shape) + [1, 1]).expand([-1, -1, num_atoms, -1])
    )
    X_mean = (mask_expand * X).sum([1, 2], keepdims=True) / (
        mask_expand.sum([1, 2], keepdims=True)
    )
    X_rescale = mask_expand * (scale[:, None, None, None] * (X - X_mean) + X_mean)
    return X_rescale


def impute_masked_X(X: torch.Tensor, C: torch.LongTensor) -> torch.Tensor:
    """Impute missing structure data to enforce chain contiguity.

        The posterior mean under a Brownian bridge is simply either the
        nearest unclamped state or a linear interpolant between the two
        nearest clamped endpoints along the chain.

    Args:
        X (torch.Tensor): Backbone coordinates with shape
            `(batch_size, num_residues, num_atoms, 3)`.
        C (LongTensor): Chain map with shape
            `(num_batch, num_residues)`.

    Returns:
        X (torch.Tensor): Imputed backbone coordinates with shape
            `(batch_size, num_residues, num_atoms, 3)`.
    """
    X_flat = X.reshape(X.shape[0], -1, 3)
    mask = (C > 0).type(torch.float32)
    X_mean, mask_atomic = atomic_mean(X_flat, mask)

    # Expand chain map into atomic level masking
    C_expand = C.unsqueeze(-1).expand(-1, -1, 4)
    C_atomic = C_expand.reshape(C.shape[0], -1)

    # Find nearest unmasked positions to the left and right
    ix = torch.arange(C_atomic.shape[1], device=X.device).reshape([1, -1])
    mask_atomic_extend = mask_atomic.squeeze(-1)
    ix_mask = mask_atomic_extend * ix - (1 - mask_atomic_extend)
    ix_left, _ = torch.cummax(ix_mask, dim=1)
    ix_flip = torch.flip(
        mask_atomic_extend * ix_mask + (1 - mask_atomic_extend) * C_atomic.shape[1],
        [1],
    )
    ix_right, _ = torch.cummin(ix_flip, dim=1)
    ix_right = torch.flip(ix_right, [1])

    ix_left = ix_left.long()
    ix_right = ix_right.long()

    clamped_left = ix_left >= 0
    clamped_right = ix_right < C_atomic.shape[1]
    ix_left[ix_left < 0] = 0
    ix_right[ix_right == C_atomic.shape[1]] = 0

    X_left = torch.gather(X_flat, 1, ix_left.unsqueeze(-1).expand(-1, -1, 3))
    X_right = torch.gather(X_flat, 1, ix_right.unsqueeze(-1).expand(-1, -1, 3))

    # Enfore that adjacent residues are same chain
    C_abs = torch.abs(C_atomic)
    C_left = torch.gather(C_abs, 1, ix_left)
    C_right = torch.gather(C_abs, 1, ix_right)
    clamped_left = clamped_left * (C_left == C_abs)
    clamped_right = clamped_right * (C_right == C_abs)

    # Build linear interpolant
    D_left = torch.abs(ix - ix_left)
    D_right = torch.abs(ix_right - ix)
    interp_theta = (D_right / (D_left + D_right + 1e-5)).unsqueeze(-1)
    X_interp = interp_theta * X_left + (1 - interp_theta) * X_right

    clamped_left = clamped_left.unsqueeze(-1)
    clamped_right = clamped_right.unsqueeze(-1)
    X_imputed_flat = mask_atomic * X_flat + (1 - mask_atomic) * (
        clamped_left * clamped_right * X_interp
        + clamped_right * (~clamped_left) * X_right
        + (~clamped_right) * clamped_left * X_left
    )

    X_imputed = X_imputed_flat.reshape(X.shape)
    return X_imputed


def expand_chain_map(C: torch.LongTensor) -> torch.Tensor:
    """Expand an integer chain map into a binary chain mask.

    Args:
        C (LongTensor): Chain map with shape
            `(num_batch, num_residues)`.

    Returns:
        mask_C (torch.Tensor): Expanded binary chain map with shape
            `(num_batch, num_residue, num_chains)`.
    """

    # Compute the per-chain averages of each feature within a chain, in place
    num_batch, num_residues = list(C.shape)
    num_chains = int(torch.max(C).item())

    # Build a position == chain expanded mask (B,L,C)
    C_expand = C.unsqueeze(-1).expand(-1, -1, num_chains)
    idx = torch.arange(num_chains, device=C.device) + 1
    idx_expand = idx.view(1, 1, -1)
    mask_C = (idx_expand == C_expand).type(torch.float32)
    return mask_C
