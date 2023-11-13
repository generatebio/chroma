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

"""Layers for measuring and building atomic geometries in proteins.

This module contains pytorch layers for computing common geometric features of 
protein backbones in a differentiable way and for converting between internal
and Cartesian coordinate representations.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Distances(nn.Module):
    """Euclidean distance layer (pairwise).

    This layer computes batched pairwise Euclidean distances, where the input
    tensor is treated as a batch of vectors with the final dimension as the
    feature dimension and the dimension for pairwise expansion can be specified.

    Args:
        distance_eps (float, optional): Small parameter to adde to squared
            distances to make gradients smooth near 0.

    Inputs:
        X (tensor): Input coordinates with shape `([...], length, [...], 3)`.
        dim (int, optional): Dimension upon which to expand to pairwise
            distances. Defaults to -2.
        mask (tensor, optional): Masking tensor with shape
            `([...], length, [...])`.

    Outputs:
        D (tensor): Distances with shape `([...], length, length, [...])`
    """

    def __init__(self, distance_eps=1e-3):
        super(Distances, self).__init__()
        self.distance_eps = distance_eps

    def forward(
        self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, dim: float = -2
    ) -> torch.Tensor:
        dim_expand = dim if dim < 0 else dim + 1
        dX = X.unsqueeze(dim_expand - 1) - X.unsqueeze(dim_expand)
        D_square = torch.sum(dX ** 2, -1)
        D = torch.sqrt(D_square + self.distance_eps)
        if mask is not None:
            mask_expand = mask.unsqueeze(dim) * mask.unsqueeze(dim + 1)
            D = mask_expand * D
        return D


class VirtualAtomsCA(nn.Module):
    """Virtual atoms layer, branching from backbone C-alpha carbons.

    This layer places virtual atom coordinates relative to backbone coordinates
    in a differentiable way.

    Args:
        virtual_type (str, optional): Type of virtual atom to place. Currently
            supported types are `dicons`, a virtual placement that was
            optimized to predict potential rotamer interactions, and `cbeta`
            which places a virtual C-beta carbon assuming ideal geometry.
        distance_eps (float, optional): Small parameter to add to squared
            distances to make gradients smooth near 0.

    Inputs:
        X (Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_atom_types, 3)`.
        C (Tensor): Chain map tensor with shape `(num_batch, num_residues)`.

    Outputs:
        X_virtual (Tensor): Virtual coordinates with shape
            `(num_batch, num_residues, 3)`.
    """

    def __init__(self, virtual_type="dicons", distance_eps=1e-3):
        super(VirtualAtomsCA, self).__init__()
        self.distance_eps = distance_eps

        """
        Geometry specifications
        dicons
            Length       CA-X:     2.3866
            Angle      N-CA-X:   111.0269
            Dihedral C-N-CA-X:  -138.886412

        cbeta
            Length       CA-X:     1.532    (Engh and Huber, 2001)
            Angle      N-CA-X:   109.5      (tetrahedral geometry)
            Dihedral C-N-CA-X:  -125.25     (109.5 / 2 - 180)
        """
        self.virtual_type = virtual_type
        virtual_geometries = {
            "dicons": [2.3866, 111.0269, -138.8864122],
            "cbeta": [1.532, 109.5, -125.25],
        }
        self.virtual_geometries = virtual_geometries
        self.distance_eps = distance_eps

    def geometry(self):
        bond, angle, dihedral = self.virtual_geometries[self.virtual_type]
        return bond, angle, dihedral

    def forward(self, X: torch.Tensor, C: torch.LongTensor) -> torch.Tensor:
        bond, angle, dihedral = self.geometry()

        ones = torch.ones([1, 1], device=X.device)
        bonds = bond * ones
        angles = angle * ones
        dihedrals = dihedral * ones

        # Build reference frame
        # 1.C -> 2.N -> 3.CA -> 4.X
        X_N, X_CA, X_C, X_O = X.unbind(2)
        X_virtual = extend_atoms(
            X_C,
            X_N,
            X_CA,
            bonds,
            angles,
            dihedrals,
            degrees=True,
            distance_eps=self.distance_eps,
        )

        # Mask missing positions
        mask = (C > 0).type(torch.float32).unsqueeze(-1)
        X_virtual = mask * X_virtual
        return X_virtual


def normed_vec(V: torch.Tensor, distance_eps: float = 1e-3) -> torch.Tensor:
    """Normalized vectors with distance smoothing.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V (Tensor): Batch of vectors with shape `(..., num_dims)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        U (Tensor): Batch of normalized vectors with shape `(..., num_dims)`.
    """
    # Unit vector from i to j
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U


def normed_cross(
    V1: torch.Tensor, V2: torch.Tensor, distance_eps: float = 1e-3
) -> torch.Tensor:
    """Normalized cross product between vectors.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V1 (Tensor): Batch of vectors with shape `(..., 3)`.
        V2 (Tensor): Batch of vectors with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        C (Tensor): Batch of cross products `v_1 x v_2` with shape `(..., 3)`.
    """
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C


def lengths(
    atom_i: torch.Tensor, atom_j: torch.Tensor, distance_eps: float = 1e-3
) -> torch.Tensor:
    """Batched bond lengths given batches of atom i and j.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        L (Tensor): Elementwise bond lengths `||x_i - x_j||` with shape `(...)`.
    """
    # Bond length of i-j
    dX = atom_j - atom_i
    L = torch.sqrt((dX ** 2).sum(dim=-1) + distance_eps)
    return L


def angles(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    distance_eps: float = 1e-3,
    degrees: bool = False,
) -> torch.Tensor:
    """Batched bond angles given atoms `i-j-k`.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        atom_k (Tensor): Atom `k` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.
        degrees (bool, optional): If True, convert to degrees. Default: False.

    Returns:
        A (Tensor): Elementwise bond angles with shape `(...)`.
    """
    # Bond angle of i-j-k
    U_ji = normed_vec(atom_i - atom_j, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    inner_prod = torch.einsum("bix,bix->bi", U_ji, U_jk)
    inner_prod = torch.clamp(inner_prod, -1, 1)
    A = torch.acos(inner_prod)
    if degrees:
        A = A * 180.0 / np.pi
    return A


def dihedrals(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    atom_l: torch.Tensor,
    distance_eps: float = 1e-3,
    degrees: bool = False,
) -> torch.Tensor:
    """Batched bond dihedrals given atoms `i-j-k-l`.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        atom_k (Tensor): Atom `k` coordinates with shape `(..., 3)`.
        atom_l (Tensor): Atom `l` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.
        degrees (bool, optional): If True, convert to degrees. Default: False.

    Returns:
        D (Tensor): Elementwise bond dihedrals with shape `(...)`.
    """
    U_ij = normed_vec(atom_j - atom_i, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    U_kl = normed_vec(atom_l - atom_k, distance_eps=distance_eps)
    normal_ijk = normed_cross(U_ij, U_jk, distance_eps=distance_eps)
    normal_jkl = normed_cross(U_jk, U_kl, distance_eps=distance_eps)
    # _inner_product = lambda a, b: torch.einsum("bix,bix->bi", a, b)
    _inner_product = lambda a, b: (a * b).sum(-1)
    cos_dihedrals = _inner_product(normal_ijk, normal_jkl)
    angle_sign = _inner_product(U_ij, normal_jkl)
    cos_dihedrals = torch.clamp(cos_dihedrals, -1, 1)
    D = torch.sign(angle_sign) * torch.acos(cos_dihedrals)
    if degrees:
        D = D * 180.0 / np.pi
    return D


def extend_atoms(
    X_1: torch.Tensor,
    X_2: torch.Tensor,
    X_3: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
    distance_eps: float = 1e-3,
    degrees: bool = False,
) -> torch.Tensor:
    """Place atom `X_4` given `X_1`, `X_2`, `X_3` and internal coordinates.

                           ___________________
                          | X_1 - X_2         |
                          |       |           |
                          |       X_3 - [X_4] |
                          |___________________|

    This uses a similar approach as NERF:
        Parsons et al, Computational Chemistry (2005).
        https://doi.org/10.1002/jcc.20237
    See the reference for further explanation about converting from internal
    coordinates to Cartesian coordinates.

    Args:
        X_1 (Tensor): First atom coordinates with shape  `(..., 3)`.
        X_2 (Tensor): Second atom coordinates with shape `(..., 3)`.
        X_3 (Tensor): Third atom coordinates with shape  `(..., 3)`.
        lengths (Tensor): Bond lengths `X_3-X_4` with shape `(...)`.
        angles (Tensor): Bond angles `X_2-X_3-X_4` with shape `(...)`.
        dihedrals (Tensor): Bond dihedrals `X_1-X_2-X_3-X_4` with shape `(...)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            This preserves differentiability for zero distances. Default: 1E-3.
        degrees (bool, optional): If True, inputs are treated as degrees.
            Default: False.

    Returns:
        X_4 (Tensor): Placed atom with shape `(..., 3)`.
    """
    if degrees:
        angles *= np.pi / 180.0
        dihedrals *= np.pi / 180.0

    r_32 = X_2 - X_3
    r_12 = X_2 - X_1
    n_1 = normed_vec(r_32, distance_eps=distance_eps)
    n_2 = normed_cross(n_1, r_12, distance_eps=distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps=distance_eps)

    lengths = lengths.unsqueeze(-1)
    cos_angle = torch.cos(angles).unsqueeze(-1)
    sin_angle = torch.sin(angles).unsqueeze(-1)
    cos_dihedral = torch.cos(dihedrals).unsqueeze(-1)
    sin_dihedral = torch.sin(dihedrals).unsqueeze(-1)

    X_4 = X_3 + lengths * (
        cos_angle * n_1
        + (sin_angle * sin_dihedral) * n_2
        + (sin_angle * cos_dihedral) * n_3
    )
    return X_4


class InternalCoords(nn.Module):
    """Internal coordinates layer.

    This layer computes internal coordinates (ICs) from a batch of protein
    backbones. To make the ICs differentiable everywhere, this layer replaces
    distance calculations of the form `sqrt(sum_sq)` with smooth, non-cusped
    approximation `sqrt(sum_sq + eps)`.

    Args:
        distance_eps (float, optional): Small parameter to add to squared
            distances to make gradients smooth near 0.

    Inputs:
        X (Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_atom_types, 3)`.
        C (Tensor): Chain map tensor with shape
            `(num_batch, num_residues)`.

    Outputs:
        dihedrals (Tensor): Backbone dihedral angles with shape
            `(num_batch, num_residues, 4)`
        angles (Tensor): Backbone bond lengths with shape
            `(num_batch, num_residues, 4)`
        lengths (Tensor): Backbone bond lengths with shape
            `(num_batch, num_residues, 4)`
    """

    def __init__(self, distance_eps=1e-3):
        super(InternalCoords, self).__init__()
        self.distance_eps = distance_eps

    def forward(
        self,
        X: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        return_masks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = (C > 0).float()
        X_chain = X[:, :, :3, :]
        num_batch, num_residues, _, _ = X_chain.shape
        X_chain = X_chain.reshape(num_batch, 3 * num_residues, 3)

        # This function historically returns the angle complement
        _lengths = lambda Xi, Xj: lengths(Xi, Xj, distance_eps=self.distance_eps)
        _angles = lambda Xi, Xj, Xk: np.pi - angles(
            Xi, Xj, Xk, distance_eps=self.distance_eps
        )
        _dihedrals = lambda Xi, Xj, Xk, Xl: dihedrals(
            Xi, Xj, Xk, Xl, distance_eps=self.distance_eps
        )

        # Compute internal coordinates associated with -[N]-[CA]-[C]-
        NCaC_L = _lengths(X_chain[:, 1:, :], X_chain[:, :-1, :])
        NCaC_A = _angles(X_chain[:, :-2, :], X_chain[:, 1:-1, :], X_chain[:, 2:, :])
        NCaC_D = _dihedrals(
            X_chain[:, :-3, :],
            X_chain[:, 1:-2, :],
            X_chain[:, 2:-1, :],
            X_chain[:, 3:, :],
        )

        # Compute internal coordinates associated with [C]=[O]
        _, X_CA, X_C, X_O = X.unbind(dim=2)
        X_N_next = X[:, 1:, 0, :]
        O_L = _lengths(X_C, X_O)
        O_A = _angles(X_CA, X_C, X_O)
        O_D = _dihedrals(X_N_next, X_CA[:, :-1, :], X_C[:, :-1, :], X_O[:, :-1, :])

        if C is None:
            C = torch.zeros_like(mask)

        # Mask nonphysical bonds and angles
        # Note: this could probably also be expressed as a Conv, unclear
        # which is faster and this probably not rate-limiting.
        C = C * (mask.type(torch.long))
        ii = torch.stack(3 * [C], dim=-1).view([num_batch, -1])
        L0, L1 = ii[:, :-1], ii[:, 1:]
        A0, A1, A2 = ii[:, :-2], ii[:, 1:-1], ii[:, 2:]
        D0, D1, D2, D3 = ii[:, :-3], ii[:, 1:-2], ii[:, 2:-1], ii[:, 3:]

        # Mask for linear backbone
        mask_L = torch.eq(L0, L1)
        mask_A = torch.eq(A0, A1) * torch.eq(A0, A2)
        mask_D = torch.eq(D0, D1) * torch.eq(D0, D2) * torch.eq(D0, D3)
        mask_L = mask_L.type(torch.float32)
        mask_A = mask_A.type(torch.float32)
        mask_D = mask_D.type(torch.float32)

        # Masks for branched oxygen
        mask_O_D = torch.eq(C[:, :-1], C[:, 1:])
        mask_O_D = mask_O_D.type(torch.float32)
        mask_O_A = mask
        mask_O_L = mask

        def _pad_pack(D, A, L, O_D, O_A, O_L):
            # Pad and pack together the components
            D = F.pad(D, (1, 2))
            A = F.pad(A, (0, 2))
            L = F.pad(L, (0, 1))
            O_D = F.pad(O_D, (0, 1))
            D, A, L = [x.reshape(num_batch, num_residues, 3) for x in [D, A, L]]
            _pack = lambda a, b: torch.cat([a, b.unsqueeze(-1)], dim=-1)
            L = _pack(L, O_L)
            A = _pack(A, O_A)
            D = _pack(D, O_D)
            return D, A, L

        D, A, L = _pad_pack(NCaC_D, NCaC_A, NCaC_L, O_D, O_A, O_L)
        mask_D, mask_A, mask_L = _pad_pack(
            mask_D, mask_A, mask_L, mask_O_D, mask_O_A, mask_O_L
        )
        mask_expand = mask.unsqueeze(-1)
        mask_D = mask_expand * mask_D
        mask_A = mask_expand * mask_A
        mask_L = mask_expand * mask_L

        D = mask_D * D
        A = mask_A * A
        L = mask_L * L

        if not return_masks:
            return D, A, L
        else:
            return D, A, L, mask_D, mask_A, mask_L


class VirtualAtomsCA(nn.Module):
    """Virtual atoms layer, branching from backbone C-alpha carbons.

    This layer places virtual atom coordinates relative to backbone coordinates
    in a differentiable way.

    Args:
        virtual_type (str, optional): Type of virtual atom to place. Currently
            supported types are `dicons`, a virtual placement that was
            optimized to predict potential rotamer interactions, and `cbeta`
            which places a virtual C-beta carbon assuming ideal geometry.
        distance_eps (float, optional): Small parameter to add to squared
            distances to make gradients smooth near 0.

    Inputs:
        X (Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_atom_types, 3)`.
        C (Tensor): Chain map tensor with shape `(num_batch, num_residues)`.

    Outputs:
        X_virtual (Tensor): Virtual coordinates with shape
            `(num_batch, num_residues, 3)`.
    """

    def __init__(self, virtual_type="dicons", distance_eps=1e-3):
        super(VirtualAtomsCA, self).__init__()
        self.distance_eps = distance_eps

        """
        Geometry specifications
        dicons
            Length       CA-X:     2.3866
            Angle      N-CA-X:   111.0269
            Dihedral C-N-CA-X:  -138.886412

        cbeta
            Length       CA-X:     1.532    (Engh and Huber, 2001)
            Angle      N-CA-X:   109.5      (tetrahedral geometry)
            Dihedral C-N-CA-X:  -125.25     (109.5 / 2 - 180)
        """
        self.virtual_type = virtual_type
        virtual_geometries = {
            "dicons": [2.3866, 111.0269, -138.8864122],
            "cbeta": [1.532, 109.5, -125.25],
        }
        self.virtual_geometries = virtual_geometries
        self.distance_eps = distance_eps

    def geometry(self):
        bond, angle, dihedral = self.virtual_geometries[self.virtual_type]
        return bond, angle, dihedral

    def forward(self, X: torch.Tensor, C: torch.LongTensor) -> torch.Tensor:
        bond, angle, dihedral = self.geometry()

        ones = torch.ones([1, 1], device=X.device)
        bonds = bond * ones
        angles = angle * ones
        dihedrals = dihedral * ones

        # Build reference frame
        # 1.C -> 2.N -> 3.CA -> 4.X
        X_N, X_CA, X_C, X_O = X.unbind(2)
        X_virtual = extend_atoms(
            X_C,
            X_N,
            X_CA,
            bonds,
            angles,
            dihedrals,
            degrees=True,
            distance_eps=self.distance_eps,
        )

        # Mask missing positions
        mask = (C > 0).type(torch.float32).unsqueeze(-1)
        X_virtual = mask * X_virtual
        return X_virtual


def quaternions_from_rotations(R: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Convert a batch of rotation matrices to quaternions.

    See en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for further
    details on converting between quaternions and rotation matrices.

    Args:
        R (tensor): Batch of rotation matrices with shape `(..., 3, 3)`.

    Returns:
        q (tensor): Batch of quaternion vectors with shape `(..., 4)`. Quaternion
        is in the order `[angle, axis_x, axis_y, axis_z]`.
    """

    batch_dims = list(R.shape)[:-2]
    R_flat = R.reshape(batch_dims + [9])
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R_flat.unbind(-1)

    # Quaternion possesses both an axis and angle of rotation
    _sqrt = lambda r: torch.sqrt(F.relu(r) + eps)
    q_angle = _sqrt(1 + R00 + R11 + R22).unsqueeze(-1)
    magnitudes = _sqrt(
        1 + torch.stack([R00 - R11 - R22, -R00 + R11 - R22, -R00 - R11 + R22], -1)
    )
    signs = torch.sign(torch.stack([R21 - R12, R02 - R20, R10 - R01], -1))
    q_axis = signs * magnitudes

    # Normalize (for safety and a missing factor of 2)
    q_unc = torch.cat((q_angle, q_axis), -1)
    q = normed_vec(q_unc, distance_eps=eps)
    return q


def rotations_from_quaternions(
    q: torch.Tensor, normalize: bool = False, eps: float = 1e-3
) -> torch.Tensor:
    """Convert a batch of quaternions to rotation matrices.

    See en.wikipedia.org/wiki/Quaternions_and_spatial_rotation for further
    details on converting between quaternions and rotation matrices.

    Returns:
        q (tensor): Batch of quaternion vectors with shape `(..., 4)`. Quaternion
            is in the order `[angle, axis_x, axis_y, axis_z]`.
        normalize (boolean, optional): Option to normalize the quaternion before
            conversion.

    Args:
        R (tensor): Batch of rotation matrices with shape `(..., 3, 3)`.
    """
    batch_dims = list(q.shape)[:-1]
    if normalize:
        q = normed_vec(q, distance_eps=eps)

    a, b, c, d = q.unbind(-1)
    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    R = torch.stack(
        [
            a2 + b2 - c2 - d2,
            2 * b * c - 2 * a * d,
            2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d,
            a2 - b2 + c2 - d2,
            2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c,
            2 * c * d + 2 * a * b,
            a2 - b2 - c2 + d2,
        ],
        dim=-1,
    )

    R = R.view(batch_dims + [3, 3])
    return R


def frames_from_backbone(X: torch.Tensor, distance_eps: float = 1e-3):
    """Convert a backbone into local reference frames.

    Args:
        X (Tensor): Backbone coordinates with shape `(..., 4, 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        R (Tensor): Reference frames with shape `(..., 3, 3)`.
        X_CA (Tensor): C-alpha coordinates with shape `(..., 3)`
    """
    X_N, X_CA, X_C, X_O = X.unbind(-2)
    u_CA_N = normed_vec(X_N - X_CA, distance_eps)
    u_CA_C = normed_vec(X_C - X_CA, distance_eps)
    n_1 = u_CA_N
    n_2 = normed_cross(n_1, u_CA_C, distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps)
    R = torch.stack([n_1, n_2, n_3], -1)
    return R, X_CA


def hat(omega: torch.Tensor) -> torch.Tensor:
    """
    Maps [x,y,z] to [[0,-z,y], [z,0,-x], [-y, x, 0]]
    Args:
        omega (torch.tensor): of size (*, 3)
    Returns:
        hat{omega} (torch.tensor): of size (*, 3, 3) skew symmetric element in so(3)
    """
    target = torch.zeros(*omega.size()[:-1], 9, device=omega.device)
    index1 = torch.tensor([7, 2, 3], device=omega.device).expand(
        *target.size()[:-1], -1
    )
    index2 = torch.tensor([5, 6, 1], device=omega.device).expand(
        *target.size()[:-1], -1
    )
    return (
        target.scatter(-1, index1, omega)
        .scatter(-1, index2, -omega)
        .reshape(*target.size()[:-1], 3, 3)
    )


def V(omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    I = torch.eye(3, device=omega.device).expand(*omega.size()[:-1], 3, 3)
    theta = omega.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt()[..., None]
    omega_hat = hat(omega)
    M1 = ((1 - theta.cos()) / theta.pow(2)) * (omega_hat)
    M2 = ((theta - theta.sin()) / theta.pow(3)) * (omega_hat @ omega_hat)
    return I + M1 + M2
