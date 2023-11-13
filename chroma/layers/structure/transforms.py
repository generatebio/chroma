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

"""Layers for batched 3D transformations, such as residue poses.

This module contains pytorch layers for computing and composing with
3D, 6-degree-of freedom transformations.
"""


from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from chroma.layers import graph
from chroma.layers.structure import geometry


def compose_transforms(
    R_a: torch.Tensor, t_a: torch.Tensor, R_b: torch.Tensor, t_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose transforms `T_compose = T_a * T_b` (broadcastable).

    Args:
        R_a (torch.Tensor): Transform `T_a` rotation matrix with shape `(...,3,3)`.
        t_a (torch.Tensor): Transform `T_a` translation with shape `(...,3)`.
        R_b (torch.Tensor): Transform `T_b` rotation matrix with shape `(...,3,3)`.
        t_b (torch.Tensor): Transform `T_b` translation with shape `(...,3)`.

    Returns:
        R_composed (torch.Tensor): Composed transform `a * b` rotation matrix with
            shape `(...,3,3)`.
        t_composed (torch.Tensor): Composed transform `a * b` translation vector with
            shape `(...,3)`.
    """
    R_composed = R_a @ R_b
    t_composed = t_a + (R_a @ t_b.unsqueeze(-1)).squeeze(-1)
    return R_composed, t_composed


def compose_translation(
    R_a: torch.Tensor, t_a: torch.Tensor, t_b: torch.Tensor
) -> torch.Tensor:
    """Compose translation component of `T_compose = T_a * T_b` (broadcastable).

    Args:
        R_a (torch.Tensor): Transform `T_a` rotation matrix with shape `(...,3,3)`.
        t_a (torch.Tensor): Transform `T_a` translation with shape `(...,3)`.
        t_b (torch.Tensor): Transform `T_b` translation with shape `(...,3)`.

    Returns:
        t_composed (torch.Tensor): Composed transform `a * b` translation vector with
            shape `(...,3)`.
    """
    t_composed = t_a + (R_a @ t_b.unsqueeze(-1)).squeeze(-1)
    return t_composed


def compose_inner_transforms(
    R_a: torch.Tensor, t_a: torch.Tensor, R_b: torch.Tensor, t_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose the relative inner transform `T_ab = T_a^{-1} * T_b`.

    Args:
        R_a (torch.Tensor): Transform `T_a` rotation matrix with shape `(...,3,3)`.
        t_a (torch.Tensor): Transform `T_a` translation with shape `(...,3)`.
        R_b (torch.Tensor): Transform `T_b` rotation matrix with shape `(...,3,3)`.
        t_b (torch.Tensor): Transform `T_b` translation with shape `(...,3)`.

    Returns:
        R_ab (torch.Tensor): Composed transform `T_a * T_b` rotation matrix with
            shape `(...,3,3)`. Inner dimensions are broadcastable.
        t_ab (torch.Tensor): Composed transform `T_a * T_b` translation vector with
            shape `(...,3)`.
    """
    R_a_inverse = R_a.transpose(-1, -2)
    R_ab = R_a_inverse @ R_b
    t_ab = (R_a_inverse @ ((t_b - t_a).unsqueeze(-1))).squeeze(-1)
    return R_ab, t_ab


def fuse_gaussians_isometric_plus_radial(
    x: torch.Tensor,
    p_iso: torch.Tensor,
    p_rad: torch.Tensor,
    direction: torch.Tensor,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse Gaussians along a dimension ``dim``. This assumes the Gaussian
    precision matrices are a sum of an isometric part P_iso together with
    a part P_rad that provides information only along one direction.

    Args:
        x (torch.Tensor): A (...,3)-shaped tensor of means.
        p_iso (torch.Tensor): A (...)-shaped tensor of weights of the isometric part of the
            precision matrix.
        p_rad (torch.Tensor): A (...)-shaped tensor of weights of the radial part of the
            precision matrix.
        direction (torch.Tensor): A (...,3)-shaped tensor of directions along which
            information is available.
        dim (int): The dimension over which to aggregate (fuse).

    Returns:
        A tuple ``(x_fused, P_fused)`` of fused mean and precision, with
        specified ``dim`` removed.
    """
    assert dim >= 0, "dimension must index from the left"

    # P_rad has information only parallel to the edge.
    outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
    inner = direction.square().sum(-1).clamp(min=1e-10)
    P_rad = (p_rad / inner)[..., None, None] * outer
    P_iso = p_iso.unsqueeze(-1).expand(p_iso.shape + (3,)).diag_embed()
    P = P_iso + P_rad

    # Compute the Bayesian fusion aka product-of-experts of the Gaussians.
    P_fused = P.sum(dim)
    Px_fused = (P @ x.unsqueeze(-1)).squeeze(-1).sum(dim)
    # There might be a cheaper way to do this, either via Cholesky
    # or hand-coding the 3x3 matrix solve operation.
    x_fused = torch.linalg.solve(P_fused, Px_fused)

    return x_fused, P_fused


def collect_neighbor_transforms(
    R_i: torch.Tensor, t_i: torch.Tensor, edge_idx: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect neighbor transforms.

    Args:
        R_i (torch.Tensor): Transform `T` rotation matrices with shape
            `(num_batch, num_residues, 3, 3)`.
        t_i (torch.Tensor): Transform `T` translations with shape
            `(num_batch, num_residues, 3)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
       R_j (torch.Tensor): Rotation matrices of neighbor transforms, with shape
           `(num_batch, num_residues, num_neighbors, 3, 3)`.
       t_j (torch.Tensor): Translations of neighbor transforms, with shape
           `(num_batch, num_residues, num_neighbors, 3)`.
    """
    num_batch, num_residues, num_neighbors = edge_idx.shape
    R_i_flat = R_i.reshape([num_batch, num_residues, 9])
    R_j = graph.collect_neighbors(R_i_flat, edge_idx).reshape(
        [num_batch, num_residues, num_neighbors, 3, 3]
    )
    t_j = graph.collect_neighbors(t_i, edge_idx)
    return R_j, t_j


def collect_neighbor_inner_transforms(
    R_i: torch.Tensor, t_i: torch.Tensor, edge_idx: torch.LongTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect inner transforms between neighbors.

    Args:
        R_i (torch.Tensor): Transform `T` rotation matrices with shape
            `(num_batch, num_residues, 3, 3)`.
        t_i (torch.Tensor): Transform `T` translations with shape
            `(num_batch, num_residues, 3)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
       R_ji (torch.Tensor): Rotation matrices of neighbor transforms, with shape
           `(num_batch, num_residues, num_neighbors, 3, 3)`.
       t_ji (torch.Tensor): Translations of neighbor transforms, with shape
           `(num_batch, num_residues, num_neighbors, 3)`.
    """
    R_j, t_j = collect_neighbor_transforms(R_i, t_i, edge_idx)
    R_ji, t_ji = compose_inner_transforms(
        R_j, t_j, R_i.unsqueeze(-3), t_i.unsqueeze(-2)
    )
    return R_ji, t_ji


def equilibrate_transforms(
    R_i: torch.Tensor,
    t_i: torch.Tensor,
    R_ji: torch.Tensor,
    t_ji: torch.Tensor,
    logit_ij: torch.Tensor,
    mask_ij: torch.Tensor,
    edge_idx: torch.LongTensor,
    iterations: int = 1,
    R_global: Optional[torch.Tensor] = None,
    t_global: Optional[torch.Tensor] = None,
    R_global_i: Optional[torch.Tensor] = None,
    t_global_i: Optional[torch.Tensor] = None,
    logit_global_i: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equilibrate neighbor transforms.

    Args:
        R_i (torch.Tensor): Transform `T` rotation matrices with shape
            `(num_batch, num_residues, 3, 3)`.
        t_i (torch.Tensor): Transform `T` translations with shape
            `(num_batch, num_residues, 3)`.
        R_ji (torch.Tensor): Rotation matrices to go between frames for nodes i and j
            with shape `(num_batch, num_residues, num_neighbors, 3, 3)`.
        t_ji (torch.Tensor): Translations to go between frames for nodes i and j with
            shape `(num_batch, num_residues, num_neighbors, 3)`.
        logit_ij (torch.Tensor): Logits for averaging neighbor transforms with shape
            `(num_batch, num_residues, num_neighbors, num_weights)`. Note that
            `num_weights` must be 1, 2, or 3; see the documentation for
            `generate.layers.structure.transforms.average_transforms` for an
            explanation of the interpretations with different `num_weights`.
        mask_ij (torch.Tensor): Mask for averaging neighbor transforms with shape
            `(num_batch, num_residues, num_neighbors)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.
        iterations (int): Number of iterations to equilibrate for.
        R_global (torch.Tensor): Optional global frame rotation matrix with shape
            `(num_batch, 3, 3)`.
        t_global (torch.Tensor): Optional global frame translation with shape
            `(num_batch, 3)`.
        R_global_i (torch.Tensor): Optional rotation matrix for global frame from
            nodes with shape `(num_batch, num_residues, 3, 3)`.
        t_global_i (torch.Tensor): Optional translation for global frame from nodes
            with shape `(num_batch, num_residues, 3)`.
        logit_global_i (torch.Tensor): Logits for averaging global frame transform
            with shape `(num_batch, num_residues, num_weights)`. `num_weights`
            should match that of `logit_ij`.

    Returns:
       R_i (torch.Tensor): Rotation matrices of equilibrated transforms, with shape
           `(num_batch, num_residues, 3, 3)`.
       t_i (torch.Tensor): Translations of equilibrated transforms, with shape
           `(num_batch, num_residues, 3)`.
    """

    # Optional global frames are treated as additional neighbor
    update_global = False
    if None not in [R_global, t_global, R_global_i, t_global_i, logit_global_i]:
        update_global = True
        num_batch, num_residues, num_neighbors = list(mask_ij.shape)
        R_global_i = R_global_i.unsqueeze(2)
        t_global_i = t_global_i.unsqueeze(2)
        R_ji = torch.cat((R_ji, R_global_i), dim=2)
        t_ji = torch.cat((t_ji, t_global_i), dim=2)
        logit_ij = torch.cat((logit_ij, logit_global_i.unsqueeze(2)), dim=2)
        R_global = R_global.reshape([num_batch, 1, 1, 3, 3]).expand(R_global_i.shape)
        t_global = t_global.reshape([num_batch, 1, 1, 3]).expand(t_global_i.shape)
        mask_i = (mask_ij.sum(2, keepdims=True) > 0).float()
        mask_ij = torch.cat((mask_ij, mask_i), dim=2)

    t_edge = None
    for i in range(iterations):
        R_j, t_j = collect_neighbor_transforms(R_i, t_i, edge_idx)
        if update_global:
            R_j = torch.cat((R_j, R_global), dim=2)
            t_j = torch.cat((t_j, t_global), dim=2)
        R_i_pred, t_i_pred = compose_transforms(R_j, t_j, R_ji, t_ji)

        if logit_ij.size(-1) == 3:
            # Compute i-j displacement in the same coordinate system as
            # t_i_pred, i.e. in global coords. Sign does not matter.
            t_edge = t_j - t_i_pred

        R_i, t_i = average_transforms(
            R_i_pred, t_i_pred, logit_ij, mask_ij, t_edge=t_edge, dim=2
        )

    return R_i, t_i


def average_transforms(
    R: torch.Tensor,
    t: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    t_edge: Optional[torch.Tensor] = None,
    dither: Optional[bool] = True,
    dither_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average transforms with optional dithering.

    Args:
        R (torch.Tensor): Transform `T` rotation matrix with shape `(...,3,3)`.
        t (torch.Tensor): Transform `T` translation with shape `(...,3)`.
        w (torch.Tensor): Logits for averaging weights with shape
            `(...,num_weights)`. `num_weights` can be 1 (single scalar
            weight per transform), 2 (separate weights for each rotation
            and translation), or 3 (one weight for rotation, two weights
            for translation corresponding to precision in all directions /
            along t_edge).
        mask (torch.Tensor): Mask for averaging weights with shape `(...)`.
        dim (int): Dimension to average along.
        t_edge (torch.Tensor, optional): Translation `T` of shape `(..., 3)`
            indicating the displacement between source and target nodes.
        dither (bool): Whether to noise final rotations.
        dither_eps (float): Fractional amount by which to noise rotations.

    Returns:
        R_avg (torch.Tensor): Average transform `T_avg` rotation matrix with
            shape `(...{reduced}...,3,3)`.
        t_avg (torch.Tensor): Average transform `T_avg` translation with
            shape `(...{reduced}...,3)`.
    """
    assert dim >= 0, "dimension must index from the left"
    w = torch.where(
        mask[..., None].bool(), w, torch.full_like(w, torch.finfo(w.dtype).min)
    )

    # We use different averaging models based on the number of weights
    num_transform_weights = w.size(-1)
    if num_transform_weights == 1:
        # Share a single scalar weight between t and R.
        probs = w.softmax(dim)
        t_probs = probs
        R_probs = probs[..., None]

        # Average translation.
        t_avg = (t * t_probs).sum(dim)
    elif num_transform_weights == 2:
        # Use separate scalar weights for each of t and R.
        probs = w.softmax(dim)
        t_probs, R_probs = probs.unbind(-1)
        t_probs = t_probs[..., None]
        R_probs = R_probs[..., None, None]

        # Average translation.
        t_avg = (t * t_probs).sum(dim)
    elif num_transform_weights == 3:
        # For R use a signed scalar weight.
        R_probs = w[..., 2].softmax(dim)[..., None, None]

        # For t use a two-parameter precision matrix P = P_isometric + P_radial.
        # We need to hand compute softmax over the shared dim x 2 elements.
        w_t = w[..., :2]
        w_t_total = w_t.logsumexp([dim, -1], True)
        p_iso, p_rad = (w_t - w_t_total).exp().unbind(-1)

        # Use Gaussian fusion for translation.
        t_edge = t_edge * mask.to(t_edge.dtype)[..., None]
        t_avg, _ = fuse_gaussians_isometric_plus_radial(t, p_iso, p_rad, t_edge, dim)
    else:
        raise NotImplementedError

    # Average rotation via SVD
    R_avg_unc = (R * R_probs).sum(dim)
    R_avg_unc = R_avg_unc + dither_eps * torch.randn_like(R_avg_unc)
    U, S, Vh = torch.linalg.svd(R_avg_unc, full_matrices=True)
    R_avg = U @ Vh

    # Enforce that matrix is rotation matrix
    d = torch.linalg.det(R_avg)
    d_expand = F.pad(d[..., None, None], (2, 0), value=1.0)
    Vh = Vh * d_expand
    R_avg = U @ Vh
    return R_avg, t_avg


def _debug_plot_transforms(
    R_ij: torch.Tensor,
    t_ij: torch.Tensor,
    logits_ij: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_ij: torch.Tensor,
    dist_eps: float = 1e-3,
):
    """Visualize 6dof frame transformations"""
    from matplotlib import pyplot as plt

    num_batch = R_ij.shape[0]
    num_residues = R_ij.shape[1]

    # Masked softmax on logits
    # logits_ij = torch.where(
    #     mask_ij.bool(), logits_ij,
    #     torch.full_like(logits_ij, torch.finfo(logits_ij.dtype).min)
    # )
    p_ij = torch.softmax(logits_ij, 2)
    p_ij = torch.log_softmax(logits_ij, 2)
    # p_ij = torch.softmax(logits_ij, 2)
    P_ij = graph.scatter_edges(p_ij[..., None], edge_idx)[..., 0]

    q_ij = geometry.quaternions_from_rotations(R_ij)
    q_ij = graph.scatter_edges(q_ij, edge_idx)
    t_ij = graph.scatter_edges(t_ij, edge_idx)

    # Converte to distance, direction, orientation
    D = torch.sqrt(t_ij.square().sum(-1))
    U = t_ij / (D[..., None] + dist_eps)
    D_max = D.max().item()
    t_ij = t_ij / D_max
    q_axis = q_ij[..., 1:]

    # Distance features
    D_img = D
    D_img_min = D_img.min().item()
    D_img_max = D_img.max().item()

    def _format(T):
        T = T.cpu().data.numpy()
        # RGB on (0,1)^3
        if len(T.shape) == 3:
            T = (T + 1) / 2
        return T

    base_width = 4
    num_cols = 4
    plt.figure(figsize=(base_width * 4, base_width * num_batch), dpi=300)
    ix = 1
    for i in range(num_batch):
        plt.subplot(num_batch, num_cols, ix)
        plt.imshow(_format(D_img[i, :, :]), cmap="inferno")
        # plt.clim([hD_min, hD_max])
        plt.axis("off")

        plt.subplot(num_batch, num_cols, ix + 1)
        plt.imshow(_format(U[i, :, :, :]))
        plt.axis("off")
        plt.subplot(num_batch, num_cols, ix + 2)
        plt.imshow(_format(q_axis[i, :, :, :]))
        plt.axis("off")

        # Confidence plots
        plt.subplot(num_batch, num_cols, ix + 3)
        plt.imshow(_format(P_ij[i, :, :]), cmap="inferno")
        # plt.clim([0, P_ij[i,:,:].max().item()])
        plt.axis("off")
        ix = ix + num_cols

    plt.tight_layout()
    return
