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

"""Layers for computing sequence complexities.
"""

import numpy as np
import torch
import torch.nn.functional as F

from chroma.constants import AA20
from chroma.layers.graph import collect_neighbors


def compositions(S: torch.Tensor, C: torch.LongTensor, w: int = 30):
    """Compute local compositions per residue.

    Args:
        S (torch.Tensor): Sequence tensor with shape `(num_batch, num_residues)`
            (long) or `(num_batch, num_residues, num_alphabet)` (float).
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        w (int, optional): Window size.

    Returns:
        P (torch.Tensor): Local compositions with shape
            `(num_batch, num_residues - w + 1, num_alphabet)`.
        N (torch.Tensor): Local counts with shape
            `(num_batch, num_residues - w + 1, num_alphabet)`.
        mask_P (torch.Tensor): Mask with shape
            `(num_batch, num_residues - w + 1)`.
    """
    device = S.device
    Q = len(AA20)
    mask_i = (C > 0).float()
    if len(S.shape) == 2:
        S = F.one_hot(S, Q)

    # Build neighborhoods and masks
    S_onehot = mask_i[..., None] * S
    kx = torch.arange(w, device=S.device) - w // 2
    edge_idx = (
        torch.arange(S.shape[1], device=S.device)[None, :, None] + kx[None, None, :]
    )
    mask_ij = (edge_idx > 0) & (edge_idx < S.shape[1])
    edge_idx = edge_idx.clamp(min=0, max=S.shape[1] - 1)
    C_i = C[..., None]
    C_j = collect_neighbors(C_i, edge_idx)[..., 0]
    mask_ij = (mask_ij & C_j.eq(C_i) & (C_i > 0) & (C_j > 0)).float()

    # Sum neighborhood composition
    S_j = mask_ij[..., None] * collect_neighbors(S_onehot, edge_idx)
    N = S_j.sum(2)

    num_N = N.sum(-1, keepdims=True)
    P = N / (num_N + 1e-5)
    mask_i = ((num_N[..., 0] > 0) & (C > 0)).float()
    mask_ij = mask_i[..., None] * mask_ij
    return P, N, edge_idx, mask_i, mask_ij


def complexity_lcp(
    S: torch.LongTensor,
    C: torch.LongTensor,
    w: int = 30,
    entropy_min: float = 2.32,
    method: str = "naive",
    differentiable=True,
    eps: float = 1e-5,
    min_coverage=0.9,
    # entropy_min: float = 2.52,
    # method = "chao-shen"
) -> torch.Tensor:
    """Compute the Local Composition Perplexity metric.

    Args:
        S (torch.Tensor): Sequence tensor with shape `(num_batch, num_residues)`
            (index tensor) or `(num_batch, num_residues, num_alphabet)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        w (int): Window size.
        grad_pseudocount (float): Pseudocount for stabilizing entropy gradients
            on backwards pass.
        eps (float): Small number for numerical stability in division and logarithms.

    Returns:
        U (torch.Tensor): Complexities with shape `(num_batch)`.
    """

    # adjust window size based on sequence length
    if S.shape[1] < w:
        w = S.shape[1]

    P, N, edge_idx, mask_i, mask_ij = compositions(S, C, w)

    # Only count windows with `min_coverage`
    min_N = int(min_coverage * w)
    mask_coverage = N.sum(-1) > int(min_coverage * w)

    H = estimate_entropy(N, method=method)
    U = mask_coverage * (torch.exp(H) - np.exp(entropy_min)).clamp(max=0).square()

    # Compute entropy as a function of perturbed counts
    if differentiable and len(S.shape) == 3:
        # Compute how a mutation changes entropy for each neighbor
        N_neighbors = collect_neighbors(N, edge_idx)
        mask_coverage_j = collect_neighbors(mask_coverage[..., None], edge_idx)
        N_ij = (N_neighbors - S[:, :, None, :])[..., None, :] + torch.eye(
            N.shape[-1], device=N.device
        )[None, None, None, ...]
        N_ij = N_ij.clamp(min=0)
        H_ij = estimate_entropy(N_ij, method=method)
        U_ij = (torch.exp(H_ij) - np.exp(entropy_min)).clamp(max=0).square()
        U_ij = mask_ij[..., None] * mask_coverage_j * U_ij
        U_differentiable = (U_ij.detach() * S[:, :, None, :]).sum([-1, -2])
        U = U.detach() + U_differentiable - U_differentiable.detach()

    U = (mask_i * U).sum(1)
    return U


def complexity_scores_lcp_t(
    t,
    S: torch.LongTensor,
    C: torch.LongTensor,
    idx: torch.LongTensor,
    edge_idx_t: torch.LongTensor,
    mask_ij_t: torch.Tensor,
    w: int = 30,
    entropy_min: float = 2.515,
    eps: float = 1e-5,
    method: str = "chao-shen",
) -> torch.Tensor:
    """Compute local LCP scores for autoregressive decoding."""
    Q = len(AA20)
    O = F.one_hot(S, Q)
    O_j = collect_neighbors(O, edge_idx_t)
    idx_i = idx[:, t, None]
    C_i = C[:, t, None]
    idx_j = collect_neighbors(idx[..., None], edge_idx_t)[..., 0]
    C_j = collect_neighbors(C[..., None], edge_idx_t)[..., 0]

    # Sum valid neighbor counts
    is_near = (idx_i - idx_j).abs() <= w / 2
    same_chain = C_i == C_j
    valid_ij_t = (is_near * same_chain * (mask_ij_t > 0)).float()[..., None]
    N_k = (valid_ij_t * O_j).sum(-2)

    # Compute counts under all possible extensions
    N_k = N_k[:, :, None, :] + torch.eye(Q, device=N_k.device)[None, None, ...]

    H = estimate_entropy(N_k, method=method)
    U = -(torch.exp(H) - np.exp(entropy_min)).clamp(max=0).square()
    return U


def estimate_entropy(
    N: torch.Tensor, method: str = "chao-shen", eps: float = 1e-11
) -> torch.Tensor:
    """Estimate entropy from counts.

        See Chao, A., & Shen, T. J. (2003) for more details.

    Args:
        N (torch.Tensor): Tensor of counts with shape `(..., num_bins)`.

    Returns:
        H (torch.Tensor): Estimated entropy with shape `(...)`.
    """
    N = N.float()
    N_total = N.sum(-1, keepdims=True)
    P = N / (N_total + eps)

    if method == "chao-shen":
        # Estimate coverage and adjusted frequencies
        singletons = N.long().eq(1).sum(-1, keepdims=True).float()
        C = 1.0 - singletons / (N_total + eps)
        P_adjust = C * P
        P_inclusion = (1.0 - (1.0 - P_adjust) ** N_total).clamp(min=eps)
        H = -(P_adjust * torch.log(P_adjust.clamp(min=eps)) / P_inclusion).sum(-1)
    elif method == "miller-maddow":
        bins = (N > 0).float().sum(-1)
        bias = (bins - 1) / (2 * N_total[..., 0] + eps)
        H = -(P * torch.log(P + eps)).sum(-1) + bias
    elif method == "laplace":
        N = N.float() + 1 / N.shape[-1]
        N_total = N.sum(-1, keepdims=True)
        P = N / (N_total + eps)
        H = -(P * torch.log(P)).sum(-1)
    else:
        H = -(P * torch.log(P + eps)).sum(-1)
    return H
