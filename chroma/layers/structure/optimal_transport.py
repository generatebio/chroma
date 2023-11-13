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

"""Layers for comparing and mapping point clouds via optimal transport.

This module contains minimalist implementations of basic optimal transport
routines which can be used to, for example, measure similarities between
point clouds of different shapes by computing optimal mappings between them.
For more information see the excellent book by Peyre,
https://arxiv.org/pdf/1803.00567.pdf
"""

import numpy as np
import torch


def optimize_couplings_sinkhorn(C, scale=1.0, iterations=10):
    """Solve entropy regularized optimized transport via Sinkhorn iteration.

    This method uses the log-domain for numerical stability.

    Args:
        C (Tensor): Batch of cost matrices with with shape `(B, I, J)`.
        scale (float, optional): Entropy regularization parameter for
            rescaling the cost matrix.
        iterations (int, optional): Number of Sinkhorn iterations.

    Returns:
        T (Tensor): Couplings map with shape `(B, I, J)`.
    """
    log_T = -C * scale

    # Initialize normalizers
    B, I, J = log_T.shape
    log_u = torch.zeros((B, I), device=log_T.device)
    log_v = torch.zeros((B, J), device=log_T.device)
    log_a = log_u - np.log(I)
    log_b = log_v - np.log(J)

    # Iterate normalizers
    for j in range(iterations):
        log_u = log_a - torch.logsumexp(log_T + log_v.unsqueeze(1), 2)
        log_v = log_b - torch.logsumexp(log_T + log_u.unsqueeze(2), 1)
    log_T = log_T + log_v.unsqueeze(1) + log_u.unsqueeze(2)
    T = torch.exp(log_T)
    return T


def optimize_couplings_gw(
    D_a, D_b, scale=200.0, iterations_outer=30, iterations_inner=10,
):
    """Gromov-Wasserstein Optimal Transport.
    https://arxiv.org/pdf/1905.07645.pdf

    Args:
        D_a (Tensor): Distance matrix describing objects in set `a` with shape `(B, I, I)`.
        D_b (Tensor): Distance matrix describing objects in set `b` with shape `(B, J, J)`.
        scale (float, optional): Entropy regularization parameter for
            rescaling the cost matrix.
        iterations_outer (int, optional): Number of outer GW iterations.
        iterations_inner (int, optional): Number of inner Sinkhorn iterations.

    Returns:
        T (Tensor): Couplings map with shape `(B, I, J)`.

    """

    # Gromov-Wasserstein Distance
    N_a = D_a.shape[1]
    N_b = D_b.shape[1]
    p_a = torch.ones_like(D_a[:, :, 0]) / N_a
    p_b = torch.ones_like(D_b[:, :, 0]) / N_b
    C_ab = (
        torch.einsum("bij,bj->bi", D_a ** 2, p_a)[:, :, None]
        + torch.einsum("bij,bj->bi", D_b ** 2, p_b)[:, None, :]
    )
    T_gw = torch.einsum("bi,bj->bij", p_a, p_b)
    for i in range(iterations_outer):
        cost = C_ab - 2.0 * torch.einsum("bik,bkl,blj->bij", D_a, T_gw, D_b)
        T_gw = optimize_couplings_sinkhorn(cost, scale, iterations=iterations_inner)

    # Compute cost
    cost = C_ab - 2.0 * torch.einsum("bik,bkl,blj->bij", D_a, T_gw, D_b)
    D_gw = (T_gw * cost).sum([-1, -2]).abs().sqrt()
    return T_gw, D_gw
