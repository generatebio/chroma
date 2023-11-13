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

"""Layers for linear algebra.

This module contains additional pytorch layers for linear algebra operations,
such as a more parallelization-friendly implementation of eigvenalue estimation.
"""

import torch


def eig_power_iteration(A, num_iterations=50, eps=1e-5):
    """Estimate largest magnitude eigenvalue and associated eigenvector.

    This uses a simple power iteration algorithm to estimate leading
    eigenvalues, which can often be considerably faster than torch's built-in
    eigenvalue routines. All steps are differentiable and small constants are
    added to any division to preserve the stability of the gradients. For more
    information on power iteration, see
    https://en.wikipedia.org/wiki/Power_iteration.

    Args:
        A (tensor): Batch of square matrices with shape
            `(..., num_dims, num_dims)`.
        num_iterations (int, optional): Number of iterations for power
            iteration. Default: 50.
        eps (float, optional): Small number to prevent division by zero.
            Default: 1E-5.

    Returns:
        lam (tensor): Batch of estimated highest-magnitude eigenvalues with
            shape `(...)`.
        v (tensor): Associated eigvector with shape `(..., num_dims)`.
    """
    _safe = lambda x: x + eps

    dims = list(A.size())[:-1]
    v = torch.randn(dims, device=A.device).unsqueeze(-1)
    for i in range(num_iterations):
        v_prev = v
        Av = torch.matmul(A, v)
        v = Av / _safe(Av.norm(p=2, dim=-2, keepdim=True))

    # Compute eigenvalue
    v_prev = v_prev.transpose(-1, -2)
    lam = torch.matmul(v_prev, Av) / _safe(torch.abs(torch.matmul(v_prev, v)))

    # Reshape
    v = v.squeeze(-1)
    lam = lam.view(list(lam.size())[:-2])
    return lam, v


def eig_leading(A, num_iterations=50):
    """Estimate largest positive eigenvalue and associated eigenvector.

    This estimates the *most positive* eigenvalue of each matrix in a batch of
    matrices by using two consecutive power iterations with spectral shifting.

    Args:
        A (tensor): Batch of square matrices with shape
            `(..., num_dims, num_dims)`.
        num_iterations (int, optional): Number of iterations for power
            iteration. Default: 50.

    Returns:
        lam (tensor): Estimated most positive eigenvalue with shape `(...)`.
        v (tensor): Associated eigenvectors with shape `(..., num_dims)`.
    """
    batch_dims = list(A.size())[:-2]

    # First pass gets largest magnitude
    lam_1, vec_1 = eig_power_iteration(A, num_iterations)

    # Second pass guaranteed to grab most positive eigenvalue
    lam_1_abs = torch.abs(lam_1)
    lam_I = lam_1_abs.reshape(batch_dims + [1, 1]) * torch.eye(4, device=A.device).view(
        [1 for _ in batch_dims] + [4, 4]
    )
    A_shift = A + lam_I
    lam_2, vec = eig_power_iteration(A_shift, num_iterations)

    # Shift back to original specta
    lam = lam_2 - lam_1_abs
    return lam, vec
