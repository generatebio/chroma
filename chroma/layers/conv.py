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

import platform

import torch
import torch.nn.functional as F

MACHINE = platform.machine()


def filter1D_linear_decay(Z, B):
    """Apply a low-pass filter with batch-heterogeneous coefficients.

    Computes `x_i = z_i + b * x_{i-1}` where `b` varies per batch member.

    Args:
        Z (torch.Tensor): Batch of one-dimensional signals with shape `(N, W)`.
        B (torch.Tensor): Batch of coefficients with shape `(N)`.

    Returns:
        X (torch.Tensor): Result of applying linear recurrence with shape `(N, W)`.
    """

    # Build filter coefficients as powers of B
    N, W = Z.shape
    k = (W - 1) - torch.arange(W, device=Z.device)
    kernel = B[:, None, None] ** k[None, None, :]

    # Pad on left to convolve from backwards in time
    Z_pad = F.pad(Z, (W - 1, 0))[None, ...]

    # Group convolution can effectively do one filter per batch
    while True:
        X = F.conv1d(Z_pad, kernel, stride=1, padding=0, groups=N)[0, :, :]
        # on arm64 (M1 Mac) this convolution erroneously sometimes produces NaNs
        if (
            (MACHINE == "arm64")
            and torch.isnan(X).any()
            and (not torch.isnan(Z_pad).any())
            and (not torch.isnan(kernel).any())
        ):
            continue
        break
    return X
