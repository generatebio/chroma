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

import torch
import torch.nn as nn


class MaskedBatchNorm1d(nn.Module):
    """A masked version of nn.BatchNorm1d. Only tested for 3D inputs.

    Args:
        num_features (int): :math:`C` from an expected input of size
            :math:`(N, C, L)`
        eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine (bool): a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats (bool) : a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Inputs:
        x (torch.tensor): of size (batch_size, num_features, sequence_length)
        input_mask (torch.tensor or None) : (optional) of dtype (byte) or (bool) of shape (batch_size, 1, sequence_length) zeroes (or False) indicate positions that cannot contribute to computation
    Outputs:
        y (torch.tensor): of size (batch_size, num_features, sequence_length)
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, 1))
            self.register_buffer("running_var", torch.ones(num_features, 1))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, input_mask=None):
        # Calculate the masked mean and variance
        B, C, L = input.shape
        if input_mask is not None and input_mask.shape != (B, 1, L):
            raise ValueError("Mask should have shape (B, 1, L).")
        if C != self.num_features:
            raise ValueError(
                "Expected %d channels but input has %d channels"
                % (self.num_features, C)
            )
        if input_mask is not None:
            masked = input * input_mask
            n = input_mask.sum()
        else:
            masked = input
            n = B * L
        # Sum
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(
            dim=2, keepdim=True
        ) / n
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * current_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (
                torch.sqrt(self.running_var + self.eps)
            )
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed


class MaskedBatchNorm2d(nn.Module):
    """A masked version of nn.BatchNorm1d. Only tested for 3D inputs.

    Args:
        num_features (int): :math:`C` from an expected input of size
            :math:`(N, C, L)`
        eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine (bool): a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats (bool) : a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Inputs:
        x (torch.tensor): of size (batch_size, num_features, sequence_length)
        input_mask (torch.tensor or None) : (optional) of dtype (byte) or (bool) of shape (batch_size, 1, sequence_length) zeroes (or False) indicate positions that cannot contribute to computation
    Outputs:
        y (torch.tensor): of size (batch_size, num_features, sequence_length)
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features,))
            self.bias = nn.Parameter(torch.zeros(num_features,))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, 1, 1, num_features))
            self.register_buffer("running_var", torch.ones(1, 1, 1, num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, mask=None):
        # Calculate the masked mean and variance
        B, L, L, C = input.size()
        if mask is not None:
            if mask.dim() != 4:
                raise ValueError(
                    f"Input mask must have four dimensions, but has {mask.dim()}"
                )
            b, l, l, d = mask.size()
            if (b != B) or (l != L):
                raise ValueError(
                    f"Input mask must have shape {(B, L, L, 1)} or {(B, L, L, C)} to match input."
                )
            if d == 1:
                mask = mask.expand(input.size())

        if C != self.num_features:
            raise ValueError(
                "Expected %d channels but input has %d channels"
                % (self.num_features, C)
            )

        if mask is None:
            mask = input.new_ones(input.size())

        masked = input * mask
        n = mask.sum(dim=(0, 1, 2), keepdim=True)
        masked_sum = (masked).sum(dim=(0, 1, 2), keepdim=True)

        current_mean = masked_sum / n
        current_var = (mask * (masked - current_mean).pow(2)).sum(
            dim=(0, 1, 2), keepdim=True
        ) / n
        # Update running stats
        with torch.no_grad():
            if self.track_running_stats and self.training:
                if self.num_batches_tracked == 0:
                    self.running_mean = current_mean.detach()
                    self.running_var = current_var.detach()
                else:
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * current_mean.detach()
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * current_var.detach()
                self.num_batches_tracked += 1
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (
                torch.sqrt(self.running_var + self.eps)
            )
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias

        normed = normed * mask
        return normed


class NormedReductionLayer(nn.Module):
    """A ReductionLayer with LayerNorms after the hidden layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def reduce(self, x, mask):
        masked_x = x * mask
        mean_x = masked_x.sum(dim=1) / torch.sum(mask, dim=1)
        return mean_x

    def forward(self, x, mask):
        reduced_x = self.norm1(self.reduce(x, mask))
        h = self.norm2(self.hidden(reduced_x))
        return self.output(self.relu(h))
