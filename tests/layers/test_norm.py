from unittest import TestCase

import torch

from chroma.layers.norm import MaskedBatchNorm1d


class TestBatchNorm(TestCase):
    def test_norm(self):
        device = (
            torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        B, C, L = (3, 5, 7)
        x1 = torch.randn(B, C, L).to(device)
        mean1 = x1.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / (B * L)
        var1 = ((x1 - mean1) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / (
            B * L
        )
        x2 = torch.randn(B, C, L).to(device)
        mean2 = x2.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / (B * L)
        var2 = ((x2 - mean2) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / (
            B * L
        )

        mbn = MaskedBatchNorm1d(C)
        mbn = mbn.to(device)

        # Test without mask in train
        mbn.train()
        out = mbn(x1)
        self.assertTrue(mean1.allclose(mbn.running_mean))
        self.assertTrue(var1.allclose(mbn.running_var))
        normed = (x1 - mean1) / torch.sqrt(var1 + mbn.eps) * mbn.weight + mbn.bias
        self.assertTrue(normed.allclose(out))
        out = mbn(x2)
        normed = (x2 - mean2) / torch.sqrt(var2 + mbn.eps) * mbn.weight + mbn.bias
        self.assertTrue(normed.allclose(out))
        self.assertTrue(
            mbn.running_mean.allclose((1 - mbn.momentum) * mean1 + mbn.momentum * mean2)
        )
        self.assertTrue(
            mbn.running_var.allclose((1 - mbn.momentum) * var1 + mbn.momentum * var2)
        )

        # Without mask in eval
        mbn.eval()
        out = mbn(x1)
        self.assertTrue(
            mbn.running_mean.allclose((1 - mbn.momentum) * mean1 + mbn.momentum * mean2)
        )
        self.assertTrue(
            mbn.running_var.allclose((1 - mbn.momentum) * var1 + mbn.momentum * var2)
        )
        normed = (x1 - mbn.running_mean) / torch.sqrt(
            mbn.running_var + mbn.eps
        ) * mbn.weight + mbn.bias
        self.assertTrue(normed.allclose(out))

        # Check that masking with all ones doesn't change values
        mask = x1.new_ones((B, 1, L))
        outm = mbn(x1, input_mask=mask)
        self.assertTrue(outm.allclose(out))
        mbn.eval()
        out = mbn(x2)
        outm = mbn(x2, input_mask=mask)
        self.assertTrue(outm.allclose(out))

        # With mask in train
        mask = torch.randn(B, 1, L)
        mask = mask > 0.0
        mask = mask.to(device)
        n = mask.sum()
        mean1 = (x1 * mask).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        var1 = (((x1 * mask) - mean1) ** 2).sum(dim=0, keepdim=True).sum(
            dim=2, keepdim=True
        ) / n
        mbn = MaskedBatchNorm1d(C)
        mbn = mbn.to(device)
        mbn.train()
        out = mbn(x1, input_mask=mask)
        self.assertTrue(mean1.allclose(mbn.running_mean))
        self.assertTrue(var1.allclose(mbn.running_var))
        normed = (x1 * mask - mean1) / torch.sqrt(
            var1 + mbn.eps
        ) * mbn.weight + mbn.bias
        self.assertTrue(normed.allclose(out))
        # With mask in eval
        mbn.eval()
        out = mbn(x1, input_mask=mask)
        normed = (x1 * mask - mbn.running_mean) / torch.sqrt(
            mbn.running_var + mbn.eps
        ) * mbn.weight + mbn.bias
        self.assertTrue(normed.allclose(out))
