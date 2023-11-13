import math
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from chroma.layers.basic import (
    MaybeOnehotEmbedding,
    MeanEmbedding,
    NodeProduct,
    NoOp,
    OneHot,
    PeriodicPositionalEncoding,
    PositionalEncoding,
    PositionWiseFeedForward,
    Transpose,
    TriangleMultiplication,
    Unsqueeze,
)


class TestBasicLayers(TestCase):
    def setUp(self):
        self.noop = NoOp()
        self.onehot = OneHot(n_tokens=4)
        self.transpose = Transpose(1, 2)
        self.unsqueeze = Unsqueeze(1)
        self.mean_embedding = MeanEmbedding(nn.Embedding(4, 64), use_softmax=False)
        self.periodic = PeriodicPositionalEncoding(64)
        self.pwff = PositionWiseFeedForward(64, 64)

    def test_noop(self):
        x = torch.randn(4, 2, 2)
        self.assertTrue((x == self.noop(x)).all().item())

    def test_onehot(self):
        input = torch.tensor([[0, 1, 2], [3, 0, 1]])
        onehot = self.onehot(input).transpose(1, 2)
        target = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]],
            ],
            dtype=onehot.dtype,
        )
        self.assertTrue((onehot == target).all().item())

    def test_mean_embedding(self):
        input = torch.tensor([[0, 1, 2], [3, 0, 1]])
        onehot = self.onehot(input)
        self.assertTrue(
            (self.mean_embedding(input) == self.mean_embedding(onehot.float()))
            .all()
            .item()
        )

    def test_triangle_multiplication(self):
        bs = 4
        nres = 25
        d_model = 12
        m = TriangleMultiplication(d_model=d_model)
        X = torch.randn(bs, nres, nres, d_model)
        mask = torch.ones(bs, nres, nres, 1)
        self.assertTrue(
            m(X, mask.bool()).size() == torch.Size([bs, nres, nres, d_model])
        )

    def test_node_product(self):
        bs = 4
        nres = 25
        d_model = 12
        m = NodeProduct(d_in=d_model, d_out=d_model)
        node_h = torch.randn(bs, nres, d_model)
        node_mask = torch.ones(bs, nres).bool()
        edge_mask = torch.ones(bs, nres, nres).bool()
        self.assertTrue(
            m(node_h, node_mask, edge_mask).size()
            == torch.Size([bs, nres, nres, d_model])
        )

    def test_transpose(self):
        x = torch.randn(4, 5, 2)
        self.assertTrue((x == self.transpose(x).transpose(1, 2)).all().item())

    def test_periodic(self):
        position = torch.arange(0.0, 4000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, 64, 2) * -(math.log(10000.0) / 64))
        self.assertTrue(
            (self.periodic.pe.squeeze()[:, 0::2] == torch.sin(position * div_term))
            .all()
            .item()
        )
        self.periodic(torch.randn(6, 30, 64))

    def test_pwff(self):
        x = torch.randn(4, 5, 64)
        self.assertTrue(self.pwff(x).size() == x.size())


@pytest.mark.parametrize(
    "d_model, d_input", [(2, 1), (12, 1), (12, 2), (12, 3), (12, 6)], ids=str
)
def test_positional_encoding(d_model, d_input):
    encoding = PositionalEncoding(d_model, d_input)

    for batch_shape in [(), (4,), (3, 2)]:
        inputs = torch.randn(batch_shape + (d_input,), requires_grad=True)
        outputs = encoding(inputs)
        assert outputs.shape == batch_shape + (d_model,)
        assert torch.isfinite(outputs).all()
        outputs.sum().backward()  # smoke test


def test_maybe_onehot_embedding():
    x = torch.empty(10, dtype=torch.long).random_(4)
    x_onehot = nn.functional.one_hot(x, 4).float()

    embedding = MaybeOnehotEmbedding(4, 8)
    expected = embedding(x)
    actual = embedding(x_onehot)
    assert torch.allclose(expected, actual)
