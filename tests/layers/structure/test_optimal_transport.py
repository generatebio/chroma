import pytest
import torch

from chroma.layers.structure.optimal_transport import (
    optimize_couplings_gw,
    optimize_couplings_sinkhorn,
)


# test sinkhorn
def test_sinkhorn():
    C = torch.Tensor([[[1, 0, 0], [0, 0, 1], [0, 1, 0]]])
    assert torch.allclose(
        optimize_couplings_sinkhorn(C).argmin(-1), torch.LongTensor([[0, 2, 1]])
    )


def test_gw():
    # TODO: need a nontrivial test
    seed1 = torch.randn(4).abs()
    adj1 = torch.outer(seed1, seed1)

    Da = torch.stack([adj1, adj1])
    Db = torch.stack([adj1, adj1])

    optimize_couplings_gw(Da, Db, scale=2)
