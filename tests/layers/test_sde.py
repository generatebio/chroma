from functools import partial

import pytest
import torch

from chroma.layers.sde import sde_integrate, sde_integrate_heun


@pytest.fixture
def y0():
    # try multiple 1D trajectories, then take mean and variance in testing
    return torch.zeros(10000)


@pytest.fixture
def tspan():
    return (0.5, 0.3)


@pytest.fixture
def N():
    return 200


@pytest.fixture
def exp_mean(y0, tspan):
    return torch.Tensor(y0 + (tspan[1] - tspan[0]) / 2).mean()


@pytest.fixture
def exp_var(tspan):
    deltat = tspan[1] - tspan[0]
    # variance contributions arising from drift and diffusion, respectively
    return torch.Tensor([deltat ** 2 / 12 + abs(deltat) / 6])


def sde_sample_func(t, y):
    f = torch.ones_like(y)
    gZ = torch.randn(y.shape)
    return f, gZ


def test_sde_integrate(y0, tspan, N, exp_mean, exp_var):
    y_trajectory = torch.stack(sde_integrate(sde_sample_func, y0, tspan, N), dim=-1)
    assert torch.allclose(torch.mean(y_trajectory, dim=-1).mean(), exp_mean, rtol=5e-2)
    assert torch.allclose(torch.var(y_trajectory, dim=-1).mean(), exp_var, rtol=5e-2)


def test_sde_integrate_heun(y0, tspan, N, exp_mean, exp_var):
    y_trajectory = torch.stack(
        sde_integrate_heun(sde_sample_func, y0, tspan, N), dim=-1
    )
    assert torch.allclose(torch.mean(y_trajectory, dim=-1).mean(), exp_mean, rtol=5e-2)
    assert torch.allclose(torch.var(y_trajectory, dim=-1).mean(), exp_var, rtol=5e-2)
