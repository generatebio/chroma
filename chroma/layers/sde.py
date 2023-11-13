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

"""Layers for integrating Stochastic Differential Equations (SDEs).


"""


from typing import Callable, Tuple

import torch
from tqdm.autonotebook import tqdm


def sde_integrate(
    sde_func: Callable,
    y0: torch.Tensor,
    tspan: Tuple,
    N: int,
    project_func: Callable = None,
    T_grid: torch.Tensor = None,
) -> list:
    """Integrate an Ito SDE with the Euler-Maruyama method.

    args:
        sde_func (function): a function that takes in time and y and returns SDE drift and diffusion terms for the evolution of y
        y0 (torch.tensor): the initial value of y, e.g. a noised protein structure tensor
        tspan (tuple): a tuple (t_i, t_f) with t_i being the initial time and t_f being the final time for integration
        N (int): number of integration steps

    returns:
        y_trajectory (list): a list of snapshots of the evolution of y as the SDE is integrated

    """

    with torch.no_grad():
        # Integrate SDE
        y_trajectory = [y0]

        if T_grid is None:
            T_grid = torch.linspace(tspan[0], tspan[1], N + 1).to(y0.device)
        else:
            assert T_grid.shape[0] == N + 1

        y = y0
        for t0, t1 in tqdm(
            zip(T_grid[:-1], T_grid[1:]), total=N, desc="Integrating SDE"
        ):
            t = t0
            dT = t1 - t0

            f, gZ = sde_func(t, y)
            y = y + dT * f + dT.abs().sqrt() * gZ
            y = y if project_func is None else project_func(t, y)

            y_trajectory.append(y)
    return y_trajectory


def sde_integrate_heun(
    sde_func: Callable,
    y0: torch.Tensor,
    tspan: Tuple,
    N: int,
    project_func: Callable = None,
    T_grid: torch.Tensor = None,
) -> list:
    """Integrate an Ito SDE with Heun's method.

    args:
        sde_func (function): a function that takes in time and y and returns SDE drift and diffusion terms for the evolution of y
        y0 (torch.tensor): the initial value of y, e.g. a noised protein structure tensor
        tspan (tuple): a tuple (t_i, t_f) with t_i being the initial time and t_f being the final time for integration
        N (int): number of integration steps

    returns:
        y_trajectory (list): a list of snapshots of the evolution of y as the SDE is integrated

    """

    with torch.no_grad():
        # Integrate SDE
        y_trajectory = [y0]
        dT = (tspan[1] - tspan[0]) / N

        if T_grid is None:
            T_grid = torch.linspace(tspan[0], tspan[1], N + 1).to(y0.device)
        else:
            assert T_grid.shape[0] == N + 1

        y = y0

        for t0, t1 in tqdm(
            zip(T_grid[:-1], T_grid[1:]), total=N, desc="Integrating SDE"
        ):
            # for i in tqdm(range(N)):
            # t = tspan[0] + i * dT
            t = t0
            dT = t1 - t0
            f, gZ = sde_func(t, y)
            y_pred = y + dT * f + dT.abs().sqrt() * gZ
            f_pred, gZ_pred = sde_func(t, y_pred)
            y_correct = y + dT * f_pred + dT.abs().sqrt() * gZ
            y = (y_pred + y_correct) / 2.0
            y = y if project_func is None else project_func(t, y)
            y_trajectory.append(y)

    return y_trajectory
