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

"""Layers for euclidean symmetry group operations 

This module contains pytorch layers for symmetry operations for point groups (Cyclic, Dihedral, Tetrahedral, Octahedral and Icosahedral)
"""

import itertools
import math
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from chroma.layers.structure import backbone

TAU = 0.5 * (1 + math.sqrt(5))

ROT_DICT = {
    "O": [
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    ],
    "T": [
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    ],
}


def get_point_group(group: str) -> torch.Tensor:
    """get representation of group elements at torch.Tensor

    Args:
        group (str): group names, selecting from {"C_{n}" , "D_{n}", "T", "O", "I" }

    Returns:
        torch.Tensor: rotation matrices for queried point groups
    """
    if group.startswith("C"):
        n = group.split("_")[1]
        G = get_Cn_groups(int(n))
    elif group.startswith("D"):
        n = group.split("_")[1]
        G = get_Dn_groups(int(n))
    elif group == "I":
        G = get_I_rotations()
    elif group == "O" or group == "T":
        G = torch.Tensor(np.array(ROT_DICT[group]))
    else:
        raise ValueError("{ } not available".format(group))

    return G


def get_Cn_groups(n: int) -> torch.Tensor:
    """get rotation matrices for Cyclic groups

    Args:
        n (int): symmetry order

    Returns:
        torch.Tensor: n x 3 x 3
    """
    G = []
    for ri in range(n):
        cos_phi = np.round(np.cos(ri * np.pi * 2 / n), 4)
        sin_phi = np.round(np.sin(ri * np.pi * 2 / n), 4)

        g = np.array(
            [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
        )
        G.append(np.round(g, 4))

    return torch.Tensor(np.array(G))


def get_Dn_groups(n: int) -> torch.Tensor:
    """get rotation matrices for Dihedral groups

    Args:
        n (int): symmetry order

    Returns:
        torch.Tensor: 2n x 3 x 3
    """
    cos_phi = np.round(
        np.cos(np.pi * 2 / n), 8
    )  # unify the choice of # of decimals to keep
    sin_phi = np.round(np.sin(np.pi * 2 / n), 8)

    rot_generator = np.array(
        [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]]
    )

    b = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    G = []
    c = rot_generator
    for itr in range(n):
        c_new = c @ rot_generator
        G.append(c_new)
        c = c_new
        G.append(b @ c)
    return torch.Tensor(np.array(G))


def get_I_rotations(tree_depth: int = 5) -> torch.Tensor:
    """get rotation matrices for the Icosahedral group (I)

    Returns:
        torch.Tensor: 60 x 3 x 3
    """

    tree_depth = 5  # tree traverse depth

    g1 = torch.Tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    g2 = torch.Tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    g3 = torch.Tensor(
        [
            [0.5, -0.5 * TAU, 0.5 / TAU],
            [0.5 * TAU, 0.5 / TAU, -0.5],
            [0.5 / TAU, 0.5, 0.5 * TAU],
        ]
    )
    # g4 = torch.Tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) reflection operation
    generators = [g1, g2, g3]

    gen1 = generators
    gen2 = generators

    sym_ops = []

    # todo: there is probably better way to do this
    # brute force search by traversing the Caley graph
    for itr in range(tree_depth):
        mat_prod = [mat_pair[0] @ mat_pair[1] for mat_pair in product(gen1, gen2)]
        sym_ops += mat_prod
        gen1 = mat_prod

    # find unique rotation matrices
    sym_ops = torch.unique(torch.round(torch.stack(sym_ops, dim=0), decimals=6), dim=0)

    return sym_ops


def subsample(
    X: torch.Tensor,
    C: torch.Tensor,
    G: torch.Tensor,
    knbr: int,
    seed_idx: Optional[int] = None,
):
    """generate substructures based on distances between subunit COM

    Args:
        X (torch.Tensor): structures
        C (torch.Tensor): chain map
        G (torch.Tensor): rotation matrices
        knbr (int): number of nearest neighbors
        seed_idx (int, optional): seed idx, this will be randomly selected if set to None. Defaults to None.

    Returns:
        tuple: substructure coordinates, chain map, indices associated with all the substructure chains, seed idx
    """

    if knbr > G.shape[0] - 1:
        knbr = G.shape[0] - 1

    G_order = G.shape[0]
    X_chain_com = X.reshape(1, G.shape[0], -1, 3).mean(-2)

    if seed_idx is None:
        seed_idx = torch.randint(0, G.shape[0], (1,)).item()

    Dis_chain = (
        (X_chain_com.unsqueeze(-2) - X_chain_com.unsqueeze(-3)).pow(2).sum(-1).sqrt()
    )

    subdomain_idx = Dis_chain[0, seed_idx].topk(knbr + 1, largest=False)[1]

    X_subdomain = X.reshape(1, G.shape[0], -1, 4, 3)[:, subdomain_idx]
    X_subdomain = X_subdomain.reshape(1, -1, 4, 3)
    C_subdomain = C.reshape(1, G.shape[0], -1)[:, : knbr + 1, :].reshape(1, -1)

    return X_subdomain, C_subdomain, subdomain_idx, seed_idx


def symmetrize_XCS(
    X: torch.Tensor,
    C: torch.LongTensor,
    S: torch.LongTensor,
    G: torch.Tensor,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """Symmetrize a protein structure with a given symmetry group.

    This function takes a data tensor X, a chain tensor C, a Sequence Tensor S, and a symmetry group tensor G. The function returns a symmetrized data tensor X_complex, a modified chain tensor C_complex, and a replicated Sequence Tensor S_complex. The function uses the torch.einsum function to apply the symmetry group G to each chain in X and concatenate them into X_complex. The function also modifies the chain labels in C by multiplying them by the symmetry index and concatenates them into C_complex. The function also replicates the Sequence Tensor S by the number of symmetry elements and concatenates them into S_complex.

    Args:
        X (torch.Tensor): Data tensor with shape `(batch_size, num_residues, 4, 3)`.
        C (torch.LongTensor): Chain tensor with shape `(batch_size, num_residues)`.
        S (torch.LongTensor): Sequence tensor with shape `(batch_size, num_residues)`.
        G (torch.Tensor): Symmetry group tensor with shape `(n_sym, 3, 3)`.
        device (str, optional): The device to use for computation. Defaults to "cpu".

    Returns:
        torch.Tensor: Symmetrized data tensor with shape `(batch_size, num_residues * n_sym, 4, 3)`.
        torch.LongTensor: Modified chain tensor with shape `(batch_size, num_residues * n_sym)`.
        torch.LongTensor: Modified Sequence tensor with shape `(batch_size * n_sym,)`.
    """
    G = G.to(S.device)
    X_complex = torch.einsum("gij,bnlj->gnli", G, X).reshape(1, -1, 4, 3).to(device)
    C_complex = torch.cat([C * (i + 1) for i in range(G.shape[0])], 1).to(device)
    S_complex = torch.cat([S for i in range(G.shape[0])], 1).to(device)
    return X_complex, C_complex, S_complex
