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

"""XCS represents protein structure as a tuple of PyTorch tensors.

The tensors in an XCS representation are:

    `X` (FloatTensor), the Cartesian coordinates representing the protein
        structure with shape `(num_batch, num_residues, num_atoms, 3)`. The
        `num_atoms` dimension can be one of two sizes: `num_atoms=4` for
        backbone-only structures or `num_atoms=14` for all-atom structures
        (excluding hydrogens). The first four atoms will always be
        `N, CA, C, O`, and the meaning of the optional 10 additional atom
        positions will vary based on the residue identity at
        a given position. Atom orders for each amino acid are defined in
        `constants.AA_GEOMETRY[TRIPLET_CODE]["atoms"]`.

    `C` (LongTensor), the chain map encoding per-residue chain assignments with
        shape `(num_batch, num_residues)`.The chain map codes positions as `0`
        when masked, poitive integers for chain indices, and negative integers
        to represent missing residues (of the corresponding positive integers).

    `S` (LongTensor), the sequence of the protein as alphabet indices with
        shape `(num_batch, num_residues)`. The standard alphabet is
        `ACDEFGHIKLMNPQRSTVWY`, also defined in `constants.AA20`.
"""


from functools import partial, wraps
from inspect import getfullargspec

import torch
from torch.nn import functional as F

try:
    pass
except ImportError:
    print("MST not installed!")


def validate_XCS(all_atom=None, sequence=True):
    """Decorator factory that adds XCS validation to any function.

    Args:
        all_atom (bool, optional): If True, requires that input structure
            tensors have 14 residues per atom. If False, reduces to 4 residues
            per atom. If None, applies no transformation on input structures.
        sequence (bool, optional): If True, makes sure that if S and O are both
            provided, that they match, i.e. that O is a one-hot version of S.
            If only one of S or O is provided, the other is generated, and both
            are passed.
    """

    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            args = list(args)
            arg_list = getfullargspec(func)[0]
            tensors = {}
            for var in ["X", "C", "S", "O"]:
                try:
                    if var in kwargs:
                        tensors[var] = kwargs[var]
                    else:
                        tensors[var] = args[arg_list.index(var)]
                except IndexError:  # empty args_list
                    tensors[var] = None
                except ValueError:  # variable not an argument of function
                    if not sequence and var in ["S", "O"]:
                        pass
                    else:
                        raise Exception(
                            f"Variable {var} is required by validation but not defined!"
                        )
            if tensors["X"] is not None and tensors["C"] is not None:
                if tensors["X"].shape[:2] != tensors["C"].shape[:2]:
                    raise ValueError(
                        f"X shape {tensors['X'].shape} does not match C shape"
                        f" {tensors['C'].shape}"
                    )
            if all_atom is not None and tensors["X"] is not None:
                if all_atom and tensors["X"].shape[2] != 14:
                    raise ValueError("Side chain atoms missing!")
                elif not all_atom:
                    if "X" in kwargs:
                        kwargs["X"] = tensors["X"][:, :, :4]
                    else:
                        args[arg_list.index("X")] = tensors["X"][:, :, :4]
            if sequence and (tensors["S"] is not None or tensors["O"] is not None):
                if tensors["O"] is None:
                    if "O" in kwargs:
                        kwargs["O"] = F.one_hot(tensors["S"], 20).float()
                    else:
                        args[arg_list.index("O")] = F.one_hot(tensors["S"], 20).float()
                elif tensors["S"] is None:
                    if "S" in kwargs:
                        kwargs["S"] = tensors["O"].argmax(dim=2)
                    else:
                        args[arg_list.index("S")] = tensors["O"].argmax(dim=2)
                else:
                    if not torch.allclose(tensors["O"].argmax(dim=2), tensors["S"]):
                        raise ValueError("S and O are both provided but don't match!")
            return func(*args, **kwargs)

        return new_func

    return decorator


validate_XC = partial(validate_XCS, sequence=False)
