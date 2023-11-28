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

from __future__ import annotations

import copy
import os
import tempfile
from typing import List, Optional, Tuple, Union

import nglview as nv
import torch

import chroma.utility.polyseq as polyseq
from chroma.constants import CHAIN_ALPHABET, PROTEIN_TOKENS
from chroma.data.system import System, SystemEntity


class Protein:
    """
    Protein: A utility class for managing proteins within the Chroma ecosystem.

    The Protein class offers a suite of methods for loading, saving, transforming, and viewing protein structures
    and trajectories from a variety of input sources such as PDBID, CIF files, and XCS representations.

    Attributes:
        sys (System): A protein system object used for various molecular operations.
        device (str): Specifies the device on which tensors are managed. Defaults to `cpu`.
    """

    sys: System
    device: str = "cpu"

    def __new__(cls, *args, **kwargs):
        """Handles automatic loading of the protein based on the input.
        Specifically deals with XCS

        Args:
            protein_input (_type_): _description_
        """

        if len(args) == 1 and isinstance(args[0], System):
            return cls.from_system(*args, **kwargs)

        elif len(args) == 3:  # 3 Tensor Arguments
            X, C, S = args
            assert isinstance(
                C, torch.Tensor
            ), f"arg[1] must be a chain (C) torch.Tensor, but get {type(C)}"
            assert isinstance(
                S, torch.Tensor
            ), f"arg[2] must be a sequence (S) torch.Tensor, but get {type(S)}"
            if isinstance(X, list):
                assert all(
                    isinstance(x, torch.Tensor) for x in X
                ), "arg[0] must be an X torch.Tensor or a list of X torch.Tensors"
                return cls.from_XCS_trajectory(X, C, S)
            elif isinstance(X, torch.Tensor):
                return cls.from_XCS(X, C, S)
            else:
                raise TypeError(
                    f"X must be a list of torch.Tensor that respects XCS format, but get {type(X), type(C), type(S)}"
                )

        elif len(args) == 1 and isinstance(args[0], str):
            if args[0].lower().startswith("s3:"):
                raise NotImplementedError(
                    "download of cifs or pdbs from s3 not supported."
                )

            if args[0].endswith(".cif"):
                return cls.from_CIF(*args, **kwargs)

            elif args[0].endswith(".pdb"):
                return cls.from_PDB(*args, **kwargs)

            else:  # PDB or Sequence String
                # Check if it is a valid PDB
                import requests

                url = f"https://data.rcsb.org/rest/v1/core/entry/{args[0]}"
                VALID_PDBID = requests.get(url).status_code == 200
                VALID_SEQUENCE = all([s in PROTEIN_TOKENS for s in args[0]])

                if VALID_PDBID:
                    # This only works if connected to the internet,
                    # so maybe better status checking will help here
                    if VALID_PDBID and VALID_SEQUENCE:
                        raise Warning(
                            "Ambuguous input, this is both a valid Sequence string and"
                            " a valid PDBID. Interpreting as a PDBID, if you wish to"
                            " initialize as a sequence string please explicitly"
                            " initialize as Protein.from_sequence(MY_SEQUENCE)."
                        )
                    return cls.from_PDBID(*args, **kwargs)
                elif VALID_SEQUENCE:
                    return cls.from_sequence(*args, **kwargs)
                else:
                    raise NotImplementedError(
                        "Could Not Identify a valid input Type. See docstring for"
                        " details."
                    )
        else:
            raise NotImplementedError(
                "Inputs must either be a 3-tuple of XCS tensors, or a single string"
            )

    @classmethod
    def from_system(cls, system: System, device: str = "cpu") -> Protein:
        protein = super(Protein, cls).__new__(cls)
        protein.sys = system
        protein.device = device
        return protein

    @classmethod
    def from_XCS(cls, X: torch.Tensor, C: torch.Tensor, S: torch.Tensor) -> Protein:
        """
        Create a Protein object from XCS representations.

        Args:
            X (torch.Tensor): A 4D tensor representing atomic coordinates of proteins.
                            Dimensions are `(batch, residues, atoms (4 or 14), coordinates (3))`.
            C (torch.Tensor): A chain label tensor of shape `(batch, residues)`. Values are integers.
                            Sign of the value indicates presence (+) or absence (-) of structural
                            information for that residue. Magnitude indicates which chain the residue belongs to.
            S (torch.Tensor): A sequence information tensor of shape `(batch, residues)`. Contains
                            non-negative integers representing residue types at each position.

        Returns:
            Protein: Initialized Protein object from the given XCS representation.
        """
        protein = super(Protein, cls).__new__(cls)
        protein.sys = System.from_XCS(X, C, S)
        protein.device = X.device
        return protein

    @classmethod
    def from_XCS_trajectory(
        cls, X_traj: List[torch.Tensor], C: torch.Tensor, S: torch.Tensor
    ) -> Protein:
        """
        Initialize a Protein object from a trajectory of XCS representations.

        Args:
            X_traj (List[torch.Tensor]): List of X tensor representations over time. Each tensor represents atomic
                                        coordinates of proteins with dimensions `(batch, residues, atoms (4 or 14), coordinates (3))`.
            C (torch.Tensor): A chain label tensor of shape `(batch, residues)`. Values are integers.
                            Sign of the value indicates presence (+) or absence (-) of structural
                            information for that residue. Magnitude indicates which chain the residue belongs to.
            S (torch.Tensor): A sequence information tensor of shape `(batch, residues)`. Contains
                            non-negative integers representing residue types at each position.

        Returns:
            Protein: Protein object initialized from the XCS trajectory.
        """
        protein = super(Protein, cls).__new__(cls)
        protein.sys = System.from_XCS(X_traj[0], C, S)
        protein.device = C.device
        for X in X_traj[1:]:
            protein.sys.add_model_from_X(X[C > 0])
        return protein

    @classmethod
    def from_PDB(cls, input_file: str, device: str = "cpu") -> Protein:
        """
        Load a Protein object from a provided PDB file.

        Args:
            input_file (str): Path to the PDB file to be loaded.
            device (str, optional): The device for tensor operations. Defaults to 'cpu'.

        Returns:
            Protein: Initialized Protein object from the provided PDB file.
        """
        protein = super(Protein, cls).__new__(cls)
        protein.sys = System.from_PDB(input_file)
        protein.device = device
        return protein

    @classmethod
    def from_CIF(
        cls, input_file: str, canonicalize: bool = True, device: str = "cpu"
    ) -> Protein:
        """
        Load a Protein object from a provided CIF format.

        Args:
            input_file (str): Path to the CIF file to be loaded.
            device (str, optional): The device for tensor operations. Defaults to 'cpu'.

        Returns:
            Protein: Initialized Protein object from the provided CIF file.
        """
        protein = super(Protein, cls).__new__(cls)
        protein.sys = System.from_CIF(input_file)
        protein.device = device
        if canonicalize:
            protein.canonicalize()
        return protein

    @classmethod
    def from_PDBID(
        cls, pdb_id: str, canonicalize: bool = True, device: str = "cpu"
    ) -> Protein:
        """
        Load a Protein object using its PDBID by fetching the corresponding CIF file from the Protein Data Bank.

        This method downloads the CIF file for the specified PDBID, processes it to create a Protein object,
        and then deletes the temporary CIF file.

        Args:
            pdb_id (str): The PDBID of the protein to fetch.
            canonicalize (bool, optional): If set to True, the protein will be canonicalized post-loading. Defaults to True.
            device (str, optional): The device for tensor operations. Defaults to 'cpu'.

        Returns:
            Protein: An instance of the Protein class initialized from the fetched CIF file corresponding to the PDBID.
        """
        from os import unlink

        from chroma.utility.fetchdb import RCSB_file_download

        file_cif = os.path.join(tempfile.gettempdir(), f"{pdb_id}.cif")
        RCSB_file_download(pdb_id, ".cif", file_cif)
        protein = cls.from_CIF(file_cif, canonicalize=canonicalize, device=device)
        unlink(file_cif)
        return protein

    @classmethod
    def from_sequence(
        cls, chains: Union[List[str], str], device: str = "cpu"
    ) -> Protein:
        """
        Load a protein object purely from Sequence with no structural content.

        Args:
            chains (Union[List[str],str]): a list of sequence strings, or a sequence string to create the protein.
            device (str, optional): which device for torch outputs should be used. Defaults to "cpu".

        Returns:
            Protein: An instance of the Protein class initialized a sequence or list of sequences.
        """

        if isinstance(chains, str):
            chains = [chains]

        system = System("system")
        for c_ix, seq in enumerate(chains):
            chain_id = CHAIN_ALPHABET[c_ix + 1]
            chain = system.add_chain(chain_id)

            # Populate the Chain
            three_letter_sequence = []
            for s_ix, s in enumerate(seq):
                resname = polyseq.to_triple(s)
                three_letter_sequence.append(resname)
                chain.add_residue(resname, s_ix + 1, "")

            # Add Entity
            sys_entity = SystemEntity(
                "polymer",
                f"Sequence Chain {chain_id}",
                "polypeptide(L)",
                three_letter_sequence,
                [False] * len(three_letter_sequence),
            )
            system.add_new_entity(sys_entity, [c_ix])

        protein = super(Protein, cls).__new__(cls)
        protein.sys = system
        protein.device = device
        return protein

    def to_CIF(self, output_file: str, force: bool = False) -> None:
        """
        Save the current Protein object to a file in CIF format.

        Args:
            output_file (str): The path where the CIF file should be saved.

        """
        if output_file.lower().startswith("s3:"):
            raise NotImplementedError("cif output to an s3 bucket not supported.")
        else:
            self.sys.to_CIF(output_file)

    def to_PDB(self, output_file: str, force: bool = False) -> None:
        """
        Save the current Protein object to a file in PDB format.

        Args:
            output_file (str): The path where the PDB file should be saved.
        """
        if output_file.lower().startswith("s3:"):
            raise NotImplementedError("pdb output to an s3 bucket not supported.")

        else:
            self.sys.to_PDB(output_file)

    def to_XCS(
        self, all_atom: bool = False, device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the current Protein object to its XCS tensor representations.

        Args:
            all_atom (bool, optional): Indicates if all atoms should be considered in the conversion. Defaults to False.
            device (str, optional): the device to export XCS tensors to. If not specified uses the device property
                set in the class. Default None.

        Returns:
            X (torch.Tensor): A 4D tensor representing atomic coordinates of proteins with dimensions
                                `(batch, residues, atoms (4 or 14), coordinates (3))`.
            C (torch.Tensor): A chain label tensor of shape `(batch, residues)`. Values are integers. Sign of
                                the value indicates presence (+) or absence (-) of structural information for that residue.
                                Magnitude indicates which chain the residue belongs to.
            S (torch.Tensor): A sequence information tensor of shape `(batch, residues)`. Contains non-negative
                                integers representing residue types at each position.
        """

        if device is None:
            device = self.device

        X, C, S = [tensor.to(device) for tensor in self.sys.to_XCS(all_atom=all_atom)]

        return X, C, S

    def to_XCS_trajectory(
        self,
        device: Optional[str] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Convert the current Protein object to its XCS tensor representations over a trajectory.

        Args:
            device (str, optional): the device to export XCS tensors to. If not specified uses the device property
                set in the class. Default None.

        Returns:
            X_traj (List[torch.Tensor]): List of X tensor representations over time. Each tensor represents atomic
                                        coordinates of proteins with dimensions `(batch, residues, atoms (4 or 14), coordinates (3))`.
            C (torch.Tensor): A chain label tensor of shape `(batch, residues)`. Values are integers. Sign of
                            the value indicates presence (+) or absence (-) of structural information for that residue.
                            Magnitude indicates which chain the residue belongs to.
            S (torch.Tensor): A sequence information tensor of shape `(batch, residues)`. Contains non-negative
                            integers representing residue types at each position.
        """
        X, C, S = [], None, None
        for i in range(self.sys.num_models()):
            self.sys.swap_model(i)
            if i == 0:
                X_frame, C, S, loc_indices = self.sys.to_XCS(get_indices=True)
            else:
                X_frame.flatten(0, 2)[:] = torch.from_numpy(
                    self.sys._locations["coor"][loc_indices, 0:3]
                )
            X.append(X_frame.clone())
            self.sys.swap_model(i)
        X = torch.cat(X)

        if device is None:
            device = self.device

        Xtraj, C, S = [tensor.to(device) for tensor in [X, C, S]]
        return [each.unsqueeze(0) for each in Xtraj], C, S

    def to(self, file_path: str, force: bool = False) -> None:
        """
        General Export for the Protein Class

        This method allows for export in pdf or cif based on the file extension.
        explicit saving is still available with the respective export methods.

        Args:
            device (str): The desired device for tensor operations, e.g., 'cpu' or 'cuda'.
        """
        if file_path.lower().endswith(".pdb"):
            self.to_PDB(file_path, force=force)
        elif file_path.lower().endswith(".cif"):
            self.to_CIF(file_path, force=force)
        else:
            raise NotImplementedError(
                "file path must end with either *.cif or *.pdb for export."
            )

    def length(self, structured: bool = False) -> None:
        """
        Retrieve the length of the protein.

        Args:
            structured (bool, optional): If set to True, returns the residue size of the structured part of the protein.
                                        Otherwise, returns the length of the entire protein. Defaults to False.

        Returns:
            int: Length of the protein or its structured part based on the 'structured' argument.
        """
        if structured:
            return self.sys.num_structured_residues()
        return self.sys.num_residues()

    __len__ = length

    def canonicalize(self) -> None:
        """
        Canonicalize the protein's backbone geometry.

        This method processes the protein to ensure it conforms to a canonical form.
        """
        self.sys.canonicalize_protein(
            level=2,
            drop_coors_unknowns=True,
            drop_coors_missing_backbone=True,
        )

    def sequence(self, format: str = "one-letter-string") -> Union[List[str], str]:
        """
        Retrieve the sequence of the protein in the specified format.

        Args:
            format (str, optional): The desired format for the sequence. Can be 'three-letter-list' or 'one-letter-string'.
                                    Defaults to 'one-letter-string'.

        Returns:
            Union[List[str], str]: The protein sequence in the desired format.

        Raises:
            Exception: If an unknown sequence format is provided.
        """
        if format == "three-letter-list":
            return list(self.sys.sequence())
        elif format == "one-letter-string":
            return self.sys.sequence("one-letter-string")
        else:
            raise Exception(f"unknown sequence format {format}")

    def display(self, representations: list = []) -> None:
        """
        Display the protein using the provided representations in NGL view.

        Args:
            representations (list, optional): List of visual representations to use in the display. Defaults to an empty list.

        Returns:
            viewer: A viewer object for interactive visualization.
        """
        from chroma.utility.ngl import SystemTrajectory, view_gsystem

        if self.sys.num_models() == 1:
            viewer = view_gsystem(self.sys)
            for rep in representations:
                viewer.add_representation(rep)

        else:
            t = SystemTrajectory(self)
            viewer = nv.NGLWidget(t)
        return viewer

    def _ipython_display_(self):
        display(self.display())

    def __str__(self):
        """Define Print Behavior
        Return Protein Sequence Along with some useful statistics.
        """
        protein_string = f"Protein: {self.sys.name}\n"
        for chain in self.sys.chains():
            if chain.sequence is not None:
                protein_string += (
                    f"> Chain {chain.cid} ({len(chain.sequence())} residues)\n"
                )
                protein_string += "".join(
                    [polyseq.to_single(s) for s in chain.sequence()]
                )
                protein_string += "\n\n"

        return protein_string

    def get_mask(self, selection: str) -> torch.Tensor:
        """
        Generate a mask tensor based on the provided residue selection.

        Args:
            selection (str): A selection string to specify which residues should be included in the mask.

        Returns:
            torch.Tensor: A mask tensor of shape `(1, protein length)`, where positions corresponding to selected residues have a value of 1.
        """
        residue_gtis = self.sys.select_residues(selection, gti=True)
        D = torch.zeros(1, self.sys.num_residues(), device=self.device)
        for gti in residue_gtis:
            D[0, gti] = 1
        return D

    def __copy__(self):
        new_system = copy.copy(self.sys)
        device = self.device
        return Protein(new_system, device=device)

    def __deepcopy__(self, memo):
        new_system = copy.deepcopy(self.sys)
        device = self.device
        return Protein(new_system, device=device)
