import copy
import tempfile
from pathlib import Path

import pytest

import chroma
from chroma.data.protein import Protein

BASE_PATH = str(Path(chroma.__file__).parent.parent)
PROTEIN_SINGLE_CHAIN = BASE_PATH + "/tests/resources/4kw4.cif"
PROTEIN_COMPLEX = BASE_PATH + "/tests/resources/3hn3.cif"
CIF_TRAJECTORY = BASE_PATH + "/tests/resources/chroma_trajectory.cif"
SEQUENCE = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
PDB_ID = "1B9C"

TESTS = [PROTEIN_SINGLE_CHAIN, PROTEIN_COMPLEX, SEQUENCE, PDB_ID]


@pytest.mark.parametrize("protein_path", TESTS)
def test_Protein(protein_path):
    # Loading Smoke Tests
    if protein_path.endswith(".pdb"):
        protein = Protein.from_PDB(protein_path)
    elif protein_path.endswith(".cif"):
        protein = Protein.from_CIF(protein_path)
    elif len(protein_path) == 4:
        protein = Protein.from_PDBID(protein_path)
    else:  # Protein Sequence Input
        protein = Protein.from_sequence(protein_path)

    # Selection Smoke Test
    # Select all structured residues
    D = protein.get_mask("all").bool()

    # Method Smoke Tests
    protein.canonicalize()
    protein.sequence()
    len(protein)
    protein.display()

    # Cycles save / load /validate
    X, C, S = protein.to_XCS()

    # XCS
    xcs_cycle_protein = Protein.from_XCS(X, C, S)
    Xt, Ct, St = xcs_cycle_protein.to_XCS()
    assert (Xt == X).all() and (Ct == C).all() and (St == S).all()

    # CIF
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=True) as temp_file:
        protein.to_CIF(temp_file.name)
        Xt, Ct, St = protein.from_CIF(temp_file.name).to_XCS()
        assert (Xt == X).all() and (Ct == C).all() and (St == S).all()

    # PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as temp_file:
        protein.to_PDB(temp_file.name)
        structured_residues = protein.sys.num_structured_residues()
        round_trip_protein = protein.from_PDB(temp_file.name)
        assert len(round_trip_protein) == structured_residues

    # smoke test copy behavior
    copy.copy(protein)
    copy.deepcopy(protein)


def compare_proteins(A, B):
    A, B = A.sys, B.sys
    if (
        A.num_chains() != B.num_chains()
        or A.num_residues() != B.num_residues()
        or A.num_atoms() != B.num_atoms()
        or A.num_atom_locations() != B.num_atom_locations()
        or A.num_structured_residues() != B.num_structured_residues()
    ):
        return False

    for cA, cB in zip(A.chains(), B.chains()):
        if (
            cA.num_residues() != cB.num_residues()
            or cA.cid != cB.cid
            or cA.segid != cB.segid
            or cA.authid != cB.authid
        ):
            print(f"chains {cA} and {cB} differ")
            return False
        for rA, rB in zip(cA.residues(), cB.residues()):
            if (
                rA.num_atoms() != rB.num_atoms()
                or rA.name != rA.name
                or rA.num != rB.num
                or rA.authid != rB.authid
                or rA.icode != rB.icode
            ):
                print(f"residues {rA} and {rB} differ")
                return False
            for aA, aB in zip(rA.atoms(), rB.atoms()):
                if (
                    aA.num_locations() != aB.num_locations()
                    or aA.name != aB.name
                    or aA.het != aB.het
                ):
                    print(f"atoms {aA} and {aB} differ")
                    return False
                for lA, lB in zip(aA.locations(), aB.locations()):
                    if (
                        (abs(lA.coors - lB.coors) > 0.01).any()
                        or lA.occ != lB.occ
                        or lA.B != lB.B
                        or lA.alt != lB.alt
                    ):
                        print(f"atoms {lA} and {lB} differ")
                        return False
    return True


def test_xcs_trajectory():
    # Load Trajectory
    protein = Protein(CIF_TRAJECTORY)

    # Save out Trajectory
    X_list, C, S = protein.to_XCS_trajectory()

    # Load back in via XCS
    protein_xcs_load = Protein(X_list, C, S)
    assert compare_proteins(protein, protein_xcs_load)

    # Print Trajectory
    print(protein)

    # Display Trajectory
    protein.display()


def test_trajectory_round_trip():
    # Load Trajectory
    protein = Protein(CIF_TRAJECTORY)

    # Save out Trajectory
    X_list, C, S = protein.to_XCS_trajectory()

    # Load back in via XCS
    protein_xcs_load = Protein.from_XCS_trajectory(X_list, C, S)
    assert compare_proteins(protein, protein_xcs_load)

    # Turn back into XCS
    X_list_1, C_1, S_1 = protein_xcs_load.to_XCS_trajectory()
    assert len(X_list) == len(X_list_1)
    assert [(x1 == x2).all() for x1, x2 in zip(X_list, X_list_1)]
    assert (C == C_1).all()
    assert (S == S_1).all()


@pytest.mark.parametrize("pdb_id", ["3bdi", "5sv5"])
def test_edge_cases(pdb_id):
    Protein(pdb_id, canonicalize=True)
