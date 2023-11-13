import copy
import filecmp
import random
import tempfile
import time
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas
import pytest
import requests

import chroma
from chroma.data.system import System

BASE_PATH = str(Path(chroma.__file__).parent.parent)
CIF_TRAJECTORY = BASE_PATH + "/tests/resources/chroma_trajectory.cif"


@pytest.fixture
def cif_file():
    file = str(
        Path(Path(chroma.__file__).parent.parent, "tests", "resources", "7bz5.cif")
    )
    return file


def download_file(url, destination_file):
    r = requests.get(url, allow_redirects=True)
    open(destination_file, "wb").write(r.content)


def test_7bz5_selection(cif_file):
    """
    Using a selector on a System, expect canonical results
    """
    valid_strings = [
        ("chain B", 218, 222),
        # very small queries that are easy to visually verify
        ("chain B and resid 3 around 2.5", 4, 4),  # selects B near any resid 3
        (
            "(chain B and resid 3) around 2.5",
            3,
            3,
        ),  # use parentheses for specific search around B, resid 3. - 14 atoms in 3 residues
        (
            "byres ((chain B and resid 3) around 2.5)",
            3,
            3,
        ),  # byres selects all atoms in all affected residues - 24 atoms; 3 residues
        (
            "(chain B and resid 3 and name CA) around 2.5",
            3,
            3,
        ),  # further refine to radius around the B, resid 3 CA - 7 atoms in those 3 resis
        # re versus name enumeration (should yield same results)
        (
            "(chain B and resid 3-5 and (re C.?)) around 12.5",
            92,
            92,
        ),  # Atoms: 329, Residues: 92
        (
            "(chain B and resid 3-5 and (name C or name CA or name CB or name CD or"
            " name CG)) around 12.5",
            92,
            92,
        ),  # Atoms: 329, Residues: 92
        # some larger queries and tests
        (
            "(chain B and resid 3) or (chain C and resid 103) saround 13",
            28,
            28,
        ),  # right-association will only expand resid range on C
        (
            "((chain B and resid 3-9) or (chain C and resid 103)) saround 13",
            48,
            49,
        ),  # expand both ranges.
        (
            "(authchain H and authresid 1-14) around 12.0",
            238,
            238,
        ),  # pymol-like use of authchain and auth res ids
        (
            "not (authchain H and authresid 1-14) around 12.0",
            959,
            1000,
        ),  # right-association lets this one work ok.
        (
            "(not ((authchain H and authresid 1-14) around 12.0)) or ((authchain H and"
            " authresid 1-14) around 12.0)",
            1148,
            1189,
        ),  # Venn full coverage
        (
            "first (not ((authchain H and authresid 1-14) around 12.0)) or ((authchain"
            " H and authresid 1-14) around 12.0)",
            0,
            1,
        ),  # testing first
        (
            "last (not ((authchain H and authresid 1-14) around 12.0)) or ((authchain H"
            " and authresid 1-14) around 12.0)",
            1,
            1,
        ),  # testing last
        ("gti 0-18 or gti 190-234", 28, 64),  # GTI access
        (
            "gti 0-18 or gti 1140-2340",
            53,
            68,
        ),  # GTI access, including some out-of-range values
    ]

    # compare with MST in right-associativity mode
    sys = System.from_CIF(cif_file)
    for cmd in valid_strings:
        print(f"cmd = {cmd}")

        # check saved selections (MST happens to be right-associative)
        for uns in [False, True]:
            print(f"allow unstructued {uns}")
            sys.save_selection(cmd[0], left_associativity=False, allow_unstructured=uns)
            assert len(sys.get_selected()) == cmd[1 + uns]

    # smoke test in left-associativity mode and test writing/reading selections
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as file:
        file.close()
        for cmd in valid_strings:
            print(f"cmd = {cmd}")

            for uns in [False, True]:
                print(f"allow unstructued {uns}")
                selname = "_some_good_selection"
                sys.save_selection(
                    cmd[0],
                    selname=selname,
                    left_associativity=False,
                    allow_unstructured=uns,
                )
                assert sys.has_selection(selname)
                sys.to_CIF(file.name)
                newsys = System.from_CIF(file.name)
                assert newsys.has_selection(selname)
                assert newsys.get_selected(selname) == sys.get_selected(selname)

            results = sys.select(cmd[0], left_associativity=False)


def test_invalid_input(cif_file):
    parse_failure_strings = {  # strings that should fail but never segfault.
        "chain B and resid 3 around 2.5 B",  # extra stuff - need parentheses
        "chain B and resid 3 around",  # can't convert empty token into distance
        "byres (chain B and resid 3)) around 2.5",  # unmatched close paren
        "chain B chain A",  # invalid connector
    }
    sys = System.from_CIF(cif_file)
    for cmd in parse_failure_strings:
        print(cmd)
        try:
            result = sys.select(cmd)
            worked = True
        except Exception as e:
            worked = False
        assert worked == False, f"expression {cmd} was meant to fail but succeeded"


def next_structure_file(num=100, cif=True):
    tmp_file = "/tmp/_pdb_list.txt"
    download_file(
        "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt", tmp_file
    )
    D = pandas.read_csv(tmp_file, sep="\t", header=None)
    pdb_ids = list(D[0])
    random.shuffle(pdb_ids)

    if cif:
        file = "/tmp/_pdb_download.cif"
    else:
        file = "/tmp/_pdb_download.pdb"
    for pdb_id in pdb_ids[:num]:
        # download CIF file
        if cif:
            file = "/tmp/_pdb_download.cif"
            download_file(f"https://files.rcsb.org/download/{pdb_id}.cif", file)
        else:
            file = "/tmp/_pdb_download.pdb"
            download_file(f"https://files.rcsb.org/download/{pdb_id}.pdb", file)
        yield pdb_id, file


def test_writing_pdb(cif_file):
    # smoke test for PDB writing
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as file:
        file.close()
        for pdb_id, cif_file in next_structure_file(num=10, cif=True):
            sys = System.from_CIF(cif_file)
            sys.to_PDB(file.name)


def test_reading_cif(cif_file):
    # smoke test for CIF reading
    for pdb_id, cif_file in next_structure_file(num=100, cif=True):
        # load into python and MST in random order, record time
        order = random.random()
        for i in range(2):
            sys = System.from_CIF(cif_file)


def test_reading_pdb(cif_file):
    # smoke test for PDB reading
    for pdb_id, pdb_file in next_structure_file(num=100, cif=False):
        # load into python and MST in random order, record time
        order = random.random()
        for i in range(2):
            sys = System.from_PDB(pdb_file)


def test_update_with_xcs(cif_file):
    sys = System.from_CIF(CIF_TRAJECTORY)
    sys_copy = copy.deepcopy(sys)

    sys.swap_model(1)
    X, C, S = sys.to_XCS()
    sys.swap_model(1)
    sys.update_with_XCS(X, C, S)

    sys_copy.swap_model(1)
    assert sys_copy.num_atom_locations() == sys.num_atom_locations()
    for loc1, loc2 in zip(sys.locations(), sys_copy.locations()):
        assert (abs(loc1.coors == loc2.coors) < 0.01).all()
