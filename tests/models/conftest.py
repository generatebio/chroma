import os
import subprocess
from pathlib import Path

import pytest

import chroma
from chroma.data import Protein


@pytest.fixture(scope="module")
def XCS():
    input_file = str(
        Path(Path(chroma.__file__).parent.parent, "tests", "resources", "1n8z.cif")
    )

    protein = Protein(input_file)
    length = 100
    return [_t[:, :length] for _t in protein.to_XCS(all_atom=True)]
    # return protein.to_XCS(all_atom=True)


@pytest.fixture(scope="module")
def XCS_backbone(XCS):
    X, C, S = XCS
    return X[:, :, :4, :], C, S
