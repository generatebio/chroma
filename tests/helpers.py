from pathlib import Path

import chroma


TEST_RESOURCES = Path(chroma.__file__).parent.parent / "tests" / "resources"


def cif_path(pdb_id: str) -> str:
    return str(TEST_RESOURCES / f"{pdb_id.lower()}.cif")
