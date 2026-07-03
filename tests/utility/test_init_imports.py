import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_do_not_eagerly_import_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import chroma, chroma.data, chroma.layers, chroma.models, chroma.utility; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
