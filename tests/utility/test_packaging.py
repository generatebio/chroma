import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(command, cwd):
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def test_build_and_import_wheel(tmp_path):
    pytest.importorskip("build")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(dist_dir),
        ],
        cwd=PROJECT_ROOT,
    )

    wheel = next(dist_dir.glob("chroma-*.whl"))
    sdist = next(dist_dir.glob("chroma-*.tar.gz"))
    assert wheel.is_file()
    assert sdist.is_file()

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    python = venv_dir / bin_dir / "python"

    _run(
        [str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)],
        cwd=tmp_path,
    )
    result = _run(
        [
            str(python),
            "-c",
            (
                "from importlib import resources; "
                "import chroma, chroma.data, chroma.layers, chroma.models, chroma.utility; "
                "data = resources.files('chroma').joinpath('assets', 'centering', 'centering_2g3n.params'); "
                "assert data.is_file(); "
                "assert 'site-packages' in chroma.__file__; "
                "print(chroma.__file__)"
            ),
        ],
        cwd=tmp_path,
    )
    assert str(venv_dir) in result.stdout
