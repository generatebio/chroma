#!/usr/bin/env python3
"""Generate explicit Conda lockfiles for supported Chroma platforms."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


DEFAULT_PLATFORMS = ("osx-arm64", "linux-64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default="environment.yml",
        help="Path to the shared Conda environment spec.",
    )
    parser.add_argument(
        "--platform",
        dest="platforms",
        action="append",
        default=[],
        help="Platform to solve. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-template",
        default="environment.{platform}.lock",
        help="Output filename template. Must contain {platform}.",
    )
    parser.add_argument(
        "--linux-glibc",
        default="2.17",
        help="glibc virtual package target for linux-64 solves.",
    )
    return parser.parse_args()


def load_env_spec(path: Path) -> tuple[list[str], list[str]]:
    with path.open() as handle:
        env = yaml.safe_load(handle)

    channels = env.get("channels") or ["conda-forge"]
    deps = []
    for dep in env.get("dependencies", []):
        if isinstance(dep, str):
            deps.append(dep)
            continue
        raise ValueError(
            f"{path} contains non-Conda dependencies ({dep!r}); "
            "explicit lockfiles in this repo must be pure Conda specs."
        )
    return channels, deps


def solve_platform(
    channels: list[str],
    deps: list[str],
    platform: str,
    linux_glibc: str,
) -> dict:
    mamba = shutil.which("mamba")
    if not mamba:
        raise RuntimeError("mamba is required to generate lockfiles")

    with tempfile.TemporaryDirectory(prefix=f"chroma-lock-{platform}-") as tmpdir:
        prefix = Path(tmpdir) / "prefix"
        cmd = [
            mamba,
            "create",
            "--dry-run",
            "--json",
            "--override-channels",
            "--platform",
            platform,
            "--prefix",
            str(prefix),
        ]
        for channel in channels:
            cmd.extend(["-c", channel])
        cmd.extend(deps)

        env = os.environ.copy()
        if platform == "linux-64":
            env["CONDA_OVERRIDE_LINUX"] = "1"
            env["CONDA_OVERRIDE_GLIBC"] = linux_glibc

        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"failed to solve {platform}")
        return json.loads(proc.stdout)


def write_lockfile(
    data: dict,
    env_path: Path,
    output_path: Path,
    platform: str,
    linux_glibc: str,
) -> None:
    links = data["actions"]["LINK"]
    lines = [
        "# This file may be used to create an environment using:",
        "# $ conda create --name <env> --file <this file>",
        f"# platform: {platform}",
        f"# source: {env_path.name}",
    ]
    if platform == "linux-64":
        lines.append(
            f"# virtual-packages: __linux=1, __glibc={linux_glibc}"
        )
    lines.append("@EXPLICIT")
    lines.extend(pkg["url"] for pkg in links)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    env_path = Path(args.env_file)
    channels, deps = load_env_spec(env_path)
    platforms = args.platforms or list(DEFAULT_PLATFORMS)

    if "{platform}" not in args.output_template:
        raise ValueError("--output-template must contain {platform}")

    for platform in platforms:
        data = solve_platform(channels, deps, platform, args.linux_glibc)
        output_path = Path(args.output_template.format(platform=platform))
        write_lockfile(
            data,
            env_path=env_path,
            output_path=output_path,
            platform=platform,
            linux_glibc=args.linux_glibc,
        )
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
