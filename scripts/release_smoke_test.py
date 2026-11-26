#!/usr/bin/env python3
"""Release smoke test for Katabatic (release gate, section 3: TestPyPI dry run).

Installs a published or locally built ``katabatic`` into a brand-new virtual
environment, then checks that a user could actually use it:

1. the package installs and imports, at the expected version;
2. its published dependencies contain no Git or URL references
   (PyPI rejects these; see the MST / private-pgm note in pyproject.toml);
3. the bundled datasets ship with the package;
4. a lightweight model (Histogram, core install only) trains on the bundled
   ``car`` dataset, samples rows and runs ``evaluate()``.

Run it with Python 3.11, from anywhere (it never imports the local source tree):

    # Release candidate on TestPyPI (after publish-testpypi.yml has run)
    python scripts/release_smoke_test.py --source testpypi --version 1.2.0rc1

    # Final release on PyPI
    python scripts/release_smoke_test.py --source pypi --version 1.2.0

    # Wheel you built yourself with `poetry build` (no upload needed)
    python scripts/release_smoke_test.py --wheel dist/katabatic-1.2.0-py3-none-any.whl

    # Also check that an extra installs, e.g. ctgan (repeatable)
    python scripts/release_smoke_test.py --source testpypi --version 1.2.0rc1 --extra ctgan

Exit code 0 means every check passed; 1 means at least one failed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REQUIRED_PYTHON = (3, 11)
PACKAGE = "katabatic"
TESTPYPI_INDEX = "https://test.pypi.org/simple/"
PYPI_INDEX = "https://pypi.org/simple/"
SMOKE_DATASET = "car"
QUICK_DIMENSIONS = ("fidelity", "utility")

# Runs inside the fresh virtual environment. Prints one JSON line prefixed with
# RESULT_MARKER so the parent can read the outcome among the library's own output.
RESULT_MARKER = "KATABATIC_SMOKE_RESULT="
SMOKE_CODE = r"""
import json, os, sys, tempfile, time
from importlib import metadata
from importlib.resources import files

checks = []
def record(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

expected_version, dataset, dimensions = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import katabatic
    record("import katabatic", True, katabatic.__file__)
    installed = katabatic.__version__
    if expected_version:
        record("version matches", installed == expected_version,
               f"installed {installed}, expected {expected_version}")
    else:
        record("version reported", installed != "0.0.0.dev0", installed)

    requires = metadata.requires("katabatic") or []
    direct = [r for r in requires if " @ " in r or "git+" in r]
    record("no Git/URL dependencies", not direct, ", ".join(direct) or f"{len(requires)} requirements checked")

    import pandas as pd
    csv = files("katabatic.datasets") / f"{dataset}.csv"
    df = pd.read_csv(csv)
    record("bundled dataset loads", len(df) > 0, f"{dataset}.csv: {df.shape[0]} rows x {df.shape[1]} cols")

    from katabatic.models.histogram import HistogramModel
    work = tempfile.mkdtemp(prefix="katabatic-smoke-")
    df.to_csv(os.path.join(work, "train_full.csv"), index=False)
    start = time.time()
    model = HistogramModel(random_state=42).train(work, synthetic_dir=os.path.join(work, "synthetic"))
    synth = model.sample(200, seed=42)
    record("train + sample (histogram)", len(synth) == 200 and list(synth.columns) == list(df.columns),
           f"{len(synth)} rows in {time.time() - start:.1f}s")

    dims = [d for d in dimensions.split(",") if d] or None
    report = model.evaluate(df, target_col=df.columns[-1], dimensions=dims)
    scores = getattr(report, "dimension_scores", {}) or {}
    composite = getattr(report, "composite_score", None)
    record("evaluate()", composite is not None and bool(scores),
           f"composite {composite}, " + ", ".join(f"{k} {v}" for k, v in scores.items()))
except Exception as exc:  # report the failure instead of a bare traceback
    record("smoke test crashed", False, f"{type(exc).__name__}: {exc}")

print("KATABATIC_SMOKE_RESULT=" + json.dumps(checks))
"""


def check_python() -> None:
    """The package only supports Python 3.11, so the test venv must use it."""
    if sys.version_info[:2] != REQUIRED_PYTHON:
        found = ".".join(map(str, sys.version_info[:3]))
        sys.exit(
            f"Python 3.11 is required (found {found}). Run this script with a "
            "Python 3.11 interpreter, e.g. `python3.11 scripts/release_smoke_test.py ...`."
        )


def requirement(version: str | None, extras: list[str]) -> str:
    """Build the pip requirement, e.g. ``katabatic[ctgan]==1.2.0rc1``."""
    name = f"{PACKAGE}[{','.join(extras)}]" if extras else PACKAGE
    return f"{name}=={version}" if version else name


def install_args(
    source: str | None, version: str | None, wheel: str | None, extras: list[str]
) -> list[str]:
    """Arguments for ``pip install`` for the chosen source."""
    if wheel:
        target = str(Path(wheel).resolve())
        return [f"{target}[{','.join(extras)}]" if extras else target]
    args = ["--no-cache-dir"]
    if source == "testpypi":
        # Katabatic comes from TestPyPI; its dependencies (pandas, numpy...) from PyPI.
        args += ["--index-url", TESTPYPI_INDEX, "--extra-index-url", PYPI_INDEX]
    return args + [requirement(version, extras)]


def venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside a virtual environment, on any OS."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(
    cmd: list[str], cwd: Path, label: str | None = None
) -> subprocess.CompletedProcess:
    print(f"$ {label or ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def print_summary(checks: list[dict]) -> bool:
    width = max(len(c["name"]) for c in checks)
    print("\nRelease smoke test")
    print("-" * (width + 40))
    for c in checks:
        status = "PASS" if c["ok"] else "FAIL"
        print(f"{status}  {c['name']:<{width}}  {c['detail']}")
    print("-" * (width + 40))
    passed = all(c["ok"] for c in checks)
    print("RESULT: " + ("all checks passed" if passed else "FAILED"))
    return passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install katabatic into a clean venv and smoke-test it."
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--source", choices=["testpypi", "pypi"], help="Install from an index."
    )
    where.add_argument("--wheel", help="Install a locally built wheel file.")
    parser.add_argument(
        "--version",
        help="Version to install and expect, e.g. 1.2.0rc1 (required with --source).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Also install an extra, e.g. ctgan. Repeatable.",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Run all six evaluation dimensions instead of fidelity + utility.",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the temporary virtual environment for inspection.",
    )
    args = parser.parse_args(argv)
    if args.source and not args.version:
        parser.error("--version is required with --source (e.g. --version 1.2.0rc1)")
    if args.wheel and not Path(args.wheel).is_file():
        parser.error(f"wheel not found: {args.wheel}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    check_python()

    workdir = Path(tempfile.mkdtemp(prefix="katabatic-release-"))
    venv_dir = workdir / "venv"
    checks: list[dict] = []
    started = time.time()
    try:
        print(f"Creating a clean virtual environment in {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = venv_python(venv_dir)

        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
        result = run(
            pip + install_args(args.source, args.version, args.wheel, args.extra),
            cwd=workdir,
        )
        label = args.wheel or requirement(args.version, args.extra)
        checks.append(
            {
                "name": "pip install",
                "ok": result.returncode == 0,
                "detail": label
                if result.returncode == 0
                else result.stderr.strip()[-600:],
            }
        )

        if result.returncode == 0:
            dimensions = "" if args.full_eval else ",".join(QUICK_DIMENSIONS)
            # Run from workdir so the local source tree can never shadow the installed package.
            smoke = run(
                [
                    str(python),
                    "-c",
                    SMOKE_CODE,
                    args.version or "",
                    SMOKE_DATASET,
                    dimensions,
                ],
                cwd=workdir,
                label=f"python -c <smoke test: {SMOKE_DATASET} dataset, histogram model>",
            )
            lines = [
                line
                for line in smoke.stdout.splitlines()
                if line.startswith(RESULT_MARKER)
            ]
            if lines:
                checks += json.loads(lines[-1][len(RESULT_MARKER) :])
            else:
                checks.append(
                    {
                        "name": "smoke test ran",
                        "ok": False,
                        "detail": (smoke.stderr or smoke.stdout).strip()[-600:],
                    }
                )
    finally:
        if args.keep_venv:
            print(f"Kept the virtual environment at {venv_dir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    passed = print_summary(checks)
    print(f"Finished in {time.time() - started:.0f}s")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
