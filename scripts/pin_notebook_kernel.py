#!/usr/bin/env python3
"""
Register a Jupyter kernelspec inside the repo .venv and patch notebook(s) to use it.

VS Code / Cursor do not provide a CLI to "select kernel" for an .ipynb; this script
writes kernelspec metadata Jupyter tools respect, and helps the editor once the
same kernel name is discoverable from the project interpreter.

Usage (from repo root):
  python scripts/pin_notebook_kernel.py examples/ganblr.ipynb
  python scripts/pin_notebook_kernel.py examples/*.ipynb
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_KERNEL = "katabatic-venv"
DEFAULT_DISPLAY = "Python (Katabatic .venv)"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def install_kernelspec(*, venv_prefix: Path, python_exe: Path, name: str, display: str) -> None:
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(venv_prefix),
            "--name",
            name,
            "--display-name",
            display,
        ],
        check=True,
    )


def patch_notebook(path: Path, *, name: str, display: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    meta = nb.setdefault("metadata", {})
    meta["kernelspec"] = {
        "display_name": display,
        "language": "python",
        "name": name,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "notebooks",
        nargs="+",
        type=Path,
        help="Notebook paths (.ipynb) to patch",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Katabatic repo root (default: parent of scripts/)",
    )
    p.add_argument("--kernel-name", default=DEFAULT_KERNEL)
    p.add_argument("--display-name", default=DEFAULT_DISPLAY)
    args = p.parse_args()

    root = (args.repo_root or repo_root_from_script()).resolve()
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        print(f"error: expected venv python at {py}", file=sys.stderr)
        print("  run: poetry install", file=sys.stderr)
        return 1

    dest = root / ".venv" / "share" / "jupyter" / "kernels" / args.kernel_name
    print(f"Installing kernelspec {args.kernel_name!r} into {root / '.venv'} ...")
    install_kernelspec(
        venv_prefix=root / ".venv",
        python_exe=py,
        name=args.kernel_name,
        display=args.display_name,
    )
    print(f"  -> {dest}")

    for nb in args.notebooks:
        nb = nb.resolve()
        if nb.suffix != ".ipynb":
            print(f"skip (not .ipynb): {nb}", file=sys.stderr)
            continue
        patch_notebook(nb, name=args.kernel_name, display=args.display_name)
        print(f"Patched kernelspec in {nb}")

    print("\nIn Cursor: reload window, then Select Kernel → pick", repr(args.display_name))
    print("If the picker still spins, use Command Palette → Python: Select Interpreter → .venv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
