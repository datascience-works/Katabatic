#!/usr/bin/env python3
"""Backward-compatible wrapper; prefer ``katabatic pin-notebook-kernel``."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    from katabatic.cli.commands.pin_notebook_kernel import run_pin_notebook_kernel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebooks", nargs="+", type=Path, help="Notebook paths (.ipynb) to patch")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--kernel-name", default="katabatic-venv")
    p.add_argument("--display-name", default="Python (Katabatic .venv)")
    args = p.parse_args()
    return run_pin_notebook_kernel(
        args.notebooks,
        repo_root=args.repo_root,
        kernel_name=args.kernel_name,
        display_name=args.display_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
