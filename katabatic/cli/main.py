"""Katabatic command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _cmd_init_model(args: argparse.Namespace) -> None:
    from katabatic.cli.commands.model_init import init_model

    init_model(args.model_name, args.dependencies or None)


def _cmd_register_dataset(args: argparse.Namespace) -> None:
    from katabatic.cli.commands.register_dataset import register_dataset_cli

    register_dataset_cli(
        args.dataset_name,
        args.csv_path,
        target_column=args.target_column,
        artifact_root=args.artifact_root,
        check_model=args.check_model,
    )


def _cmd_pin_notebook_kernel(args: argparse.Namespace) -> None:
    if not args.notebooks:
        print(
            "Usage: katabatic pin-notebook-kernel <notebook.ipynb> [more.ipynb ...]\n"
            "  Registers kernelspec in .venv and sets kernelspec metadata on each notebook."
        )
        sys.exit(1)
    script = Path(__file__).resolve().parents[2] / "scripts" / "pin_notebook_kernel.py"
    subprocess.check_call([sys.executable, str(script), *args.notebooks])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="katabatic",
        description="Katabatic: synthetic tabular data generation and evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-model", help="Scaffold a new model package")
    p_init.add_argument("model_name", help="Model name (e.g. mymodel)")
    p_init.add_argument(
        "dependencies",
        nargs="*",
        help="Optional dependency names for pyproject extras",
    )
    p_init.set_defaults(func=_cmd_init_model)

    p_reg = sub.add_parser("register-dataset", help="Register a dataset in the artifact registry")
    p_reg.add_argument("dataset_name", help="Registry name for the dataset")
    p_reg.add_argument("csv_path", help="Path to source CSV")
    p_reg.add_argument("--target-column", default=None, help="Target column name")
    p_reg.add_argument(
        "--artifact-root",
        default=None,
        help="Artifact store root (default: artifacts/)",
    )
    p_reg.add_argument(
        "--check-model",
        default=None,
        help="Verify compatibility with a model (e.g. ganblr)",
    )
    p_reg.set_defaults(func=_cmd_register_dataset)

    p_pin = sub.add_parser(
        "pin-notebook-kernel",
        help="Pin Jupyter kernelspec for notebooks to the project venv",
    )
    p_pin.add_argument("notebooks", nargs="*", help="Notebook paths")
    p_pin.set_defaults(func=_cmd_pin_notebook_kernel)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
