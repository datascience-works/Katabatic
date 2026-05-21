"""Deprecated entry point; use ``katabatic`` CLI instead."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "python main.py is deprecated; use the `katabatic` command (e.g. katabatic init-model).",
    DeprecationWarning,
    stacklevel=1,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: katabatic <command> [args...]")
        print("  Commands: init-model, register-dataset, pin-notebook-kernel")
        sys.exit(1)

    from katabatic.cli.main import main as cli_main

    # Map legacy subcommand style: python main.py init-model foo -> katabatic init-model foo
    cli_main([sys.argv[1], *sys.argv[2:]])


if __name__ == "__main__":
    main()
 