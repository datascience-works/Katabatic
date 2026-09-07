#!/usr/bin/env python3
"""Cross-platform setup utility for Katabatic."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys

SUPPORTED_OS = {"Windows", "Darwin", "Linux"}
REQUIRED_PYTHON = (3, 11)


def detect_architecture() -> str:
    """Detect the CPU architecture running Katabatic."""
    return platform.machine()


def detect_os() -> str:
    """Detect the operating system running Katabatic."""
    system = platform.system()

    if system not in SUPPORTED_OS:
        raise RuntimeError(
            f"Unsupported operating system: {system}. "
            f"Supported systems: {', '.join(sorted(SUPPORTED_OS))}"
        )

    return system


def is_windows(operating_system: str) -> bool:
    """Return True when running on Windows."""
    return operating_system == "Windows"


def check_python() -> None:
    """Ensure Python 3.11.x is being used."""
    version = sys.version_info

    if version[:2] != REQUIRED_PYTHON:
        raise RuntimeError(
            "Katabatic requires Python 3.11.x. "
            f"Detected Python {version.major}.{version.minor}.{version.micro}."
        )


def check_poetry() -> str:
    """Return the Poetry executable if it is installed."""
    poetry = shutil.which("poetry")

    if poetry is None:
        raise RuntimeError(
            "Poetry was not found on PATH. "
            "Please install Poetry before running the Katabatic setup."
        )

    return poetry


def check_prerequisites(operating_system: str) -> None:
    """Check platform-specific prerequisites."""
    if is_windows(operating_system):
        print("Checking Windows prerequisites...")

        if shutil.which("powershell") or shutil.which("pwsh"):
            print("PowerShell found.")
        else:
            raise RuntimeError(
                "PowerShell was not found. "
                "Please install PowerShell before running the Katabatic setup."
            )

    elif operating_system == "Darwin":
        print("Checking macOS prerequisites...")

    elif operating_system == "Linux":
        print("Checking Linux prerequisites...")


def install(
    poetry: str,
    operating_system: str,
    model: str | None = None,
) -> None:
    """Install Katabatic dependencies using the detected OS."""
    command = [poetry, "install"]

    if model:
        command.extend(["-E", model])

    print("Installing Katabatic dependencies...")
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up the Katabatic development environment."
    )
    parser.add_argument(
        "--model",
        help="Optional model extra to install, e.g. ganblr or great.",
    )
    args = parser.parse_args()

    try:
        operating_system = detect_os()
        architecture = detect_architecture()

        print(f"Detected operating system: {operating_system}")
        print(f"Detected architecture: {architecture}")
        print(
            f"Detected Python version: "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        check_python()
        check_prerequisites(operating_system)
        poetry = check_poetry()
        print(f"Poetry found: {poetry}")

        install(poetry, operating_system, args.model)

        print("Katabatic setup completed successfully.")
        return 0

    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
