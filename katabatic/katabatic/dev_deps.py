#!/usr/bin/env python3
"""
Katabatic Development Dependency Manager

This script helps manage model-specific dependencies for development
while keeping the main package clean for PyPI distribution.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True,
                                capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def install_model_deps(model_name):
    """Install dependencies for a specific model."""
    model_path = Path(f"katabatic/models/{model_name}")

    if not model_path.exists():
        print(f"Model '{model_name}' not found in katabatic/models/")
        return False

    if not (model_path / "pyproject.toml").exists():
        print(f"No pyproject.toml found for model '{model_name}'")
        print(f"Using main package extras instead...")
        return run_command(f"poetry install -E {model_name}")

    print(f"Installing dependencies for {model_name} model...")

    # Install in the model directory
    if not run_command("poetry install", cwd=model_path):
        return False

    print(f"✓ Dependencies for {model_name} installed successfully!")
    return True


def install_all_models():
    """Install dependencies for all models."""
    models_dir = Path("katabatic/models")
    success = True

    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in ["__pycache__"]:
            if not install_model_deps(model_dir.name):
                success = False

    return success


def create_dev_environment(model_name=None):
    """Create a development environment for a specific model or all models."""
    # First install the main package in development mode
    print("Installing main katabatic package in development mode...")
    if not run_command("poetry install"):
        return False

    if model_name:
        return install_model_deps(model_name)
    else:
        print("Installing all model dependencies...")
        return install_all_models()


def list_models():
    """List available models."""
    models_dir = Path("katabatic/models")
    models = []

    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in ["__pycache__"]:
            models.append(model_dir.name)

    return models


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("Katabatic Development Dependency Manager")
        print("\nUsage:")
        print(
            "  python dev_deps.py install <model_name>  # Install deps for specific model")
        print("  python dev_deps.py install all          # Install deps for all models")
        print("  python dev_deps.py list                 # List available models")
        print("  python dev_deps.py setup <model_name>   # Setup dev environment for model")
        print("\nAvailable models:")
        for model in list_models():
            print(f"  - {model}")
        return

    command = sys.argv[1]

    if command == "list":
        models = list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")

    elif command == "install":
        if len(sys.argv) < 3:
            print("Please specify a model name or 'all'")
            return

        target = sys.argv[2]
        if target == "all":
            install_all_models()
        else:
            install_model_deps(target)

    elif command == "setup":
        if len(sys.argv) < 3:
            print("Please specify a model name")
            return

        model_name = sys.argv[2]
        create_dev_environment(model_name)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
