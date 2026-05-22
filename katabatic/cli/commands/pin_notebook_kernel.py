"""Pin Jupyter kernelspec for notebooks to the project virtualenv."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEFAULT_KERNEL = "katabatic-venv"
DEFAULT_DISPLAY = "Python (Katabatic .venv)"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default cwd) to find the Katabatic project root."""
    for candidate in [start or Path.cwd(), *(start or Path.cwd()).parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "katabatic").is_dir():
            return candidate
    return start or Path.cwd()


def install_kernelspec(
    *,
    venv_prefix: Path,
    python_exe: Path,
    name: str,
    display: str,
) -> Path:
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
    return venv_prefix / "share" / "jupyter" / "kernels" / name


def patch_notebook(path: Path, *, name: str, display: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    meta = nb.setdefault("metadata", {})
    meta["kernelspec"] = {
        "display_name": display,
        "language": "python",
        "name": name,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def run_pin_notebook_kernel(
    notebooks: list[str | Path],
    *,
    repo_root: Path | None = None,
    kernel_name: str = DEFAULT_KERNEL,
    display_name: str = DEFAULT_DISPLAY,
) -> int:
    root = (repo_root or find_project_root()).resolve()
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        print(f"error: expected venv python at {py}", file=sys.stderr)
        print("  run: poetry install", file=sys.stderr)
        return 1

    print(f"Installing kernelspec {kernel_name!r} into {root / '.venv'} ...")
    dest = install_kernelspec(
        venv_prefix=root / ".venv",
        python_exe=py,
        name=kernel_name,
        display=display_name,
    )
    print(f"  -> {dest}")

    patched = 0
    for nb_path in notebooks:
        nb = Path(nb_path).resolve()
        if nb.suffix != ".ipynb":
            print(f"skip (not .ipynb): {nb}", file=sys.stderr)
            continue
        patch_notebook(nb, name=kernel_name, display=display_name)
        print(f"Patched kernelspec in {nb}")
        patched += 1

    if patched == 0:
        print("error: no notebooks were patched", file=sys.stderr)
        return 1

    print("\nIn Cursor: reload window, then Select Kernel → pick", repr(display_name))
    print("If the picker still spins, use Command Palette → Python: Select Interpreter → .venv")
    return 0
