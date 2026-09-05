"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import importlib
import importlib.util

import pandas as pd
import pytest


def require_backend(module: str, probe: str) -> object:
    """Skip the calling module unless ``module`` is a real, usable install.

    ``pytest.importorskip`` is not sufficient on its own. An interrupted
    uninstall leaves an empty directory behind, which Python imports happily as
    an implicit namespace package: the import succeeds, the skip does not fire,
    and the test later dies with a confusing ``AttributeError`` from deep inside
    the model (e.g. "module 'torch' has no attribute 'save'").

    Call this at module scope BEFORE importing any model package, since those
    import their backend eagerly.
    """
    spec = None
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        pytest.skip(
            f"{module} is not installed (or is an empty namespace shell)",
            allow_module_level=True,
        )

    mod = importlib.import_module(module)
    if not hasattr(mod, probe):
        pytest.skip(
            f"{module} is present but unusable (no attribute {probe!r})",
            allow_module_level=True,
        )
    return mod


@pytest.fixture
def tiny_binary_csv(tmp_path):
    """30-row binary classification CSV (last column is target)."""
    df = pd.DataFrame(
        {
            "f0": list(range(30)),
            "f1": [i % 3 for i in range(30)],
            "y": [0, 1] * 15,
        }
    )
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path
