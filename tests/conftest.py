"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


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
