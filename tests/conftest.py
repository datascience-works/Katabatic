"""Shared pytest fixtures."""

from __future__ import annotations

import os

# Limit native-library threads before importing pandas, NumPy,
# XGBoost, TensorFlow or other compiled machine-learning libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def tiny_binary_csv(tmp_path):
    """Create a small binary-classification CSV."""
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
