from __future__ import annotations

import pandas as pd
import pytest

from katabatic.evaluate.tstr.evaluation import TSTREvaluation


def _write_split(tmp_path, x_synth, y_synth, x_test, y_test):
    synthetic_dir = tmp_path / "synthetic"
    real_test_dir = tmp_path / "real_test"
    synthetic_dir.mkdir()
    real_test_dir.mkdir()

    x_synth.to_csv(synthetic_dir / "x_synth.csv", index=False)
    pd.DataFrame({"y": y_synth}).to_csv(synthetic_dir / "y_synth.csv", index=False)
    x_test.to_csv(real_test_dir / "x_test.csv", index=False)
    pd.DataFrame({"y": y_test}).to_csv(real_test_dir / "y_test.csv", index=False)

    return str(synthetic_dir), str(real_test_dir)


@pytest.mark.parametrize("n", [40])
def test_tstr_evaluation_numeric_features(tmp_path, n):
    """Baseline test for numeric features."""
    x = pd.DataFrame({"f0": range(n), "f1": [i % 5 for i in range(n)]})
    y = [i % 2 for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    assert "LR" in results
    assert "Accuracy" in results["LR"]


def test_tstr_evaluation_categorical_features(tmp_path):
    """Regression test on categorical features."""
    categories_a = ["low", "med", "high"]
    categories_b = ["small", "big"]
    n = 40
    x = pd.DataFrame(
        {
            "size": [categories_a[i % 3] for i in range(n)],
            "price": [categories_b[i % 2] for i in range(n)],
        }
    )
    y = [i % 2 for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    assert "LR" in results
    assert "Accuracy" in results["LR"]


def test_tstr_evaluation_mixed_features(tmp_path):
    """Mixed numeric + categorical columns (e.g. the shipped adult.csv)."""
    n = 40
    x = pd.DataFrame(
        {
            "age": range(n),
            "workclass": [["Private", "Self-emp"][i % 2] for i in range(n)],
        }
    )
    y = [i % 2 for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    assert "LR" in results
    assert "Accuracy" in results["LR"]
