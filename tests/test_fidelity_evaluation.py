"""Unit tests for FidelityEvaluation (katabatic/evaluate/fidelity/evaluation.py)."""

import pandas as pd
import pytest

from katabatic.evaluate.fidelity.evaluation import FidelityEvaluation


@pytest.fixture
def numeric_data():
    real = pd.DataFrame({"a": range(30), "b": range(30, 60)})
    synth = pd.DataFrame({"a": range(2, 32), "b": range(28, 58)})
    return real, synth


@pytest.fixture
def categorical_data():
    real = pd.DataFrame({"cat": ["x", "y", "z"] * 10})
    synth = pd.DataFrame({"cat": ["x", "y", "z"] * 10})
    return real, synth


@pytest.fixture
def mixed_data():
    real = pd.DataFrame({"cat": ["x", "y"] * 15, "num": range(30)})
    synth = pd.DataFrame({"cat": ["x", "y"] * 15, "num": range(1, 31)})
    return real, synth


def test_numeric_only_skips_jsd(numeric_data):
    real, synth = numeric_data
    ev = FidelityEvaluation(
        real, synth, categorical_cols=[], continuous_cols=["a", "b"]
    )
    results = ev.evaluate()

    assert results["categorical_jsd"] == {}
    assert results["categorical_score"] is None
    assert results["continuous_score"] is not None
    # Two continuous columns -> correlation diff is computable.
    assert results["correlation_score"] is not None


def test_categorical_only_skips_wasserstein(categorical_data):
    real, synth = categorical_data
    ev = FidelityEvaluation(real, synth, categorical_cols=["cat"], continuous_cols=[])
    results = ev.evaluate()

    assert results["continuous_wasserstein"] == {}
    assert results["continuous_score"] is None
    # Identical distributions -> JSD of 0, perfect categorical score.
    assert results["categorical_score"] == 1.0
    assert results["summary"]["mean_jsd"] == 0.0


def test_mixed_features_all_components_present(mixed_data):
    real, synth = mixed_data
    ev = FidelityEvaluation(
        real, synth, categorical_cols=["cat"], continuous_cols=["num"]
    )
    results = ev.evaluate()

    assert results["categorical_score"] is not None
    assert results["continuous_score"] is not None
    # Only one continuous column -> no correlation matrix to compare.
    assert results["correlation_score"] is None
    assert 0.0 <= results["fidelity_score"] <= 1.0


def test_dcr_identical_data_is_near_zero(mixed_data):
    real, _ = mixed_data
    ev = FidelityEvaluation(
        real, real.copy(), categorical_cols=["cat"], continuous_cols=["num"]
    )
    dcr = ev._compute_dcr()

    assert dcr is not None
    assert dcr == pytest.approx(0.0, abs=1e-9)


def test_dcr_none_with_no_usable_columns():
    real = pd.DataFrame({"only": ["a", "b", "c"]})
    synth = pd.DataFrame({"only": ["a", "b", "c"]})
    ev = FidelityEvaluation(real, synth, categorical_cols=[], continuous_cols=[])

    assert ev._compute_dcr() is None


def test_dcr_none_on_empty_frame():
    real = pd.DataFrame({"num": []})
    synth = pd.DataFrame({"num": [1.0, 2.0]})
    ev = FidelityEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])

    assert ev._compute_dcr() is None


def test_evaluate_includes_dcr_and_summary(mixed_data):
    real, synth = mixed_data
    ev = FidelityEvaluation(
        real, synth, categorical_cols=["cat"], continuous_cols=["num"]
    )
    results = ev.evaluate()

    assert "dcr_mean" in results
    assert results["dcr_mean"] is not None
    assert results["summary"] == {
        "fidelity_score": results["fidelity_score"],
        "mean_jsd": results["categorical_jsd"]["avg"],
        "dcr_mean": results["dcr_mean"],
    }


def test_evaluate_without_artifact_kwargs_does_not_write_files(mixed_data, tmp_path):
    real, synth = mixed_data
    # No _artifact_store/_evaluation_ref passed -> plain DataFrame-mode usage
    # (SyntheticEvaluationPipeline's call path); must not attempt any file I/O.
    ev = FidelityEvaluation(
        real, synth, categorical_cols=["cat"], continuous_cols=["num"]
    )
    results = ev.evaluate()

    assert list(tmp_path.iterdir()) == []
    assert results["fidelity_score"] is not None
