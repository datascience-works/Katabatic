"""Unit tests for katabatic/pipeline/evaluation_pipeline.py (SyntheticEvaluationPipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from katabatic.evaluate.fidelity.evaluation import FidelityEvaluation
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline


def _frame(n=20, seed=0):
    rng = np.random.default_rng(seed)
    y = [i % 2 for i in range(n)]
    return pd.DataFrame(
        {
            "num": rng.normal(size=n) + np.array(y) * 2,
            "cat": [["x", "y"][i % 2] for i in range(n)],
            "label": y,
        }
    )


class _StubModel:
    def __init__(self, data):
        self._data = data

    def sample(self, n_samples, seed=0):
        return self._data.copy().iloc[:n_samples].reset_index(drop=True)


def test_default_dimensions_exclude_stability():
    pipeline = SyntheticEvaluationPipeline()
    assert "stability" not in pipeline.dimensions
    assert set(pipeline.dimensions) == {
        "fidelity",
        "utility",
        "diversity",
        "privacy",
        "consistency",
    }


def test_unknown_dimension_raises_at_construction():
    with pytest.raises(ValueError, match="Unknown dimensions"):
        SyntheticEvaluationPipeline(dimensions=["not_a_real_dimension"])


def test_warns_when_column_types_not_provided(capsys):
    SyntheticEvaluationPipeline(dimensions=["fidelity"])
    captured = capsys.readouterr()
    assert "auto-detected" in captured.out.lower()


def test_no_warning_when_column_types_provided(capsys):
    SyntheticEvaluationPipeline(
        dimensions=["fidelity"], categorical_cols=["cat"], continuous_cols=["num"]
    )
    captured = capsys.readouterr()
    assert "auto-detected" not in captured.out.lower()


def test_run_full_non_stability_pipeline_and_saves_report(tmp_path):
    real = _frame(seed=1)
    synth = _frame(seed=2)
    pipeline = SyntheticEvaluationPipeline(
        categorical_cols=["cat"],
        continuous_cols=["num"],
        n_folds=2,
    )
    report = pipeline.run(
        real,
        synth,
        target_col="label",
        output_dir=str(tmp_path),
        report_prefix="t_",
    )

    for dim in ["fidelity", "utility", "diversity", "privacy", "consistency"]:
        assert dim in report.dimension_results
        assert "error" not in report.dimension_results[dim]
    assert 0.0 <= report.composite_score <= 1.0
    assert (tmp_path / "t_evaluation_report.json").exists()
    assert (tmp_path / "t_evaluation_summary.csv").exists()


def test_run_infers_target_col_from_last_column_with_warning(capsys):
    real = _frame(seed=3)
    synth = _frame(seed=4)
    pipeline = SyntheticEvaluationPipeline(
        dimensions=["fidelity"], categorical_cols=["cat"], continuous_cols=["num"]
    )
    pipeline.run(real, synth)

    captured = capsys.readouterr()
    assert "target_col not provided" in captured.out
    assert "'label'" in captured.out


def test_run_with_stability_requires_model_and_dispatches_correctly():
    real = _frame(seed=5)
    synth = _frame(seed=6)
    model = _StubModel(synth)
    pipeline = SyntheticEvaluationPipeline(
        dimensions=["stability"],
        categorical_cols=["cat"],
        continuous_cols=["num"],
        n_stability_runs=2,
        stability_seeds=[0, 1],
    )
    report = pipeline.run(real, synth, target_col="label", model=model)

    assert "stability" in report.dimension_results
    assert "error" not in report.dimension_results["stability"]


def test_validate_inputs_rejects_non_dataframe_real_data():
    pipeline = SyntheticEvaluationPipeline(dimensions=["fidelity"])
    with pytest.raises(TypeError, match="real_data must be a pandas DataFrame"):
        pipeline._validate_inputs([1, 2, 3], pd.DataFrame({"a": [1]}), "a")


def test_validate_inputs_rejects_non_dataframe_synthetic_data():
    pipeline = SyntheticEvaluationPipeline(dimensions=["fidelity"])
    with pytest.raises(TypeError, match="synthetic_data must be a pandas DataFrame"):
        pipeline._validate_inputs(pd.DataFrame({"a": [1]}), [1, 2, 3], "a")


def test_validate_inputs_rejects_empty_real_data():
    pipeline = SyntheticEvaluationPipeline(dimensions=["fidelity"])
    with pytest.raises(ValueError, match="real_data is empty"):
        pipeline._validate_inputs(
            pd.DataFrame({"a": []}), pd.DataFrame({"a": [1]}), "a"
        )


def test_validate_inputs_rejects_empty_synthetic_data():
    pipeline = SyntheticEvaluationPipeline(dimensions=["fidelity"])
    with pytest.raises(ValueError, match="synthetic_data is empty"):
        pipeline._validate_inputs(
            pd.DataFrame({"a": [1]}), pd.DataFrame({"a": []}), "a"
        )


def test_validate_inputs_requires_target_col_for_utility_and_consistency():
    pipeline = SyntheticEvaluationPipeline(dimensions=["utility"])
    with pytest.raises(ValueError, match="target_col.*is required"):
        pipeline._validate_inputs(
            pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), ""
        )


def test_validate_inputs_requires_model_for_stability():
    pipeline = SyntheticEvaluationPipeline(dimensions=["stability"])
    with pytest.raises(ValueError, match="'model' must be provided"):
        pipeline._validate_inputs(
            pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), "a", model=None
        )


def test_dimension_exception_is_caught_and_degrades_gracefully(monkeypatch):
    real = _frame(seed=7)
    synth = _frame(seed=8)
    pipeline = SyntheticEvaluationPipeline(
        dimensions=["fidelity"], categorical_cols=["cat"], continuous_cols=["num"]
    )

    def _boom(self):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(FidelityEvaluation, "evaluate", _boom)
    report = pipeline.run(real, synth, target_col="label")

    assert report.dimension_results["fidelity"]["error"] == "synthetic failure"
    assert report.dimension_results["fidelity"]["fidelity_score"] == 0.0
    assert report.composite_score == 0.0


def test_build_evaluator_utility_requires_target_col_directly():
    pipeline = SyntheticEvaluationPipeline(dimensions=["utility"])
    real = _frame()
    with pytest.raises(ValueError, match="required for the utility dimension"):
        pipeline._build_evaluator("utility", real, real.copy(), None, None)


def test_build_evaluator_consistency_requires_target_col_directly():
    pipeline = SyntheticEvaluationPipeline(dimensions=["consistency"])
    real = _frame()
    with pytest.raises(ValueError, match="required for the consistency dimension"):
        pipeline._build_evaluator("consistency", real, real.copy(), None, None)


def test_build_evaluator_unknown_dimension_raises_directly():
    pipeline = SyntheticEvaluationPipeline(dimensions=["fidelity"])
    real = _frame()
    with pytest.raises(ValueError, match="No evaluator registered"):
        pipeline._build_evaluator("not_real", real, real.copy(), "label", None)


def test_build_evaluator_privacy_dispatch():
    pipeline = SyntheticEvaluationPipeline(
        dimensions=["privacy"], categorical_cols=["cat"], continuous_cols=["num"]
    )
    real = _frame()
    ev = pipeline._build_evaluator("privacy", real, real.copy(), "label", None)
    assert ev.near_dup_threshold == pipeline.near_dup_threshold


def test_target_col_not_in_real_data_uses_full_frames():
    # target_col explicitly set to a column absent from real_data -> the
    # feature/target split is skipped and full frames are passed through.
    real = _frame(seed=9)
    synth = _frame(seed=10)
    pipeline = SyntheticEvaluationPipeline(
        dimensions=["fidelity"],
        categorical_cols=["cat"],
        continuous_cols=["num", "label"],
    )
    report = pipeline.run(real, synth, target_col="does_not_exist")
    assert "error" not in report.dimension_results["fidelity"]
