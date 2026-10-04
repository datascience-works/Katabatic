"""Tests for the shared Model.evaluate() (six-dimension scoring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from katabatic.models.base_model import EVALUATION_DIMENSIONS, Model


class _ResampleModel(Model):
    """Minimal model: 'generates' by resampling its training rows."""

    def __init__(self, as_numpy: bool = False):
        super().__init__()
        self.as_numpy = as_numpy
        self.df: pd.DataFrame | None = None
        self.sample_calls: list[tuple[int | None, int | None]] = []

    def train(self, data_dir, *args, synthetic_dir=None, artifact_state_dir=None, **kw):
        self.df = pd.read_csv(f"{data_dir}/train_full.csv")
        self.is_fitted = True
        return self

    def sample(self, n_samples=None, *args, seed=None, **kwargs):
        self.sample_calls.append((n_samples, seed))
        out = self.df.sample(n=n_samples, replace=True, random_state=seed or 0)
        out = out.reset_index(drop=True)
        return out.to_numpy() if self.as_numpy else out


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=n) + y,
            "cat": rng.choice(["a", "b", "c"], n),
            "y": y,
        }
    )
    df.to_csv(tmp_path / "train_full.csv", index=False)
    return tmp_path, df


def _fitted(data, **kw):
    path, _ = data
    return _ResampleModel(**kw).train(str(path))


def test_evaluate_returns_all_six_dimensions(data):
    _, df = data
    model = _fitted(data)

    report = model.evaluate(
        df, target_col="y", categorical_cols=["cat"], continuous_cols=["num"]
    )

    assert set(report.dimension_scores) == set(EVALUATION_DIMENSIONS)
    assert all(0.0 <= v <= 1.0 for v in report.dimension_scores.values())
    assert 0.0 <= report.composite_score <= 1.0
    # One sample for scoring, then seeded re-samples for stability.
    assert model.sample_calls[0] == (len(df), None)
    assert any(seed is not None for _, seed in model.sample_calls[1:])


def test_evaluate_before_training_raises(data):
    _, df = data
    with pytest.raises(RuntimeError, match="Call train"):
        _ResampleModel().evaluate(df)


def test_evaluate_rejects_synthetic_data_missing_a_column(data):
    _, df = data
    model = _fitted(data)
    with pytest.raises(ValueError, match="missing real columns: \\['cat'\\]"):
        model.evaluate(df, synthetic_data=df.drop(columns=["cat"]))


def test_evaluate_uses_given_synthetic_data_and_dimension_subset(data):
    _, df = data
    model = _fitted(data)
    # Reordered columns plus an extra one are aligned to the real data.
    synth = df[["y", "cat", "num"]].assign(extra=1)

    report = model.evaluate(
        df, target_col="y", synthetic_data=synth, dimensions=["fidelity", "privacy"]
    )

    assert set(report.dimension_scores) == {"fidelity", "privacy"}
    assert model.sample_calls == []


def test_evaluate_accepts_numpy_samples_and_saves_report(data, tmp_path):
    _, df = data
    model = _fitted(data, as_numpy=True)
    out_dir = tmp_path / "report"

    report = model.evaluate(
        df, target_col="y", dimensions=["fidelity"], output_dir=str(out_dir)
    )

    assert set(report.dimension_scores) == {"fidelity"}
    assert any(out_dir.glob("*.json"))


class _RecordingPipeline:
    """A custom evaluation pipeline: records what it was given."""

    def __init__(self):
        self.kwargs = None

    def run(self, real_data, synthetic_data, **kwargs):
        self.kwargs = {
            "real_data": real_data,
            "synthetic_data": synthetic_data,
            **kwargs,
        }
        return {"my_metric": 0.5}


def test_evaluate_with_custom_pipeline(data):
    _, df = data
    model = _fitted(data)
    pipeline = _RecordingPipeline()

    result = model.evaluate(df, pipeline=pipeline, target_col="y", test_data=df)

    assert result == {"my_metric": 0.5}
    got = pipeline.kwargs
    assert list(got["synthetic_data"].columns) == list(df.columns)
    assert len(got["synthetic_data"]) == len(df)
    assert got["model"] is model
    assert got["target_col"] == "y"
    assert got["test_data"] is df
    assert model.sample_calls == [(len(df), None)]


@pytest.mark.parametrize(
    "default_only", [{"dimensions": ["fidelity"]}, {"weights": {"fidelity": 1.0}}]
)
def test_custom_pipeline_rejects_default_pipeline_options(data, default_only):
    _, df = data
    model = _fitted(data)
    with pytest.raises(TypeError, match="configure your pipeline instead"):
        model.evaluate(df, pipeline=_RecordingPipeline(), **default_only)


def test_custom_pipeline_needs_a_run_method(data):
    _, df = data
    model = _fitted(data)
    with pytest.raises(TypeError, match="run\\(\\) method"):
        model.evaluate(df, pipeline=object())
