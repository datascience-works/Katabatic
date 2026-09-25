"""Unit tests for katabatic/evaluate/stability/evaluation.py."""

import pandas as pd
import pytest

from katabatic.evaluate.stability.evaluation import StabilityEvaluation


class _SeededModel:
    """Returns a small, seed-dependent perturbation of real_data."""

    def __init__(self, real_data):
        self._real = real_data

    def sample(self, n_samples, seed=0):
        df = self._real.copy().iloc[:n_samples].reset_index(drop=True)
        df["num"] = df["num"] + seed
        return df


class _UnseededModel:
    """Does not accept a seed kwarg at all -> triggers the TypeError fallback."""

    def __init__(self, real_data):
        self._real = real_data

    def sample(self, n_samples):
        return self._real.copy().iloc[:n_samples].reset_index(drop=True)


class _FailingModel:
    def sample(self, n_samples, seed=0):
        raise RuntimeError("boom")


class _NoneReturningModel:
    def sample(self, n_samples, seed=0):
        return None


class _WrongTypeModel:
    def sample(self, n_samples, seed=0):
        return [1, 2, 3]


class _UnseededAndFailingModel:
    """No seed support, and the unseeded fallback call also raises."""

    def sample(self, n_samples):
        raise RuntimeError("still broken")


@pytest.fixture
def real_data():
    return pd.DataFrame({"cat": ["a", "b"] * 10, "num": list(range(20))})


def test_seeded_model_produces_varying_scores(real_data):
    ev = StabilityEvaluation(
        real_data,
        _SeededModel(real_data),
        n_runs=3,
        seeds=[0, 1, 2],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    assert results["n_runs"] == 3
    assert results["seeds"] == [0, 1, 2]
    assert len(results["per_run_scores"]) == 3
    assert 0.0 <= results["stability_score"] <= 1.0
    assert "fidelity" in results["mean_scores"]
    assert "diversity" in results["mean_scores"]


def test_seed_len_mismatch_raises():
    real = pd.DataFrame({"num": [1, 2, 3]})
    with pytest.raises(ValueError, match="len\\(seeds\\)"):
        StabilityEvaluation(real, model=object(), n_runs=3, seeds=[0, 1])


def test_default_seeds_and_n_samples(real_data):
    ev = StabilityEvaluation(real_data, _SeededModel(real_data), n_runs=2)
    assert ev.seeds == [0, 1]
    assert ev.n_samples == len(real_data)


def test_unseeded_model_falls_back_and_warns(real_data, capsys):
    ev = StabilityEvaluation(
        real_data,
        _UnseededModel(real_data),
        n_runs=2,
        seeds=[0, 1],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    results = ev.evaluate()

    captured = capsys.readouterr()
    assert "does not accept a 'seed' argument" in captured.out
    # Deterministic model -> identical runs -> std 0 -> perfect stability.
    assert results["stability_score"] == 1.0


def test_model_raising_is_skipped(real_data, capsys):
    ev = StabilityEvaluation(real_data, _FailingModel(), n_runs=2, seeds=[0, 1])
    results = ev.evaluate()

    assert results == {"stability_score": 0.0, "error": "All sampling runs failed."}
    captured = capsys.readouterr()
    assert "model.sample() raised" in captured.out


def test_model_returning_none_is_skipped(real_data):
    ev = StabilityEvaluation(real_data, _NoneReturningModel(), n_runs=2, seeds=[0, 1])
    results = ev.evaluate()
    assert results == {"stability_score": 0.0, "error": "All sampling runs failed."}


def test_unseeded_fallback_that_also_raises_is_skipped(real_data, capsys):
    ev = StabilityEvaluation(real_data, _UnseededAndFailingModel(), n_runs=1, seeds=[0])
    results = ev.evaluate()

    assert results == {"stability_score": 0.0, "error": "All sampling runs failed."}
    captured = capsys.readouterr()
    assert "does not accept a 'seed' argument" in captured.out
    assert "model.sample() raised: still broken" in captured.out


def test_model_returning_wrong_type_raises(real_data):
    ev = StabilityEvaluation(real_data, _WrongTypeModel(), n_runs=1, seeds=[0])
    with pytest.raises(TypeError, match="must return a pandas DataFrame"):
        ev.evaluate()


def test_sample_aligns_extra_columns_to_real_data(real_data):
    class _ExtraColsModel:
        def sample(self, n_samples, seed=0):
            df = real_data.copy()
            df["extra"] = "unused"
            return df

    ev = StabilityEvaluation(
        real_data,
        _ExtraColsModel(),
        n_runs=1,
        seeds=[0],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    synth = ev._sample(0)
    assert list(synth.columns) == ["cat", "num"]


def test_print_summary_runs_without_error(real_data, capsys):
    ev = StabilityEvaluation(
        real_data,
        _SeededModel(real_data),
        n_runs=2,
        seeds=[0, 1],
        categorical_cols=["cat"],
        continuous_cols=["num"],
    )
    ev.evaluate()
    captured = capsys.readouterr()
    assert "Stability Evaluation" in captured.out
    assert "Runs completed" in captured.out
