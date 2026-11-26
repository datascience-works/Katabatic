"""Unit tests for NaiveBayesModel: fitting, dtypes, ranges, save and reload."""

# Lets type hints like "str | None" work on all Python versions (matches the rest of the repo).
from __future__ import annotations

# Standard library logging, used to check the model logs messages instead of printing.
import logging

import numpy as np
import pandas as pd
import pytest

# The model and helper functions under test.
from katabatic.models.naivebayes.models import NaiveBayesModel
from katabatic.models.naivebayes.utils import infer_feature_types, is_integer_valued

# Tag every test in this file with the "naivebayes" marker (already defined in pyproject.toml),
# so they run with:  pytest -m naivebayes
pytestmark = pytest.mark.naivebayes


# ---------------------------------------------------------------------------
# Fixtures: reusable test data, created fresh for each test.
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_df():
    """Small dataset built to trigger every known Naive Bayes bug."""
    # Fixed seed so the test data is the same on every run.
    rng = np.random.default_rng(0)
    n = 600  # Number of rows: big enough for stable stats, small enough to be fast.
    df = pd.DataFrame(
        {
            # Integer-coded categorical (16 codes, like Adult's education-num).
            # Bug: used to be treated as continuous, producing codes like -8 or 24.
            "edu_num": rng.integers(1, 17, n),
            # Integer continuous column stored as float (as happens when a CSV has blanks).
            # Bug: used to come out as unrounded floats or all NaN.
            "age": rng.integers(17, 91, n).astype(float),
            # Mostly zeros with a few large values (like capital-gain).
            # Bug: the unbounded Gaussian produced negative values.
            "gain": np.where(rng.random(n) < 0.9, 0, rng.integers(1000, 99999, n)),
            # Plain text categorical, to make sure strings still work.
            "colour": rng.choice(["red", "green", "blue"], n),
            # Target with 3 classes, one of them rare (5%), to test class balance.
            "y": rng.choice([0, 1, 2], n, p=[0.7, 0.25, 0.05]),
        }
    )
    # Blank out the first 5 ages (rows 0-4) to simulate missing values.
    df.loc[:4, "age"] = np.nan
    return df


@pytest.fixture
def data_dir(tmp_path, mixed_df):
    """Write mixed_df as train_full.csv in a temp folder, the format train() expects."""
    # tmp_path is a unique temporary folder pytest creates (and cleans up) for each test.
    d = tmp_path / "split"
    d.mkdir()
    # index=False so pandas doesn't add an extra unnamed index column.
    mixed_df.to_csv(d / "train_full.csv", index=False)
    return d


def _train(data_dir, tmp_path, **kwargs):
    """Helper: train a model with a fixed seed, saving outputs into the temp folder."""
    model = NaiveBayesModel(seed=7)
    model.train(
        str(data_dir),
        # Synthetic CSVs go to the temp folder, not into the repo.
        synthetic_dir=str(tmp_path / "synth"),
        # Passing this makes train() save the pickle state file (needed for reload tests).
        artifact_state_dir=str(tmp_path / "state"),
        **kwargs,  # Lets a test pass extra options like categorical_cols if needed.
    )
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_is_integer_valued():
    """The helper must detect whole numbers by value, not by dtype."""
    assert is_integer_valued(pd.Series([1, 2, 3]))  # Normal int column.
    assert is_integer_valued(
        pd.Series([1.0, np.nan, 3.0])
    )  # Ints stored as float with a blank: the bug case.
    assert not is_integer_valued(
        pd.Series([1.5, 2.0])
    )  # Has a real decimal, so not integer.
    assert not is_integer_valued(pd.Series(["a", "b"]))  # Text is never integer.
    assert not is_integer_valued(
        pd.Series([True, False])
    )  # Booleans are flags, not numbers.


def test_low_cardinality_ints_detected_as_categorical(mixed_df):
    """Auto-detection (no column lists given) should classify each column correctly."""
    # All columns except the target.
    features = [c for c in mixed_df.columns if c != "y"]
    types = infer_feature_types(mixed_df, features)
    assert (
        types["edu_num"] == "categorical"
    )  # 16 integer codes (<= 20), so categorical.
    assert types["colour"] == "categorical"  # Text, so categorical.
    assert types["age"] == "continuous"  # 74 distinct values (> 20), so continuous.
    assert types["gain"] == "continuous"  # Many distinct values, so continuous.


def test_train_writes_synthetic_files(data_dir, tmp_path, mixed_df):
    """train() must create x_synth.csv and y_synth.csv with the right shape and columns."""
    _train(data_dir, tmp_path)
    x = pd.read_csv(tmp_path / "synth" / "x_synth.csv")
    y = pd.read_csv(tmp_path / "synth" / "y_synth.csv")
    # By default train() generates as many rows as the training data.
    assert len(x) == len(y) == len(mixed_df)
    # Feature columns keep their original order; the target is in its own file.
    assert list(x.columns) == ["edu_num", "age", "gain", "colour"]
    assert list(y.columns) == ["y"]


def test_integer_columns_stay_integer(data_dir, tmp_path):
    """The handover bug: integer columns must come out as an integer dtype."""
    model = _train(data_dir, tmp_path)
    out = model.sample(1000)
    for col in ["edu_num", "age", "gain"]:
        # The f-string after the comma is the message shown if the assert fails.
        assert pd.api.types.is_integer_dtype(out[col]), f"{col} is {out[col].dtype}"


def test_no_missing_values_generated(data_dir, tmp_path):
    """Blanks in training data must not turn into NaN in synthetic data."""
    out = _train(data_dir, tmp_path).sample(1000)
    # .isna() marks NaN cells; the first .any() checks each column, the second checks all columns.
    assert not out.isna().any().any()


def test_values_stay_inside_training_range(data_dir, tmp_path, mixed_df):
    """No impossible values: continuous within min/max, categoricals only from seen values."""
    out = _train(data_dir, tmp_path).sample(2000)
    for col in ["age", "gain"]:
        # Pandas min()/max() skip the NaNs in "age" automatically.
        assert out[col].min() >= mixed_df[col].min()  # e.g. no negative gain.
        assert out[col].max() <= mixed_df[col].max()
    # "<=" on sets means "is a subset of": no new, made-up categories.
    assert set(out["edu_num"]) <= set(mixed_df["edu_num"])  # e.g. no code 24.
    assert set(out["colour"]) <= set(mixed_df["colour"])


def test_every_class_appears_and_rows_are_shuffled(data_dir, tmp_path):
    """The rare class must still appear, and rows must not be grouped by class."""
    out = _train(data_dir, tmp_path).sample(200)
    # All 3 classes present, including the rare 5% one (about 10 of 200 rows).
    assert set(out["y"]) == {0, 1, 2}
    # If rows were grouped by class, y would be sorted in one direction.
    assert not out["y"].is_monotonic_increasing
    assert not out["y"].is_monotonic_decreasing


def test_same_seed_gives_same_output(data_dir, tmp_path):
    """Two models with the same seed and data must produce identical synthetic data."""
    # Separate sub-folders so the two runs don't overwrite each other's files.
    a = _train(data_dir, tmp_path / "a").sample(300)
    b = _train(data_dir, tmp_path / "b").sample(300)
    # Fails with a clear diff if any value, dtype or column differs.
    pd.testing.assert_frame_equal(a, b)


def test_save_and_reload(data_dir, tmp_path, monkeypatch):
    """A saved model must reload with the same settings and still be able to sample."""
    model = _train(data_dir, tmp_path)
    # Path of the pickle file train() should have written ("naivebayes_state.pkl").
    state_file = tmp_path / "state" / NaiveBayesModel.ARTIFACT_STATE_FILES[0]
    assert state_file.is_file()

    # load_from_ref() normally finds the file through an artifact store.
    # monkeypatch temporarily replaces _require_state_file so it returns our file directly
    # (the real store is already covered by tests/test_integration_naivebayes.py).
    # pytest undoes the replacement after the test.
    monkeypatch.setattr(
        NaiveBayesModel,
        "_require_state_file",
        classmethod(lambda cls, s, r: state_file),
    )
    # store and ref aren't needed because of the patch above.
    reloaded = NaiveBayesModel.load_from_ref(store=None, ref=None)

    assert reloaded.is_fitted  # Ready to sample.
    assert reloaded.feature_types_ == model.feature_types_  # Same column types.
    assert (
        reloaded._cont_bounds_ == model._cont_bounds_
    )  # Bounds saved and loaded (edits 2k and 2l).
    out = reloaded.sample(50)
    # Features first, target last, same as the original model.
    assert list(out.columns) == ["edu_num", "age", "gain", "colour", "y"]
    assert len(out) == 50


def test_sample_before_train_raises():
    """Sampling an untrained model must give a clear error, not a crash."""
    # The test passes only if the code inside the "with" raises RuntimeError.
    with pytest.raises(RuntimeError):
        NaiveBayesModel().sample(10)


def test_logs_instead_of_print(data_dir, tmp_path, caplog, capsys):
    """Messages must go through logging, and the old print() must be gone."""
    # caplog captures log records; at_level makes sure INFO messages from this module are recorded.
    with caplog.at_level(logging.INFO, logger="katabatic.models.naivebayes.models"):
        _train(data_dir, tmp_path)
    # The logger.info from edit 2j.
    assert "synthetic data saved" in caplog.text
    # The logger.warning from edit 2d (the 5 blank ages).
    assert "missing" in caplog.text
    # capsys captures anything printed; the old print started with "[NaiveBayes]".
    assert "[NaiveBayes]" not in capsys.readouterr().out
