"""Integration smoke tests for TabKDE (core deps only, no extras)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from katabatic.artifacts import LocalArtifactStore
from katabatic.artifacts.refs import ModelRef
from katabatic.models.base_model import EVALUATION_DIMENSIONS
from katabatic.models.registry import ModelRegistry
from katabatic.models.tabkde.models import TabKDEModel
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


@pytest.mark.integration
@pytest.mark.tabkde
def test_tabkde_registry_load():
    cls = ModelRegistry.load_model("tabkde")
    assert cls.__name__ == "TabKDEModel"
    assert hasattr(cls, "train")
    assert hasattr(cls, "sample")
    assert ModelRegistry.is_supported("tabkde")


@pytest.mark.integration
@pytest.mark.tabkde
def test_tabkde_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    model = TabKDEModel(random_state=42)
    pipe = TrainTestSplitPipeline(model=model)

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="tabkde",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/tabkde_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
        mr.root_relpath
    )

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file() and y_synth.is_file()

    # The inverse copula interpolates between training values, so a numeric
    # label would come back as fractions that TSTR classifiers reject. The
    # label must round-trip as exactly the original class values.
    y_values = pd.read_csv(y_synth)["y"]
    assert set(y_values.unique()) <= {0, 1}, sorted(y_values.unique())

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = TabKDEModel.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = TabKDEModel.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 10
    assert list(out.columns) == ["f0", "f1", "y"]

    # The reloaded state must be identical, not merely loadable.
    pd.testing.assert_frame_equal(model.sample(10, seed=7), reloaded.sample(10, seed=7))

    train_df = pd.read_csv(
        store.open_path(f"{res['dataset_ref'].train_relpath}/train_full.csv")
    )
    assert len(reloaded.sample()) == len(train_df)

    report = reloaded.evaluate(train_df, target_col="y")
    assert set(report.dimension_scores) == set(EVALUATION_DIMENSIONS)
    assert all(0.0 <= v <= 1.0 for v in report.dimension_scores.values())


@pytest.mark.integration
@pytest.mark.tabkde
def test_tabkde_numeric_labels_round_trip_exactly(tmp_path):
    """Regression: numeric labels came back interpolated (e.g. 0.37).

    The inverse copula interpolates between sorted training values, so without
    forcing the label through the categorical path roughly 2% of a large draw
    lands between classes, and TSTR classifiers reject the label column.
    A 2000-row seeded draw exposes this deterministically; the pipeline smoke
    test's small draw does not.
    """
    rng = np.random.default_rng(0)
    pd.DataFrame({"f0": rng.normal(size=30), "f1": rng.normal(size=30)}).to_csv(
        tmp_path / "x_train.csv", index=False
    )
    pd.DataFrame({"y": rng.integers(0, 2, 30)}).to_csv(
        tmp_path / "y_train.csv", index=False
    )

    model = TabKDEModel(random_state=0)
    model.train(tmp_path)
    labels = model.sample(2000, seed=0)["y"]

    assert set(labels.unique()) <= {0, 1}, sorted(labels.unique())[:10]


@pytest.mark.integration
@pytest.mark.tabkde
def test_tabkde_seed_is_reproducible(tmp_path, tiny_binary_csv):
    """sample(seed=...) must reach the sampler's own generator.

    Seeding numpy's global RNG does not, because the sampler draws from
    np.random.default_rng(random_state).
    """
    df = pd.read_csv(tiny_binary_csv)
    model = TabKDEModel()
    model.fit(df.drop(columns=["y"]), df["y"].astype("category"))

    pd.testing.assert_frame_equal(model.sample(15, seed=3), model.sample(15, seed=3))
    assert not model.sample(15, seed=3).equals(model.sample(15, seed=4))


@pytest.mark.integration
@pytest.mark.tabkde
def test_tabkde_successive_draws_are_fresh(tiny_binary_csv):
    """Regression: a fixed random_state made every sample() call return identical rows."""
    df = pd.read_csv(tiny_binary_csv)
    x, y = df.drop(columns=["y"]), df["y"].astype("category")

    model = TabKDEModel(random_state=5)
    model.fit(x, y)
    first = model.sample(15)
    assert not first.equals(model.sample(15))

    # A fresh model with the same random_state replays the sequence.
    replay = TabKDEModel(random_state=5)
    replay.fit(x, y)
    pd.testing.assert_frame_equal(replay.sample(15), first)


@pytest.mark.integration
@pytest.mark.tabkde
def test_tabkde_load_from_ref_missing_state(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = ModelRef(
        model_name="tabkde",
        dataset_name="missing",
        dataset_version="v1",
        train_run_id="none",
    )
    with pytest.raises(FileNotFoundError):
        TabKDEModel.load_from_ref(store, ref)
