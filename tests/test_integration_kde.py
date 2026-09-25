"""Integration smoke tests for the KDE synthesizer (core deps only, no extras)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from katabatic.artifacts import LocalArtifactStore
from katabatic.models.kde.models import KDESynthesizer
from katabatic.models.registry import ModelRegistry
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


@pytest.mark.integration
@pytest.mark.kde
def test_kde_registry_load():
    cls = ModelRegistry.load_model("kde")
    assert cls.__name__ == "KDESynthesizer"
    assert hasattr(cls, "train")
    assert hasattr(cls, "sample")
    assert ModelRegistry.is_supported("kde")


@pytest.mark.integration
@pytest.mark.kde
def test_kde_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipe = TrainTestSplitPipeline(model=KDESynthesizer())

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="kde",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/kde_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
        mr.root_relpath
    )

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file() and y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = KDESynthesizer.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = KDESynthesizer.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
    assert list(out.columns) == ["f0", "f1", "y"]

    train_csv = store.open_path(f"{res['dataset_ref'].train_relpath}/train_full.csv")
    assert len(reloaded.sample()) == len(pd.read_csv(train_csv))

    with pytest.raises(NotImplementedError):
        reloaded.evaluate()


def test_kde_categorical_detection_from_info_json(tmp_path):
    import json

    data_dir = tmp_path / "car_like"
    data_dir.mkdir()

    df = pd.DataFrame(
        {
            "0": [0, 1, 2, 0, 1, 2, 0, 1],
            "1": [3, 3, 3, 3, 3, 3, 3, 3],
            "6": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(data_dir / "train_full.csv", index=False)
    (data_dir / "info.json").write_text(
        json.dumps({"cat_col_idx": [0, 1], "target_col_idx": [2]})
    )

    model = KDESynthesizer()
    model.train(data_dir=str(data_dir), synthetic_dir=str(tmp_path / "synth"))

    assert model._kde.feature_types_["0"] == "categorical"
    assert model._kde.feature_types_["1"] == "categorical"

    sampled = model.sample(n_samples=20)
    assert len(sampled) == 20
    # Categorical columns must only ever emit values observed in training data.
    assert set(sampled["0"].unique()).issubset({0, 1, 2})


def test_kde_continuous_columns_are_sampled_independently(tmp_path):
    """Regression: a fixed KDE seed made every column copy the same training row."""
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": 2 * x1 + rng.normal(scale=0.05, size=n),
            "y": rng.integers(0, 2, size=n),
        }
    )
    data_dir = tmp_path / "corr"
    data_dir.mkdir()
    df.to_csv(data_dir / "train_full.csv", index=False)

    model = KDESynthesizer(seed=42)
    model.train(
        str(data_dir), categorical_cols=[], synthetic_dir=str(tmp_path / "synth")
    )

    first = model.sample(n)
    second = model.sample(n)

    # Features are independent given the class, so the real x1/x2 correlation
    # (~1.0) must not carry over into the synthetic data.
    assert abs(first["x1"].corr(first["x2"])) < 0.2
    # Successive draws are fresh, and a fresh model with the same seed replays them.
    assert not np.allclose(first["x1"], second["x1"])
    replay = KDESynthesizer(seed=42)
    replay.train(
        str(data_dir), categorical_cols=[], synthetic_dir=str(tmp_path / "synth2")
    )
    assert np.allclose(replay.sample(n)["x1"], first["x1"])
