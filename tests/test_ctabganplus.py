"""Integration smoke tests for CTAB-GAN+."""

from __future__ import annotations

import re

import pytest

from tests.conftest import require_backend

require_backend("torch", "save")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.ctabganplus.models import CTABGANPlus  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


@pytest.mark.integration
@pytest.mark.ctabganplus
def test_ctabganplus_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")

    model = CTABGANPlus(
        config={
            "epochs": 2,
            "batch_size": 10,
            "class_dim": (16, 16),
            "num_channels": 8,
        }
    )
    pipe = TrainTestSplitPipeline(model=model)

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="ctabganplus",
        model_kwargs={
            "categorical_cols": ["f1"],
            "continuous_cols": ["f0"],
        },
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(
        r"^models/ctabganplus_smoke_train-\d{8}-\d{6}$",
        mr.root_relpath,
    ), mr.root_relpath

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file()
    assert y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert store.open_path(ev.metrics_relpath).is_file()
    assert store.open_path(ev.report_relpath).is_file()

    fitted_model = pipe.last_model
    assert fitted_model.is_fitted, "model was not marked fitted after training"

    out = fitted_model.sample(10)
    assert len(out) == 10


@pytest.mark.ctabganplus
def test_ctabganplus_sample_is_seed_reproducible(tmp_path, tiny_binary_csv):
    """Regression test for the seed-handling bug flagged in PR review."""
    import pandas as pd

    df = pd.read_csv(tiny_binary_csv)
    x_train = df.drop(columns=["y"])
    y_train = df[["y"]]

    data_dir = tmp_path / "split"
    data_dir.mkdir()
    x_train.to_csv(data_dir / "x_train.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False)

    model = CTABGANPlus(
        config={
            "epochs": 2,
            "batch_size": 10,
            "class_dim": (16, 16),
            "num_channels": 8,
        }
    )
    model.train(
        str(data_dir),
        categorical_cols=["f1"],
        continuous_cols=["f0"],
    )

    df1 = model.sample(10, seed=123)
    df2 = model.sample(10, seed=123)

    assert df1.equals(df2), "sample() is not reproducible for a fixed seed"
