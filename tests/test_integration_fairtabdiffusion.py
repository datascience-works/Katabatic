"""Integration smoke tests for FairTabDiffusion (requires katabatic[fairtabdiffusion])."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import require_backend

require_backend("torch", "save")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.fairtabdiffusion.models import FairTabDiffusion  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)

FAST = {"epochs": 5, "timesteps": 10, "hidden": 32, "batch_size": 16}


@pytest.mark.integration
@pytest.mark.fairtabdiffusion
def test_fairtabdiffusion_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipe = TrainTestSplitPipeline(model=FairTabDiffusion(sensitive_col="f1", **FAST))

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="fairtabdiffusion",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(
        r"^models/fairtabdiffusion_smoke_train-\d{8}-\d{6}$",
        mr.root_relpath,
    ), mr.root_relpath

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file()
    assert y_synth.is_file()

    x_synth_df = pd.read_csv(x_synth)
    y_synth_df = pd.read_csv(y_synth)
    assert len(x_synth_df) == len(y_synth_df)
    assert list(x_synth_df.columns) == ["f0", "f1"]
    assert list(y_synth_df.columns) == ["y"]

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = FairTabDiffusion.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = FairTabDiffusion.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
    assert list(out.columns) == ["f0", "f1", "y"]
    assert reloaded.sample(10, seed=3).equals(reloaded.sample(10, seed=3))

    train_csv = store.open_path(f"{res['dataset_ref'].train_relpath}/train_full.csv")
    assert len(reloaded.sample()) == len(pd.read_csv(train_csv))

    with pytest.raises(NotImplementedError):
        reloaded.evaluate()


@pytest.mark.integration
@pytest.mark.fairtabdiffusion
@pytest.mark.parametrize("balanced", [True, False])
def test_fairtabdiffusion_label_distribution_follows_balanced_flag(tmp_path, balanced):
    """Regression: balanced_sampling=False used to leave the label uniform."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    df = pd.DataFrame({"x": rng.normal(size=n) + 3 * y, "y": y})
    df.to_csv(tmp_path / "train_full.csv", index=False)

    model = FairTabDiffusion(
        epochs=40, timesteps=20, hidden=32, batch_size=64, balanced_sampling=balanced
    )
    model.train(str(tmp_path), continuous_cols=["x"])

    # Real minority share is ~0.15; uniform would be 0.5. 0.4 separates the two
    # with margin across seeds/torch builds (observed 0.13-0.31 vs 0.49-0.57).
    minority_share = (model.sample(2000, seed=0)["y"] == 1).mean()
    if balanced:
        assert 0.4 < minority_share < 0.65
    else:
        assert minority_share < 0.4
