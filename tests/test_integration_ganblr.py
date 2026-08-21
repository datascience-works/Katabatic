"""Integration smoke tests for GANBLR (requires katabatic[ganblr])."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from katabatic.artifacts import LocalArtifactStore
from katabatic.models.ganblr.models import GANBLR
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


@pytest.mark.integration
@pytest.mark.ganblr
def test_ganblr_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipe = TrainTestSplitPipeline(model=GANBLR())

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="ganblr",
        model_kwargs={"train_epochs": 1, "batch_size": 8},
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/ganblr_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
        mr.root_relpath
    )

    state_pkl = store.open_path(f"{mr.state_relpath}/ganblr_model.pkl")
    assert state_pkl.is_file(), "Expected pickled GANBLR state"

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file() and y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()
