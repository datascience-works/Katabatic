"""Integration smoke tests for CTGAN (requires katabatic[ctgan])."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")

from katabatic.artifacts import LocalArtifactStore
from katabatic.models.ctgan.models import CTGANModel
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


@pytest.mark.integration
@pytest.mark.ctgan
def test_ctgan_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipe = TrainTestSplitPipeline(model=CTGANModel())

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="ctgan",
        model_kwargs={"epochs": 1},
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/ctgan_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
        mr.root_relpath
    )

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file()
    assert y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = CTGANModel.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = CTGANModel.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
    assert list(out.columns) == list(reloaded.output_order)
