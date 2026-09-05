"""Integration smoke tests for GANBLR (requires katabatic[ganblr])."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import require_backend

require_backend("tensorflow", "keras")
pytest.importorskip("pgmpy")
pytest.importorskip("pyitlib")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.ganblr.models import GANBLR  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


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

    state_file = GANBLR.ARTIFACT_STATE_FILES[0]
    state_pkl = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_pkl.is_file(), f"state file was never written: {state_pkl}"

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file()
    assert y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    reloaded = GANBLR.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
