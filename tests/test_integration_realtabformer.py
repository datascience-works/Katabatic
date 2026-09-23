"""Integration smoke tests for REaLTabFormer (requires katabatic[realtabformer])."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import require_backend

require_backend("realtabformer", "REaLTabFormer")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.realtabformer.models import REaLTabFormerModel  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


@pytest.mark.integration
@pytest.mark.realtabformer
def test_realtabformer_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")

    model = REaLTabFormerModel(epochs=1, batch_size=2, device="cpu")
    pipe = TrainTestSplitPipeline(model=model)

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="realtabformer",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(
        r"^models/realtabformer_smoke_train-\d{8}-\d{6}$", mr.root_relpath
    ), mr.root_relpath

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file()
    assert y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    for state_file in REaLTabFormerModel.ARTIFACT_STATE_FILES:
        state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
        assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = REaLTabFormerModel.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(5)
    assert len(out) == 5
    assert list(out.columns) == reloaded.column_names
