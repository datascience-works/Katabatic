"""Integration smoke tests for GReaT (requires katabatic[great])."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import require_backend

require_backend("torch", "save")
require_backend("transformers", "AutoModelForCausalLM")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.great.models import GReaT  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


@pytest.mark.integration
@pytest.mark.great
def test_great_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")

    model = GReaT(llm="gpt2", epochs=3, batch_size=8, report_to=[])
    pipe = TrainTestSplitPipeline(model=model)

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="great",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/great_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
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

    for state_file in GReaT.ARTIFACT_STATE_FILES:
        state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
        assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = GReaT.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(5, device="cpu", k=5)
    assert len(out) > 0, "reloaded model failed to generate any samples"
    assert list(out.columns) == reloaded.columns
