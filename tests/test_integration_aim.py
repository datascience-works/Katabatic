"""Integration smoke tests for AIM (requires katabatic[aim])."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import require_backend

require_backend("mbi", "Dataset")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.aim.models import AIMModel  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


@pytest.mark.integration
@pytest.mark.aim
def test_aim_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")

    model = AIMModel(
        epsilon=1.0,
        delta=1e-5,
        degree=2,
        max_bins=4,
        max_cells=100,
        max_model_size=20,
        max_iters=100,
        rounds=3,
        seed=42,
    )

    pipe = TrainTestSplitPipeline(model=model)

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="aim",
        model_kwargs={
            "categorical_cols": ["f1", "y"],
            "continuous_cols": ["f0"],
        },
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]

    assert re.match(
        r"^models/aim_smoke_train-\d{8}-\d{6}$",
        mr.root_relpath,
    ), mr.root_relpath

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")

    assert x_synth.is_file()
    assert y_synth.is_file()

    ev = res["evaluation_refs"][0]

    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = AIMModel.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")

    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = AIMModel.load_from_ref(store, mr)

    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)

    assert len(out) == 10
    assert list(out.columns) == ["f0", "f1", "y"]
