"""Integration smoke tests for TabEBM."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from katabatic.artifacts import LocalArtifactStore
from katabatic.models.tabebm.models import TabEBMModel
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline


@pytest.mark.integration
@pytest.mark.tabebm
def test_tabebm_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipe = TrainTestSplitPipeline(model=TabEBMModel())

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="tabebm",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(
        r"^models/tabebm_smoke_train-\d{8}-\d{6}$",
        mr.root_relpath,
    ), mr.root_relpath

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file()
    assert y_synth.is_file()

    x_synth_df = __import__("pandas").read_csv(x_synth)
    y_synth_df = __import__("pandas").read_csv(y_synth)

    assert len(x_synth_df) == len(y_synth_df)
    assert list(x_synth_df.columns) == ["f0", "f1"]
    assert list(y_synth_df.columns) == ["y"]

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = TabEBMModel.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = TabEBMModel.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
    assert list(out.columns) == ["f0", "f1", "y"]

    # Omitting n_samples defaults to the number of training rows.
    default_out = reloaded.sample()
    assert len(default_out) == len(reloaded._x_train)

    with pytest.raises(NotImplementedError):
        reloaded.evaluate()
