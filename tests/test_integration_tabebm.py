"""Integration smoke tests for TabEBM."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from katabatic.artifacts import LocalArtifactStore
from katabatic.models.base_model import EVALUATION_DIMENSIONS
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

    train_df = pd.read_csv(
        store.open_path(f"{res['dataset_ref'].train_relpath}/train_full.csv")
    )
    report = reloaded.evaluate(train_df, target_col="y")
    assert set(report.dimension_scores) == set(EVALUATION_DIMENSIONS)
    assert all(0.0 <= v <= 1.0 for v in report.dimension_scores.values())


def test_tabebm_rare_class_survives_subsampling(tmp_path):
    """Regression: a class dropped by subsampling crashed or shifted labels."""
    import numpy as np
    import pandas as pd

    from katabatic.models.tabebm.models import TabEBMConfig

    # Class "b" is rare and appears between "a" and "c", so a positional key
    # mismatch would mislabel "c" rows as "b".
    labels = ["a"] * 600 + ["b"] * 2 + ["c"] * 600
    centre = {"a": 0.0, "b": 50.0, "c": 100.0}
    rng = np.random.default_rng(0)
    x = [centre[lab] + rng.normal() for lab in labels]
    pd.DataFrame({"x": x}).to_csv(tmp_path / "x_train.csv", index=False)
    pd.DataFrame({"y": labels}).to_csv(tmp_path / "y_train.csv", index=False)

    model = TabEBMModel(config=TabEBMConfig(max_data_size=100, sgld_steps=20))
    model.train(str(tmp_path))
    out = model.sample(3000)

    assert set(out["y"]) == {"a", "b", "c"}
    for lab, mean in out.groupby("y")["x"].mean().items():
        assert abs(mean - centre[lab]) < 10, (lab, mean)
