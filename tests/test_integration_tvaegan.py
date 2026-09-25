"""Integration smoke tests for TVAE-GAN (requires katabatic[tvaegan])."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import require_backend

require_backend("torch", "save")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.registry import ModelRegistry  # noqa: E402
from katabatic.models.tvaegan.models import TVAEGANModel  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


@pytest.mark.integration
@pytest.mark.tvaegan
def test_tvaegan_registry_load():
    cls = ModelRegistry.load_model("tvaegan")
    assert cls.__name__ == "TVAEGANModel"
    assert hasattr(cls, "train")
    assert hasattr(cls, "sample")
    assert ModelRegistry.is_supported("tvaegan")


@pytest.mark.integration
@pytest.mark.tvaegan
def test_tvaegan_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    # Epochs go to the constructor: the pipeline routes model_kwargs into
    # train(), which TVAEGANModel does not read hyperparameters from.
    model = TVAEGANModel(epochs=2, seed=42)
    pipe = TrainTestSplitPipeline(model=model)

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="tvaegan",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/tvaegan_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
        mr.root_relpath
    )

    x_synth = store.open_path(f"{mr.synthetic_relpath}/x_synth.csv")
    y_synth = store.open_path(f"{mr.synthetic_relpath}/y_synth.csv")
    assert x_synth.is_file() and y_synth.is_file()

    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert Path(store.open_path(ev.metrics_relpath)).is_file()
    assert Path(store.open_path(ev.report_relpath)).is_file()

    state_file = TVAEGANModel.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = TVAEGANModel.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
    assert list(out.columns) == ["f0", "f1", "y"]

    # The reloaded weights must be identical, not merely loadable: with the
    # same seed the reloaded decoder must reproduce the trained model's output.
    pd.testing.assert_frame_equal(model.sample(10, seed=7), reloaded.sample(10, seed=7))

    # sample() only exercises the decoder. evaluate() runs the encoder and
    # discriminator too, so it proves all three networks were restored.
    test_dir = store.open_path(res["dataset_ref"].test_relpath)
    loss = reloaded.evaluate(data_dir=str(test_dir), split="test")
    assert math.isfinite(loss)


@pytest.mark.integration
@pytest.mark.tvaegan
def test_tvaegan_load_from_ref_missing_state(tmp_path):
    from katabatic.artifacts.refs import ModelRef

    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = ModelRef(
        model_name="tvaegan",
        dataset_name="missing",
        dataset_version="v1",
        train_run_id="none",
    )
    with pytest.raises(FileNotFoundError):
        TVAEGANModel.load_from_ref(store, ref)
