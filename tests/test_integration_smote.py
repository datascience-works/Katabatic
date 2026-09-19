"""Integration tests for the SMOTE model family (requires katabatic[smote])."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import require_backend

require_backend("imblearn", "over_sampling")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.models.smote import models as smote_models  # noqa: E402
from katabatic.models.smote.models import SMOTEModel  # noqa: E402
from katabatic.pipeline.train_test_split.pipeline import (  # noqa: E402
    TrainTestSplitPipeline,
)


def _write_training_data(tmp_path, df):
    data_dir = tmp_path / "data"
    synth_dir = tmp_path / "synth"

    data_dir.mkdir()
    synth_dir.mkdir()

    df.to_csv(data_dir / "train_full.csv", index=False)

    return str(data_dir), str(synth_dir)


@pytest.mark.integration
@pytest.mark.smote
def test_smote_artifact_pipeline_smoke(tmp_path, tiny_binary_csv):
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipe = TrainTestSplitPipeline(model=SMOTEModel())

    res = pipe.run(
        input_csv=str(tiny_binary_csv),
        dataset_name="smoke",
        artifact_store=store,
        model_name="smote",
        test_size=0.3,
        seed=42,
    )

    mr = res["model_ref"]
    assert re.match(r"^models/smote_smoke_train-\d{8}-\d{6}$", mr.root_relpath), (
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

    state_file = SMOTEModel.ARTIFACT_STATE_FILES[0]
    state_path = store.open_path(f"{mr.state_relpath}/{state_file}")
    assert state_path.is_file(), f"state file was never written: {state_path}"

    reloaded = SMOTEModel.load_from_ref(store, mr)
    assert reloaded.is_fitted, "reloaded model is not marked fitted"

    out = reloaded.sample(10)
    assert len(out) == 10
    assert list(out.columns) == list(reloaded.column_names)


@pytest.mark.integration
@pytest.mark.smote
def test_smotenc_handles_mixed_data_and_uses_aligned_schedule(
    tmp_path,
    monkeypatch,
):
    df = pd.DataFrame(
        {
            "age": [
                25,
                31,
                42,
                29,
                35,
                46,
                38,
                50,
                27,
                44,
                33,
                36,
                41,
                48,
                39,
                45,
            ],
            "hours": [
                40,
                38,
                45,
                40,
                42,
                50,
                37,
                45,
                39,
                41,
                36,
                44,
                46,
                48,
                43,
                47,
            ],
            "workclass": [
                "Private",
                "Gov",
                "Private",
                "Self",
                "Private",
                "Gov",
                "Private",
                "Self",
                "Private",
                "Gov",
                "Private",
                "Gov",
                "Self",
                "Private",
                "Gov",
                "Self",
            ],
            "class": ["majority"] * 10 + ["minority"] * 6,
        }
    )

    data_dir, synth_dir = _write_training_data(tmp_path, df)

    calls = []
    original_helper = smote_models._paper_aligned_anchor_rows

    def spy_anchor_rows(n_anchors, n_samples, random_state):
        calls.append((n_anchors, n_samples))
        return original_helper(
            n_anchors,
            n_samples,
            random_state,
        )

    monkeypatch.setattr(
        smote_models,
        "_paper_aligned_anchor_rows",
        spy_anchor_rows,
    )

    model = SMOTEModel(
        variant="smotenc",
        categorical_features=["workclass"],
        k_neighbors=3,
        sampling_strategy="auto",
        random_state=42,
    )

    model.train(data_dir, synthetic_dir=synth_dir)

    assert model.variant == "smotenc"
    assert model.smote.k_neighbors == 3

    # 10 majority and 6 minority -> SMOTE-NC generates 4 samples.
    assert (6, 4) in calls

    output = pd.read_csv(f"{synth_dir}/x_synth.csv")

    assert set(output["workclass"]).issubset(set(df["workclass"]))


@pytest.mark.integration
@pytest.mark.smote
def test_smoten_handles_categorical_data_and_uses_aligned_schedule(
    tmp_path,
    monkeypatch,
):
    df = pd.DataFrame(
        {
            "feature_a": [
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "x",
                "x",
                "y",
                "y",
                "x",
            ],
            "feature_b": [
                "low",
                "med",
                "high",
                "low",
                "med",
                "high",
                "med",
                "high",
                "low",
                "low",
                "med",
                "high",
                "med",
                "high",
            ],
            "class": ["majority"] * 9 + ["minority"] * 5,
        }
    )

    data_dir, synth_dir = _write_training_data(tmp_path, df)

    calls = []
    original_helper = smote_models._paper_aligned_anchor_rows

    def spy_anchor_rows(n_anchors, n_samples, random_state):
        calls.append((n_anchors, n_samples))
        return original_helper(
            n_anchors,
            n_samples,
            random_state,
        )

    monkeypatch.setattr(
        smote_models,
        "_paper_aligned_anchor_rows",
        spy_anchor_rows,
    )

    model = SMOTEModel(
        variant="smoten",
        k_neighbors=3,
        sampling_strategy="auto",
        random_state=42,
    )

    model.train(data_dir, synthetic_dir=synth_dir)

    assert model.variant == "smoten"
    assert model.smote.k_neighbors == 3

    # 9 majority and 5 minority -> SMOTE-N generates 4 samples.
    assert (5, 4) in calls

    output = pd.read_csv(f"{synth_dir}/x_synth.csv")

    assert set(output["feature_a"]).issubset(set(df["feature_a"]))
    assert set(output["feature_b"]).issubset(set(df["feature_b"]))


@pytest.mark.integration
@pytest.mark.smote
def test_tiny_class_adjusts_k_neighbors(tmp_path):
    df = pd.DataFrame(
        {
            "feature_a": [
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "x",
                "y",
            ],
            "feature_b": [
                "low",
                "med",
                "high",
                "med",
                "high",
                "low",
                "low",
                "high",
            ],
            "class": [
                "majority",
                "majority",
                "majority",
                "majority",
                "majority",
                "majority",
                "minority",
                "minority",
            ],
        }
    )

    data_dir, synth_dir = _write_training_data(tmp_path, df)

    model = SMOTEModel(
        variant="smoten",
        k_neighbors=5,
        random_state=42,
    )

    model.train(data_dir, synthetic_dir=synth_dir)

    # Only two minority samples are available, so k=5 cannot be used.
    assert model.smote.k_neighbors == 1

    with open(
        f"{synth_dir}/metadata.json",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["training"]["k_neighbors"] == 1


@pytest.mark.integration
@pytest.mark.smote
def test_smoten_same_seed_is_reproducible(tmp_path):
    df = pd.DataFrame(
        {
            "feature_a": [
                "a",
                "b",
                "c",
                "a",
                "b",
                "c",
                "a",
                "b",
                "x",
                "x",
                "y",
                "y",
                "x",
            ],
            "feature_b": [
                "low",
                "med",
                "high",
                "low",
                "med",
                "high",
                "med",
                "high",
                "low",
                "med",
                "high",
                "med",
                "high",
            ],
            "class": ["majority"] * 8 + ["minority"] * 5,
        }
    )

    data_dir, synth_dir = _write_training_data(tmp_path, df)

    model = SMOTEModel(
        variant="smoten",
        k_neighbors=3,
        random_state=42,
    )

    model.train(data_dir, synthetic_dir=synth_dir)

    first = model.sample()
    second = model.sample()

    pd.testing.assert_frame_equal(first, second)
