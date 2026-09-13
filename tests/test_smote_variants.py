import json

import pandas as pd
import pytest

from katabatic.models.smote import models as smote_models
from katabatic.models.smote.models import SMOTEModel


def _write_training_data(tmp_path, df):
    data_dir = tmp_path / "data"
    synth_dir = tmp_path / "synth"

    data_dir.mkdir()
    synth_dir.mkdir()

    df.to_csv(data_dir / "train_full.csv", index=False)

    return str(data_dir), str(synth_dir)


def test_invalid_variant_is_rejected():
    with pytest.raises(ValueError, match="variant must be one of"):
        SMOTEModel(variant="invalid")


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

    model.train(data_dir, synth_dir)

    assert model.variant == "smotenc"
    assert model.smote.k_neighbors == 3

    # 10 majority and 6 minority -> SMOTE-NC generates 4 samples.
    assert (6, 4) in calls

    output = pd.read_csv(f"{synth_dir}/x_synth.csv")

    assert set(output["workclass"]).issubset(set(df["workclass"]))


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

    model.train(data_dir, synth_dir)

    assert model.variant == "smoten"
    assert model.smote.k_neighbors == 3

    # 9 majority and 5 minority -> SMOTE-N generates 4 samples.
    assert (5, 4) in calls

    output = pd.read_csv(f"{synth_dir}/x_synth.csv")

    assert set(output["feature_a"]).issubset(set(df["feature_a"]))
    assert set(output["feature_b"]).issubset(set(df["feature_b"]))


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

    model.train(data_dir, synth_dir)

    # Only two minority samples are available, so k=5 cannot be used.
    assert model.smote.k_neighbors == 1

    with open(
        f"{synth_dir}/metadata.json",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["training"]["k_neighbors"] == 1


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

    model.train(data_dir, synth_dir)

    first = model.sample()
    second = model.sample()

    pd.testing.assert_frame_equal(first, second)


def test_default_variant_remains_smote():
    model = SMOTEModel()

    assert model.variant == "smote"
    assert model.k_neighbors == 5
