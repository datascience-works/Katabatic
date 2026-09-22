from __future__ import annotations

import pandas as pd
import pytest

from katabatic.artifacts.local import LocalArtifactStore
from katabatic.artifacts.refs import ModelRef
from katabatic.models.mst import MSTModel


def test_mst_initial_state():
    model = MSTModel()

    assert model.is_fitted is False
    assert model.epsilon == 3.0
    assert model.delta is None


def test_mst_invalid_epsilon():
    with pytest.raises(
        ValueError,
        match="epsilon must be greater than 0",
    ):
        MSTModel(epsilon=0)


def test_mst_invalid_delta():
    with pytest.raises(
        ValueError,
        match="delta must be between 0 and 1",
    ):
        MSTModel(delta=1.0)


def test_mst_required_dependencies():
    assert MSTModel.get_required_dependencies() == [
        "snsynth",
        "mbi",
        "opendp",
    ]


def test_mst_sample_before_training():
    model = MSTModel()

    with pytest.raises(
        RuntimeError,
        match="Call train",
    ):
        model.sample(10)


def test_mst_evaluate_before_training():
    model = MSTModel()

    with pytest.raises(
        RuntimeError,
        match="Call train",
    ):
        model.evaluate()


def test_mst_infer_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["A", "B", "C"],
            "active": [True, False, True],
        }
    )

    result = MSTModel._infer_categorical_columns(df)

    assert result == [
        "city",
        "active",
    ]


def test_mst_resolve_default_delta():
    model = MSTModel()

    delta = model._resolve_delta(100)

    assert delta == pytest.approx(0.001)


def test_mst_resolve_explicit_delta():
    model = MSTModel(
        delta=0.0001,
    )

    assert model._resolve_delta(100) == 0.0001


@pytest.mark.integration
@pytest.mark.mst
def test_mst_artifact_round_trip(tmp_path):
    """
    Verify that MST can train, persist its fitted state,
    reload through ModelRef, and continue sampling.
    """
    pytest.importorskip("snsynth")
    pytest.importorskip("mbi")
    pytest.importorskip("opendp")

    data_dir = tmp_path / "data"
    synth_dir = tmp_path / "synthetic"
    artifacts_root = tmp_path / "artifacts"

    data_dir.mkdir()

    base_df = pd.DataFrame(
        {
            "age": [
                20,
                25,
                30,
                35,
                40,
                45,
            ],
            "city": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
            "label": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )
    df = pd.concat([base_df] * 10, ignore_index=True)

    df.to_csv(
        data_dir / "train_full.csv",
        index=False,
    )

    # Create Katabatic's local artifact store.
    store = LocalArtifactStore(
        artifacts_root,
    )

    # Reference representing this trained MST artifact.
    ref = ModelRef(
        model_name="mst",
        dataset_name="test",
        dataset_version="v1",
        train_run_id="run1",
    )

    # Resolve the standard artifact-state directory.
    artifact_state_dir = store.open_path(
        ref.state_relpath,
    )

    artifact_state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Create and train MST.
    model = MSTModel(
        epsilon=3.0,
        categorical_columns=[
            "age",
            "city",
            "label",
        ],
    )

    model.train(
        str(data_dir),
        synthetic_dir=str(synth_dir),
        artifact_state_dir=str(artifact_state_dir),
    )

    # Confirm training completed.
    assert model.is_fitted is True

    # Confirm model state was persisted.
    assert (artifact_state_dir / "mst_state.pkl").exists()

    # Confirm synthetic outputs were generated.
    assert (synth_dir / "x_synth.csv").exists()

    assert (synth_dir / "y_synth.csv").exists()

    assert (synth_dir / "metadata.json").exists()

    # Confirm the original trained model can sample.
    original_sample = model.sample(3)

    assert len(original_sample) == 3

    assert list(original_sample.columns) == [
        "age",
        "city",
        "label",
    ]

    # Reload the fitted model through Katabatic's
    # standard artifact mechanism.
    restored = MSTModel.load_from_ref(
        store,
        ref,
    )

    # Confirm restored model state.
    assert restored.is_fitted is True

    assert restored.epsilon == 3.0

    assert restored.column_names == [
        "age",
        "city",
        "label",
    ]

    # Verify that the reloaded model can still generate data.
    restored_sample = restored.sample(3)

    assert len(restored_sample) == 3

    assert list(restored_sample.columns) == [
        "age",
        "city",
        "label",
    ]

    with pytest.raises(NotImplementedError):
        restored.evaluate()
