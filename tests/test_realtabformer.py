from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from katabatic.models.realtabformer.models import REaLTabFormerModel
from katabatic.models.realtabformer.utils import (
    load_training_data,
    resolve_synth_dir,
    save_metadata,
    save_synthetic_data,
)


def test_model_initialization():
    model = REaLTabFormerModel(epochs=5, batch_size=4)

    assert model.epochs == 5
    assert model.batch_size == 4
    assert model.device == "cpu"
    assert model.random_state == 1029
    assert model.is_fitted is False


def test_invalid_epochs():
    with pytest.raises(ValueError, match="epochs must be greater than 0"):
        REaLTabFormerModel(epochs=0)


def test_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        REaLTabFormerModel(batch_size=0)


def test_required_dependencies():
    assert REaLTabFormerModel.get_required_dependencies() == ["realtabformer"]


def test_load_training_data_from_train_full(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [40000, 50000],
            "class": ["A", "B"],
        }
    )

    df.to_csv(tmp_path / "train_full.csv", index=False)

    loaded = load_training_data(tmp_path)

    pd.testing.assert_frame_equal(loaded, df)


def test_load_training_data_from_x_and_y(tmp_path):
    x_train = pd.DataFrame(
        {
            "age": [20, 30],
            "income": [40000, 50000],
        }
    )

    y_train = pd.DataFrame(
        {
            "class": ["A", "B"],
        }
    )

    x_train.to_csv(tmp_path / "x_train.csv", index=False)
    y_train.to_csv(tmp_path / "y_train.csv", index=False)

    loaded = load_training_data(tmp_path)

    expected = pd.concat([x_train, y_train], axis=1)

    pd.testing.assert_frame_equal(loaded, expected)


def test_sample_before_training_raises():
    model = REaLTabFormerModel()

    with pytest.raises(RuntimeError, match="must be trained before sampling"):
        model.sample(5)


def test_sample_returns_dataframe():
    model = REaLTabFormerModel()

    model.model = MagicMock()
    model.model.sample.return_value = pd.DataFrame(
        {
            "age": [21, 31],
            "class": ["A", "B"],
        }
    )

    model.is_fitted = True
    model.training_rows = 2
    model.column_names = ["age", "class"]

    result = model.sample(2)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    model.model.sample.assert_called_once_with(
        n_samples=2,
        device="cpu",
    )


def test_evaluate_before_training_raises():
    model = REaLTabFormerModel()

    with pytest.raises(RuntimeError, match="must be trained before evaluation"):
        model.evaluate()


def test_evaluate_after_training():
    model = REaLTabFormerModel()
    model.is_fitted = True

    assert model.evaluate() == 0.0


def test_save_synthetic_data(tmp_path):
    synthetic_df = pd.DataFrame(
        {
            "age": [21, 31],
            "income": [41000, 51000],
            "class": ["A", "B"],
        }
    )

    save_synthetic_data(
        synthetic_df,
        tmp_path,
        target_column="class",
    )

    assert (tmp_path / "x_synth.csv").exists()
    assert (tmp_path / "y_synth.csv").exists()

    x_synth = pd.read_csv(tmp_path / "x_synth.csv")
    y_synth = pd.read_csv(tmp_path / "y_synth.csv")

    assert list(x_synth.columns) == ["age", "income"]
    assert list(y_synth.columns) == ["class"]


def test_save_metadata(tmp_path):
    save_metadata(
        tmp_path,
        target_column="class",
        row_count=10,
        model_parameters={
            "epochs": 1,
            "batch_size": 8,
        },
    )

    assert (tmp_path / "metadata.json").exists()


def test_resolve_synth_dir_explicit(tmp_path):
    target_dir = tmp_path / "custom_output"

    result = resolve_synth_dir(
        target_dir,
        tmp_path,
    )

    assert result == target_dir
    assert result.exists()


@patch("realtabformer.REaLTabFormer")
def test_train_with_mock_model(mock_realtabformer, tmp_path):
    training_df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [40000, 50000, 60000],
            "class": ["A", "B", "A"],
        }
    )

    training_df.to_csv(tmp_path / "train_full.csv", index=False)

    synthetic_df = pd.DataFrame(
        {
            "age": [22, 32, 42],
            "income": [42000, 52000, 62000],
            "class": ["A", "B", "A"],
        }
    )

    mock_instance = MagicMock()
    mock_instance.sample.return_value = synthetic_df
    mock_realtabformer.return_value = mock_instance

    output_dir = tmp_path / "synthetic"

    model = REaLTabFormerModel(
        epochs=1,
        batch_size=2,
    )

    model.train(
        tmp_path,
        synthetic_dir=output_dir,
    )

    assert model.is_fitted is True
    assert model.target_column == "class"
    assert model.training_rows == 3

    mock_realtabformer.assert_called_once()

    mock_instance.fit.assert_called_once()

    fit_args, fit_kwargs = mock_instance.fit.call_args

    assert fit_kwargs["device"] == "cpu"
    assert fit_kwargs["n_critic"] == 0

    assert (output_dir / "x_synth.csv").exists()
    assert (output_dir / "y_synth.csv").exists()
    assert (output_dir / "metadata.json").exists()
