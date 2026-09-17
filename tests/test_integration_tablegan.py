from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("torch")

from katabatic.models.tablegan.models import TableGANModel


@pytest.mark.integration
@pytest.mark.tablegan
def test_tablegan_train_and_sample(tmp_path) -> None:
    data_dir = tmp_path / "split"
    synthetic_dir = tmp_path / "synthetic"

    data_dir.mkdir()

    train_df = pd.DataFrame(
        {
            "age": [
                21.0,
                25.0,
                29.0,
                34.0,
                38.0,
                42.0,
                47.0,
                52.0,
            ],
            "income": [
                30000.0,
                35000.0,
                42000.0,
                48000.0,
                55000.0,
                61000.0,
                68000.0,
                75000.0,
            ],
            "city": [
                "Melbourne",
                "Sydney",
                "Brisbane",
                "Melbourne",
                "Sydney",
                "Brisbane",
                "Melbourne",
                "Sydney",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    train_df.to_csv(
        data_dir / "train_full.csv",
        index=False,
    )

    model = TableGANModel(
        epochs=1,
        batch_size=4,
        noise_dim=16,
        base_channels=8,
        seed=42,
        device="cpu",
    )

    returned_model = model.train(
        str(data_dir),
        str(synthetic_dir),
        categorical_cols=["city", "target"],
        continuous_cols=["age", "income"],
    )

    assert returned_model is model
    assert model.is_fitted

    synthetic = model.sample(5)

    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape == (5, 4)
    assert synthetic.columns.tolist() == train_df.columns.tolist()

    assert (
        synthetic["age"]
        .between(
            train_df["age"].min(),
            train_df["age"].max(),
        )
        .all()
    )

    assert (
        synthetic["income"]
        .between(
            train_df["income"].min(),
            train_df["income"].max(),
        )
        .all()
    )

    assert set(synthetic["city"]).issubset(set(train_df["city"]))
    assert set(synthetic["target"]).issubset({"0", "1"})

    assert (synthetic_dir / "x_synth.csv").is_file()
    assert (synthetic_dir / "y_synth.csv").is_file()
    assert (synthetic_dir / "metadata.json").is_file()

    saved_x = pd.read_csv(synthetic_dir / "x_synth.csv")
    saved_y = pd.read_csv(synthetic_dir / "y_synth.csv")

    assert saved_x.shape == (len(train_df), 3)
    assert saved_y.shape == (len(train_df), 1)

    with open(
        synthetic_dir / "metadata.json",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["model"] == "TableGAN"
    assert metadata["schema"]["columns"] == train_df.columns.tolist()
    assert metadata["matrix_side_length"] == 4


@pytest.mark.integration
def test_tablegan_x_y_training_files(tmp_path) -> None:
    data_dir = tmp_path / "split"
    synthetic_dir = tmp_path / "synthetic"

    data_dir.mkdir()

    x_train = pd.DataFrame(
        {
            "feature_1": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
            ],
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
        }
    )

    y_train = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    x_train.to_csv(
        data_dir / "x_train.csv",
        index=False,
    )

    y_train.to_csv(
        data_dir / "y_train.csv",
        index=False,
    )

    model = TableGANModel(
        epochs=1,
        batch_size=4,
        noise_dim=16,
        base_channels=8,
        seed=42,
        device="cpu",
    )

    model.train(
        str(data_dir),
        str(synthetic_dir),
        categorical_cols=["category", "target"],
        continuous_cols=["feature_1"],
    )

    synthetic = model.sample(3)

    assert synthetic.shape == (3, 3)
    assert synthetic.columns.tolist() == [
        "feature_1",
        "category",
        "target",
    ]


@pytest.mark.integration
def test_tablegan_missing_training_files(tmp_path) -> None:
    model = TableGANModel(
        epochs=1,
        batch_size=4,
    )

    with pytest.raises(FileNotFoundError):
        model.train(str(tmp_path))
