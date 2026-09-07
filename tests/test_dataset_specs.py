from pathlib import Path

import pandas as pd
import pytest

from katabatic.datasets.specs import get_dataset_spec, validate_dataset_spec

DATASET_DIR = Path("katabatic/datasets")


@pytest.mark.parametrize(
    "dataset_name,target_col",
    [
        ("adult", "class"),
        ("car", "6"),
        ("magic", "class"),
        ("nursery", "8"),
        ("shuttle", "class"),
    ],
)
def test_dataset_specs_match_real_csvs(dataset_name, target_col):
    df = pd.read_csv(DATASET_DIR / f"{dataset_name}.csv")

    spec = validate_dataset_spec(df, dataset_name)

    assert spec["target_col"] == target_col
    assert target_col in df.columns


def test_adult_spec():
    spec = get_dataset_spec("adult")

    assert "workclass" in spec["categorical_cols"]
    assert "education-num" in spec["categorical_cols"]
    assert "age" in spec["continuous_cols"]
    assert "fnlwgt" in spec["continuous_cols"]


def test_car_is_all_categorical():
    spec = get_dataset_spec("car")

    assert spec["categorical_cols"] == ["0", "1", "2", "3", "4", "5"]
    assert spec["continuous_cols"] == []


def test_magic_is_all_continuous():
    spec = get_dataset_spec("magic")

    assert spec["categorical_cols"] == []
    assert len(spec["continuous_cols"]) == 10
    assert "fLength" in spec["continuous_cols"]
    assert "fDist" in spec["continuous_cols"]


def test_nursery_is_all_categorical():
    spec = get_dataset_spec("nursery")

    assert spec["categorical_cols"] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    assert spec["continuous_cols"] == []


def test_shuttle_is_all_continuous():
    spec = get_dataset_spec("shuttle")

    assert spec["categorical_cols"] == []
    assert spec["continuous_cols"] == [
        "time",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
    ]


def test_validation_rejects_missing_column():
    df = pd.read_csv(DATASET_DIR / "adult.csv").drop(columns=["age"])

    with pytest.raises(ValueError, match="missing expected columns"):
        validate_dataset_spec(df, "adult")


def test_unknown_dataset_returns_none():
    assert get_dataset_spec("unknown_dataset") is None
