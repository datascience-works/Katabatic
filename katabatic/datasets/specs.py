from __future__ import annotations

from typing import Any

import pandas as pd

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "adult": {
        "target_col": "class",
        "categorical_cols": [
            "workclass",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        "continuous_cols": [
            "age",
            "fnlwgt",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ],
    },
    "car": {
        "target_col": "6",
        "categorical_cols": ["0", "1", "2", "3", "4", "5"],
        "continuous_cols": [],
    },
    "magic": {
        "target_col": "class",
        "categorical_cols": [],
        "continuous_cols": [
            "fLength",
            "fWidth",
            "fSize",
            "fConc",
            "fConc1",
            "fAsym",
            "fM3Long",
            "fM3Trans",
            "fAlpha",
            "fDist",
        ],
    },
    "nursery": {
        "target_col": "8",
        "categorical_cols": ["0", "1", "2", "3", "4", "5", "6", "7"],
        "continuous_cols": [],
    },
    "shuttle": {
        "target_col": "class",
        "categorical_cols": [],
        "continuous_cols": [
            "time",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
            "a7",
            "a8",
        ],
    },
}


def get_dataset_spec(dataset_name: str) -> dict[str, Any] | None:
    """Return the known specification for a benchmark dataset."""
    spec = DATASET_SPECS.get(dataset_name.lower())

    if spec is None:
        return None

    return {
        "target_col": spec["target_col"],
        "categorical_cols": list(spec["categorical_cols"]),
        "continuous_cols": list(spec["continuous_cols"]),
    }


def validate_dataset_spec(
    df: pd.DataFrame,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Validate that a DataFrame matches the expected schema for a known dataset.

    Returns the dataset specification when validation succeeds.
    """
    spec = get_dataset_spec(dataset_name)

    if spec is None:
        raise ValueError(f"No dataset specification found for {dataset_name!r}")

    expected_columns = (
        spec["categorical_cols"]
        + spec["continuous_cols"]
        + [spec["target_col"]]
    )

    missing_columns = [
        column for column in expected_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_name!r} is missing expected columns: "
            f"{missing_columns}"
        )

    return spec