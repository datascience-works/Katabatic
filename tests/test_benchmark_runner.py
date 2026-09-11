from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.runner import RunConfig, resolve_column_types


def test_resolve_column_types_auto_detects():
    df = pd.DataFrame(
        {
            "age": [21, 25, 30],
            "income": [50000.0, 62000.0, 71000.0],
            "city": ["Melbourne", "Sydney", "Brisbane"],
            "target": ["yes", "no", "yes"],
        }
    )

    config = RunConfig(
        dataset_name="test",
        model_name="test_model",
        target_col_raw="target",
    )

    resolve_column_types(config, df)

    assert config.categorical_cols == ["city"]
    assert config.continuous_cols == ["age", "income"]


def test_resolve_column_types_excludes_target():
    df = pd.DataFrame(
        {
            "age": [21, 25, 30],
            "city": ["Melbourne", "Sydney", "Brisbane"],
            "target": ["yes", "no", "yes"],
        }
    )

    config = RunConfig(
        dataset_name="test",
        model_name="test_model",
        target_col_raw="target",
    )

    resolve_column_types(config, df)

    assert "target" not in config.categorical_cols
    assert "target" not in config.continuous_cols


def test_resolve_column_types_preserves_manual_values():
    df = pd.DataFrame(
        {
            "education_code": [1, 2, 3],
            "age": [21, 25, 30],
            "target": ["yes", "no", "yes"],
        }
    )

    config = RunConfig(
        dataset_name="test",
        model_name="test_model",
        target_col_raw="target",
        categorical_cols=["education_code"],
        continuous_cols=["age"],
    )

    resolve_column_types(config, df)

    assert config.categorical_cols == ["education_code"]
    assert config.continuous_cols == ["age"]


def test_resolve_column_types_rejects_partial_manual_configuration():
    df = pd.DataFrame(
        {
            "age": [21, 25, 30],
            "city": ["Melbourne", "Sydney", "Brisbane"],
            "target": ["yes", "no", "yes"],
        }
    )

    config = RunConfig(
        dataset_name="test",
        model_name="test_model",
        target_col_raw="target",
        categorical_cols=["city"],
    )

    with pytest.raises(ValueError, match="categorical_cols and continuous_cols"):
        resolve_column_types(config, df)


def test_resolve_column_types_rejects_missing_target():
    df = pd.DataFrame(
        {
            "age": [21, 25, 30],
            "city": ["Melbourne", "Sydney", "Brisbane"],
        }
    )

    config = RunConfig(
        dataset_name="test",
        model_name="test_model",
        target_col_raw="target",
    )

    with pytest.raises(ValueError, match="target"):
        resolve_column_types(config, df)


def test_resolve_column_types_uses_known_dataset_spec():
    df = pd.DataFrame(
        {
            "fLength": [1, 2, 3],
            "fWidth": [1, 2, 3],
            "fSize": [1, 2, 3],
            "fConc": [1, 2, 3],
            "fConc1": [1, 2, 3],
            "fAsym": [1, 2, 3],
            "fM3Long": [1, 2, 3],
            "fM3Trans": [1, 2, 3],
            "fAlpha": [1, 2, 3],
            "fDist": [1, 2, 3],
            "class": ["g", "h", "g"],
        }
    )

    config = RunConfig(
        dataset_name="magic",
        model_name="test_model",
        target_col_raw="temporary",
    )

    resolve_column_types(config, df)

    assert config.target_col_raw == "class"
    assert config.categorical_cols == []
    assert config.continuous_cols == [
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
    ]


def test_resolve_column_types_model_override_has_priority():
    df = pd.DataFrame(
        {
            "fLength": [1, 2, 3],
            "class": ["g", "h", "g"],
        }
    )

    config = RunConfig(
        dataset_name="magic",
        model_name="special_model",
        target_col_raw="class",
        categorical_cols=["fLength"],
        continuous_cols=[],
    )

    resolve_column_types(config, df)

    assert config.categorical_cols == ["fLength"]
    assert config.continuous_cols == []
