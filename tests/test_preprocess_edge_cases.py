"""Edge-case tests for katabatic/utils/preprocess.py."""

import pandas as pd
import pytest

from katabatic.utils.preprocess import (
    load_and_clean_data,
    preprocess_dataset,
)


def test_preprocess_dataset_empty_dataset_raises(tmp_path):
    raw = tmp_path / "empty.csv"
    raw.write_text("feature,label\n")

    out_path = tmp_path / "out.csv"

    with pytest.raises(IndexError):
        preprocess_dataset(str(raw), str(out_path))


def test_load_and_clean_data_drops_all_null_columns(tmp_path):
    raw = tmp_path / "all_null.csv"
    raw.write_text("feature,empty,label\n1,?,yes\n2,?,no\n")

    result = load_and_clean_data(str(raw))

    assert "empty" not in result.columns
    assert list(result.columns) == ["feature", "label"]


def test_preprocess_dataset_single_class_target(tmp_path):
    raw = tmp_path / "single_class.csv"
    raw.write_text("feature,label\n1,yes\n2,yes\n3,yes\n")

    out_path = tmp_path / "out.csv"

    preprocess_dataset(str(raw), str(out_path))

    result = pd.read_csv(out_path)

    assert result["label"].tolist() == ["yes", "yes", "yes"]
    assert result["label"].nunique() == 1


def test_preprocess_dataset_invalid_target_column_lists_available_columns(tmp_path):
    raw = tmp_path / "invalid_target.csv"
    raw.write_text("feature,label\n1,yes\n2,no\n")

    out_path = tmp_path / "out.csv"

    with pytest.raises(ValueError) as exc_info:
        preprocess_dataset(
            str(raw),
            str(out_path),
            target_col="invalid_column",
        )

    assert "invalid_column" in str(exc_info.value)
    assert "feature" in str(exc_info.value)
    assert "label" in str(exc_info.value)
