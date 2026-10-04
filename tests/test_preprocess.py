"""Unit tests for katabatic/utils/preprocess.py."""

import pandas as pd

from katabatic.utils.preprocess import (
    fill_categorical_nulls,
    load_and_clean_data,
    preprocess_dataset,
    preprocess_tabular,
    process_numerical_columns,
)


def test_load_and_clean_data_treats_question_mark_as_nan(tmp_path):
    csv = tmp_path / "raw.csv"
    csv.write_text("a,b\n1,?\n2,3\n")

    df = load_and_clean_data(str(csv))

    assert df["b"].isna().sum() == 1


def test_load_and_clean_data_drops_all_nan_columns(tmp_path):
    csv = tmp_path / "raw.csv"
    csv.write_text("a,b,c\n1,?,x\n2,?,y\n")

    df = load_and_clean_data(str(csv))

    assert "b" not in df.columns
    assert list(df.columns) == ["a", "c"]


def test_process_numerical_columns_fills_median_and_drops_constant():
    df = pd.DataFrame(
        {
            "num": [1.0, None, 3.0],
            "const": [5, 5, 5],
            "cat": ["x", "y", "z"],
        }
    )

    out = process_numerical_columns(df)

    assert out["num"].isna().sum() == 0
    assert out["num"].iloc[1] == 2.0  # median of [1, 3]
    assert "const" not in out.columns
    assert "cat" in out.columns  # untouched, non-numeric


def test_process_numerical_columns_handles_non_coercible_gracefully():
    # A column that looks numerical-ish per is_numerical but has bad values
    # should not blow up the whole pipeline.
    df = pd.DataFrame({"num": [1, 2, 3]})
    out = process_numerical_columns(df)
    assert list(out["num"]) == [1.0, 2.0, 3.0]


def test_fill_categorical_nulls_normalises_missing_values():
    df = pd.DataFrame({"cat": ["a", None, "nan", "  b  "]})

    out = fill_categorical_nulls(df)

    assert out["cat"].tolist() == ["a", "Missing", "Missing", "b"]


def test_fill_categorical_nulls_ignores_numeric_columns():
    df = pd.DataFrame({"num": [1, 2, 3]})
    out = fill_categorical_nulls(df)
    assert out["num"].tolist() == [1, 2, 3]


def test_preprocess_dataset_default_target_is_last_column(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("f0,f1,label\n1,a,yes\n2,b,no\n3,?,yes\n")
    out_path = tmp_path / "out" / "clean.csv"

    preprocess_dataset(str(raw), str(out_path))

    result = pd.read_csv(out_path)
    assert list(result.columns) == ["f0", "f1", "label"]
    assert result["label"].tolist() == ["yes", "no", "yes"]
    assert result["f1"].tolist() == ["a", "b", "Missing"]


def test_preprocess_dataset_explicit_target_col(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("f0,f1,label\n1,a,yes\n2,b,no\n")
    out_path = tmp_path / "out" / "clean.csv"

    preprocess_dataset(str(raw), str(out_path), target_col="f1")

    result = pd.read_csv(out_path)
    # f1 moved to last position as the target; f0 and label are features.
    assert list(result.columns) == ["f0", "label", "f1"]
    assert result["f1"].tolist() == ["a", "b"]


def test_preprocess_dataset_missing_target_col_raises(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("f0,f1\n1,2\n")
    out_path = tmp_path / "out.csv"

    try:
        preprocess_dataset(str(raw), str(out_path), target_col="nope")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "nope" in str(e)


def test_preprocess_dataset_target_nan_filled_as_missing(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("f0,label\n1,?\n2,yes\n")
    out_path = tmp_path / "out.csv"

    preprocess_dataset(str(raw), str(out_path))

    result = pd.read_csv(out_path)
    assert result["label"].tolist() == ["Missing", "yes"]


def test_preprocess_tabular_is_alias_for_preprocess_dataset():
    assert preprocess_tabular is preprocess_dataset
