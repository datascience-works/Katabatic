import pytest

from benchmarks.validation import validate_column_names


def test_valid_column_names():
    validate_column_names(
        available_columns=["age", "gender", "income", "target"],
        categorical_cols=["gender"],
        continuous_cols=["age", "income"],
        target_col="target",
    )


def test_missing_categorical_column():
    with pytest.raises(ValueError, match="Categorical columns not found"):
        validate_column_names(
            available_columns=["age", "target"],
            categorical_cols=["gender"],
            continuous_cols=["age"],
            target_col="target",
        )


def test_missing_continuous_column():
    with pytest.raises(ValueError, match="Continuous columns not found"):
        validate_column_names(
            available_columns=["gender", "target"],
            categorical_cols=["gender"],
            continuous_cols=["income"],
            target_col="target",
        )


def test_target_column_cannot_be_a_feature():
    with pytest.raises(ValueError, match="cannot be included"):
        validate_column_names(
            available_columns=["age", "target"],
            categorical_cols=[],
            continuous_cols=["target"],
            target_col="target",
        )


def test_column_cannot_be_both_categorical_and_continuous():
    with pytest.raises(ValueError, match="both categorical and continuous"):
        validate_column_names(
            available_columns=["age", "target"],
            categorical_cols=["age"],
            continuous_cols=["age"],
            target_col="target",
        )
