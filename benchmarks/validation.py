def validate_column_names(
    available_columns: list,
    categorical_cols: list,
    continuous_cols: list,
    target_col: str,
) -> None:
    """Validate configured feature columns against the processed dataset."""

    available = set(available_columns)

    categorical = set(categorical_cols or [])
    continuous = set(continuous_cols or [])

    missing_categorical = categorical - available
    missing_continuous = continuous - available

    if missing_categorical or missing_continuous:
        errors = []

        if missing_categorical:
            errors.append(
                f"Categorical columns not found: {sorted(missing_categorical)}"
            )

        if missing_continuous:
            errors.append(f"Continuous columns not found: {sorted(missing_continuous)}")

        raise ValueError("Column validation failed. " + " ".join(errors))

    if target_col in categorical or target_col in continuous:
        raise ValueError(
            f"Target column '{target_col}' cannot be included "
            "in categorical_cols or continuous_cols."
        )

    overlap = categorical & continuous

    if overlap:
        raise ValueError(
            f"Columns cannot be both categorical and continuous: {sorted(overlap)}"
        )
