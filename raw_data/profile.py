from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from katabatic.artifacts.refs import artifact_path_segment


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    nuniq = series.nunique(dropna=True)
    if nuniq <= 32 and not pd.api.types.is_float_dtype(series):
        return "categorical"
    return "text"


def infer_task_for_target(y: pd.Series) -> tuple[str, int | None]:
    """Return (task_string, n_classes or None)."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10:
        return "regression", None
    classes = pd.unique(y.dropna())
    n_classes = len(classes)
    if n_classes <= 1:
        return "multiclass_classification", n_classes
    if n_classes == 2:
        return "binary_classification", 2
    return "multiclass_classification", n_classes


def infer_dataset_profile(
    csv_path: str | Path,
    *,
    target_column: str | None = None,
    dataset_name: str,
) -> dict[str, Any]:
    """
    Build registry metadata from a CSV (header row).
    """
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 1:
        raise ValueError("CSV has no columns")

    if target_column is None:
        target_column = str(df.columns[-1])
    if target_column not in df.columns:
        raise ValueError(f"target column {target_column!r} not in CSV columns")

    y = df[target_column]
    task, n_classes = infer_task_for_target(y)

    column_schema: list[dict[str, Any]] = []
    for name in df.columns:
        col = df[name]
        kind = _column_kind(col)
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "n_unique": int(col.nunique(dropna=True)),
        }
        if name == target_column:
            entry["role"] = "target"
        else:
            entry["role"] = "feature"
        column_schema.append(entry)

    logical = artifact_path_segment(dataset_name)
    return {
        "dataset_name": dataset_name,
        "dataset_key": logical,
        "source_csv": str(csv_path),
        "target_column": target_column,
        "task": task,
        "n_classes": n_classes,
        "n_rows": len(df),
        "n_features": int(df.shape[1] - 1),
        "column_schema": column_schema,
    }
