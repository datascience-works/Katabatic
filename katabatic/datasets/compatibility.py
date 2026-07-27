from __future__ import annotations

from typing import Any

from katabatic.models.registry import ModelRegistry


def _requirements_for_model(model_name: str) -> dict[str, Any] | None:
    info = ModelRegistry.get_model_info(model_name)
    if not info:
        return None
    return info.get("dataset_requirements")


def check_dataset_for_model(
    dataset_entry: dict[str, Any],
    model_name: str,
) -> tuple[bool, str]:
    """
    Return (ok, message). When no requirements are declared for the model, allow any dataset.
    """
    req = _requirements_for_model(model_name)
    if not req:
        return True, "no dataset_requirements declared for model"

    task = dataset_entry.get("task")
    allowed = req.get("allowed_tasks")
    if allowed and task not in allowed:
        return False, f"task {task!r} not in allowed_tasks {allowed!r}"

    n_classes = dataset_entry.get("n_classes")
    if n_classes is not None:
        mx = req.get("max_classes")
        if mx is not None and n_classes > mx:
            return False, f"n_classes {n_classes} exceeds max_classes {mx}"
        mn = req.get("min_classes")
        if mn is not None and n_classes < mn:
            return False, f"n_classes {n_classes} below min_classes {mn}"

    if req.get("requires_numeric_only"):
        for col in dataset_entry.get("column_schema", []):
            if col.get("role") == "feature" and col.get("kind") not in ("numeric",):
                return False, "model requires numeric-only feature columns"

    return True, "ok"
