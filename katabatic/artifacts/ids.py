from __future__ import annotations

from datetime import UTC, datetime


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def new_split_id() -> str:
    return f"split-{_utc_stamp()}"


def new_train_id() -> str:
    return f"train-{_utc_stamp()}"


def new_eval_id() -> str:
    return f"eval-{_utc_stamp()}"
