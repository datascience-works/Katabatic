from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.local import LocalArtifactStore
from katabatic.artifacts.refs import artifact_path_segment
from katabatic.datasets.profile import infer_dataset_profile


def _iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


REGISTRY_RELPATH = "registry/datasets.json"


class DatasetRegistry:
    """JSON-backed dataset registry under an artifact store root."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    @classmethod
    def from_root(cls, root: str | Path) -> DatasetRegistry:
        return cls(LocalArtifactStore(root))

    @property
    def _relpath(self) -> str:
        return REGISTRY_RELPATH

    def _load_raw(self) -> dict[str, Any]:
        if not self._store.exists(self._relpath):
            return {"datasets": {}}
        return self._store.load_json(self._relpath)

    def _save_raw(self, data: dict[str, Any]) -> None:
        self._store.save_json(self._relpath, data)

    def get(self, dataset_name: str) -> dict[str, Any] | None:
        key = artifact_path_segment(dataset_name)
        return self._load_raw()["datasets"].get(key)

    def register(
        self,
        dataset_name: str,
        csv_path: str | Path,
        *,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        key = artifact_path_segment(dataset_name)
        raw = self._load_raw()
        if key in raw["datasets"]:
            raise ValueError(
                f"dataset name already registered: {dataset_name!r} (key {key!r})"
            )

        profile = infer_dataset_profile(
            csv_path, target_column=target_column, dataset_name=dataset_name
        )
        entry = {
            **profile,
            "registered_at": _iso_now(),
        }
        raw["datasets"][key] = entry
        self._save_raw(raw)
        return entry

    def register_if_absent(
        self,
        dataset_name: str,
        csv_path: str | Path,
        *,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Register when the logical name is missing; otherwise return the existing entry.
        Emits a warning if column names differ from a fresh profile of ``csv_path``.
        """
        _ = artifact_path_segment(dataset_name)
        existing = self.get(dataset_name)
        if existing is None:
            return self.register(dataset_name, csv_path, target_column=target_column)

        tc = target_column or existing.get("target_column")
        try:
            profile = infer_dataset_profile(
                csv_path, target_column=tc, dataset_name=dataset_name
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Could not re-profile CSV for registered dataset {dataset_name!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return existing

        old_names = [c["name"] for c in existing.get("column_schema", [])]
        new_names = [c["name"] for c in profile.get("column_schema", [])]
        if old_names != new_names:
            warnings.warn(
                f"Registered dataset {dataset_name!r} column schema differs from {csv_path!r}",
                UserWarning,
                stacklevel=2,
            )
        return existing
