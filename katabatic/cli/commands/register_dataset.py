from __future__ import annotations

from pathlib import Path

from katabatic.artifacts.local import LocalArtifactStore
from katabatic.datasets.compatibility import check_dataset_for_model
from katabatic.datasets.registry import DatasetRegistry


def register_dataset_cli(
    dataset_name: str,
    csv_path: str,
    *,
    target_column: str | None = None,
    artifact_root: str | None = None,
    check_model: str | None = None,
) -> None:
    root = Path(artifact_root or "artifacts")
    store = LocalArtifactStore(root)
    reg = DatasetRegistry(store)
    entry = reg.register_if_absent(dataset_name, csv_path, target_column=target_column)
    print(f"Registered dataset {dataset_name!r} at {store.root / 'registry' / 'datasets.json'}")
    print(f"  task={entry['task']!r}, n_rows={entry['n_rows']}, target={entry['target_column']!r}")
    if check_model:
        ok, msg = check_dataset_for_model(entry, check_model)
        print(f"  compatibility with {check_model!r}: {msg}" + ("" if ok else " (failed)"))
        if not ok:
            raise SystemExit(1)
