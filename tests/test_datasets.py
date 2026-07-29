from __future__ import annotations


import pandas as pd

from katabatic.artifacts import LocalArtifactStore
from katabatic.datasets.compatibility import check_dataset_for_model
from katabatic.datasets.registry import DatasetRegistry


def test_dataset_registry_register_and_duplicate(tmp_path):
    csv_path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv_path, index=False)
    store = LocalArtifactStore(tmp_path / "art")
    reg = DatasetRegistry(store)
    reg.register("myds", str(csv_path))
    assert reg.get("myds") is not None
    try:
        reg.register("myds", str(csv_path))
        assert False, "expected duplicate error"
    except ValueError as e:
        assert "already registered" in str(e).lower()


def test_register_if_absent_skips_duplicate(tmp_path):
    csv_path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv_path, index=False)
    store = LocalArtifactStore(tmp_path / "art")
    reg = DatasetRegistry(store)
    reg.register("myds", str(csv_path))
    again = reg.register_if_absent("myds", str(csv_path))
    assert again == reg.get("myds")


def test_register_if_absent_creates_entry(tmp_path):
    csv_path = tmp_path / "e.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv_path, index=False)
    store = LocalArtifactStore(tmp_path / "art2")
    reg = DatasetRegistry(store)
    entry = reg.register_if_absent("fresh", str(csv_path))
    assert entry["n_rows"] == 2
    assert reg.get("fresh") is not None


def test_check_dataset_for_model_ganblr(tmp_path):
    csv_path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv_path, index=False)
    store = LocalArtifactStore(tmp_path / "art")
    reg = DatasetRegistry(store)
    entry = reg.register("bin", str(csv_path), target_column="y")
    ok, _ = check_dataset_for_model(entry, "ganblr")
    assert ok
