"""Tests for benchmarks/metadata.py."""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from metadata import (  # noqa: E402
    calculate_file_hash,
    save_experiment_metadata,
)


@dataclass
class FakeConfig:
    model_name: str = "ctgan"
    dataset_name: str = "adult"
    seed: int = 42
    test_size: float = 0.2
    categorical_cols: list | None = None


def test_calculate_file_hash_returns_sha256(tmp_path):
    dataset = tmp_path / "adult.csv"
    dataset.write_text("age,class\n25,yes\n30,no\n")

    expected = hashlib.sha256(dataset.read_bytes()).hexdigest()

    assert calculate_file_hash(str(dataset)) == expected


def test_save_experiment_metadata_writes_expected_fields(tmp_path):
    dataset = tmp_path / "adult.csv"
    dataset.write_text("age,class\n25,yes\n30,no\n")

    results_dir = tmp_path / "results"
    config = FakeConfig(categorical_cols=["class"])

    path = save_experiment_metadata(
        str(results_dir),
        config,
        str(dataset),
        model_settings={"epochs": 100, "batch_size": 512},
    )

    metadata = json.loads(Path(path).read_text())

    assert metadata["experiment"]["model_name"] == "ctgan"
    assert metadata["experiment"]["dataset_name"] == "adult"
    assert metadata["dataset"]["file_name"] == "adult.csv"
    assert metadata["dataset"]["file_size_bytes"] == dataset.stat().st_size
    assert metadata["dataset"]["sha256"] == calculate_file_hash(str(dataset))
    assert metadata["configuration"]["seed"] == 42
    assert metadata["seed"] == 42
    assert metadata["model_settings"]["epochs"] == 100
    assert metadata["model_settings"]["batch_size"] == 512
    assert "recorded_at" in metadata["experiment"]


def test_save_metadata_creates_results_directory(tmp_path):
    dataset = tmp_path / "adult.csv"
    dataset.write_text("age,class\n25,yes\n")

    results_dir = tmp_path / "new" / "results"

    path = save_experiment_metadata(
        str(results_dir),
        FakeConfig(),
        str(dataset),
    )

    assert results_dir.is_dir()
    assert Path(path).is_file()


def test_save_metadata_raises_for_missing_dataset(tmp_path):
    missing_dataset = tmp_path / "missing.csv"

    try:
        save_experiment_metadata(
            str(tmp_path / "results"),
            FakeConfig(),
            str(missing_dataset),
        )
    except FileNotFoundError as error:
        assert "Dataset file not found" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_model_settings_default_to_empty_dictionary(tmp_path):
    dataset = tmp_path / "adult.csv"
    dataset.write_text("age,class\n25,yes\n")

    path = save_experiment_metadata(
        str(tmp_path / "results"),
        FakeConfig(),
        str(dataset),
    )

    metadata = json.loads(Path(path).read_text())

    assert metadata["model_settings"] == {}
