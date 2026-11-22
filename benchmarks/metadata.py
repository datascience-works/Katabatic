"""Utilities for saving reproducible benchmark experiment metadata."""

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime


def calculate_file_hash(file_path: str) -> str:
    """Return the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_experiment_metadata(
    results_dir: str,
    config,
    raw_data_path: str,
    model_settings: dict | None = None,
) -> str:
    """Save the configuration and dataset fingerprint for an experiment."""

    if not os.path.isfile(raw_data_path):
        raise FileNotFoundError(f"Dataset file not found: {raw_data_path}")

    if is_dataclass(config):
        config_data = asdict(config)
    else:
        config_data = dict(vars(config))

    metadata = {
        "experiment": {
            "model_name": config.model_name,
            "dataset_name": config.dataset_name,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        "dataset": {
            "file_name": os.path.basename(raw_data_path),
            "file_size_bytes": os.path.getsize(raw_data_path),
            "sha256": calculate_file_hash(raw_data_path),
        },
        "configuration": config_data,
        "seed": config_data.get("seed"),
        "model_settings": model_settings or {},
    }

    os.makedirs(results_dir, exist_ok=True)

    metadata_path = os.path.join(results_dir, "experiment_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, default=str)

    return metadata_path
