from __future__ import annotations

import re
from dataclasses import dataclass


def artifact_path_segment(name: str) -> str:
    """
    Sanitize a logical name for use inside artifact directory / file segments.
    Allows lowercase letters, digits, hyphen; coerces other chars to hyphen.
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unnamed"


@dataclass(frozen=True)
class DatasetRef:
    dataset_name: str
    dataset_version: str
    """Root relative to artifact store: datasets/<name>/<version>"""

    @property
    def root_relpath(self) -> str:
        return f"datasets/{self.dataset_name}/{self.dataset_version}"

    @property
    def train_relpath(self) -> str:
        return f"{self.root_relpath}/train"

    @property
    def test_relpath(self) -> str:
        return f"{self.root_relpath}/test"

    @property
    def extra_relpath(self) -> str:
        return f"{self.root_relpath}/extra"


@dataclass(frozen=True)
class ModelRef:
    model_name: str
    dataset_name: str
    dataset_version: str
    train_run_id: str

    @property
    def root_relpath(self) -> str:
        m = artifact_path_segment(self.model_name)
        d = artifact_path_segment(self.dataset_name)
        return f"models/{m}_{d}_{self.train_run_id}"

    @property
    def state_relpath(self) -> str:
        return f"{self.root_relpath}/state"

    @property
    def synthetic_relpath(self) -> str:
        return f"{self.root_relpath}/synthetic"


@dataclass(frozen=True)
class EvaluationRef:
    evaluation_type: str
    eval_run_id: str
    model_name: str
    dataset_name: str
    dataset_version: str
    train_run_id: str
    test_dataset_version: str

    @property
    def root_relpath(self) -> str:
        """Mirrors :attr:`ModelRef.root_relpath` stem under ``evaluations/``."""
        m = artifact_path_segment(self.model_name)
        d = artifact_path_segment(self.dataset_name)
        return f"evaluations/{m}_{d}_{self.train_run_id}"

    @property
    def metrics_relpath(self) -> str:
        et = artifact_path_segment(self.evaluation_type)
        return f"{self.root_relpath}/{et}_metrics.json"

    @property
    def report_relpath(self) -> str:
        et = artifact_path_segment(self.evaluation_type)
        return f"{self.root_relpath}/{et}_report.csv"
