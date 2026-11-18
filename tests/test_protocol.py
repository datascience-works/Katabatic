"""Tests for benchmarks/protocol.py and benchmarks/check_protocol.py."""

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import check_protocol  # noqa: E402
import protocol  # noqa: E402


def test_split_constants_match_runner_defaults():
    runner = (BENCHMARKS / "runner.py").read_text()
    assert f"test_size: float = {protocol.TEST_SIZE}" in runner
    assert f"seed: int = {protocol.SEED}" in runner


@pytest.mark.parametrize(
    "dataset,expected",
    [("car", 100), ("nursery", 100), ("magic", 100), ("adult", 150), ("shuttle", 150)],
)
def test_epochs_for_dataset(dataset, expected):
    assert protocol.epochs_for(dataset) == expected


def test_every_model_folder_has_a_spec():
    folders = {p.name for p in (BENCHMARKS / "examples").iterdir() if p.is_dir()}
    assert folders == set(protocol.MODEL_SPECS)


def test_every_spec_uses_a_known_policy():
    policies = {protocol.STANDARD, protocol.SPECIFIC, protocol.FIXED}
    assert all(s.policy in policies for s in protocol.MODEL_SPECS.values())


def _script(tmp_path, body):
    path = tmp_path / "run_ctgan_car.py"
    path.write_text(body)
    return path


def test_check_script_ok(tmp_path):
    path = _script(tmp_path, "m = M(epochs=100, seed=42)\n")
    row = check_protocol.check_script(path, "ctgan", "car")
    assert row["seed_status"] == "OK"
    assert row["budget_status"] == "OK"


def test_check_script_detects_wrong_epochs_and_seed(tmp_path):
    path = _script(tmp_path, "m = M(epochs=7, seed=1)\n")
    row = check_protocol.check_script(path, "ctgan", "adult")
    assert row["seed_status"] == "DIFF"
    assert row["budget_status"] == "DIFF"
    assert row["expected"] == 150


def test_check_script_resolves_module_constants(tmp_path):
    path = _script(tmp_path, "EPOCHS = 100\nm = M(epochs=EPOCHS, seed=42)\n")
    assert check_protocol.check_script(path, "ctgan", "car")["budget"] == 100


def test_check_script_reports_missing_seed(tmp_path):
    path = _script(tmp_path, "m = M(epochs=100)\n")
    row = check_protocol.check_script(path, "ctgan", "car")
    assert row["seed_status"] == "MISSING"


def test_strict_mode_returns_nonzero_on_findings(monkeypatch):
    rows = [
        {
            "model": "ctgan",
            "dataset": "car",
            "seed": None,
            "seed_status": "MISSING",
            "budget": 100,
            "expected": 100,
            "budget_status": "OK",
        }
    ]
    monkeypatch.setattr(check_protocol, "scan", lambda: rows)
    assert check_protocol.main(["--strict"]) == 1
    assert check_protocol.main([]) == 0
