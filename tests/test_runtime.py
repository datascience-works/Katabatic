"""Tests for benchmarks/runtime.py and its hooks in benchmarks/runner.py."""

import json
import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import runtime  # noqa: E402


class FakeClock:
    """Returns 0, 10, 20... so durations are exact."""

    def __init__(self):
        self.now = -10.0

    def __call__(self):
        self.now += 10.0
        return self.now


def test_detect_hardware_has_the_expected_keys():
    info = runtime.detect_hardware()
    for key in ("system", "processor", "cpu_count", "ram_total_gb", "accelerator"):
        assert key in info
    assert isinstance(info["accelerator"], str)


def test_phases_add_up_to_the_total():
    timer = runtime.RunTimer("ctgan", "car", clock=FakeClock())  # t0 = 0
    timer.begin("preprocess")  # 10
    timer.end("preprocess")  # 20 -> 10 s
    timer.begin("evaluate")  # 30
    timer.end("evaluate")  # 40 -> 10 s
    timer.finish()  # 50 -> total 50 s
    data = timer.summary()
    assert data["total_seconds"] == 50.0
    assert data["preprocess_seconds"] == 10.0
    assert data["evaluate_seconds"] == 10.0
    assert data["model_seconds"] == 30.0


def test_end_without_begin_is_ignored():
    timer = runtime.RunTimer("ctgan", "car", clock=FakeClock())
    timer.end("evaluate")
    assert timer.summary()["evaluate_seconds"] == 0.0


def test_save_writes_a_json_file(tmp_path):
    timer = runtime.RunTimer("ctgan", "car")
    timer.finish()
    path = timer.save(str(tmp_path / "results"))
    assert path.endswith("ctgan_car_runtime.json")
    data = json.loads(Path(path).read_text())
    assert data["model"] == "ctgan"
    assert data["dataset"] == "car"
    assert "hardware" in data


def test_finish_run_prints_report_and_saves(tmp_path, capsys):
    runtime.start_run("ctgan", "car")
    data = runtime.finish_run(str(tmp_path))
    out = capsys.readouterr().out
    assert "Evaluation Runtime Report" in out
    assert "Computation Hardware Summary" in out
    assert data["model"] == "ctgan"
    assert (tmp_path / "ctgan_car_runtime.json").exists()


def test_finish_run_does_nothing_without_a_timer(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_TIMER", None)
    assert runtime.finish_run(str(tmp_path)) is None


def test_runner_calls_the_runtime_hooks():
    source = (BENCHMARKS / "runner.py").read_text()
    for call in (
        "runtime.start_run(",
        'runtime.current().end("preprocess")',
        'runtime.current().begin("evaluate")',
        "runtime.finish_run(",
    ):
        assert call in source


def test_phase_context_manager_records_extra_phases():
    runtime.start_run("great", "car")
    with runtime.phase("train"):
        pass
    with runtime.phase("generate"):
        pass
    phases = runtime.current().summary()["phases"]
    assert set(phases) == {"train", "generate"}


def test_phase_without_a_timer_does_nothing(monkeypatch):
    monkeypatch.setattr(runtime, "_TIMER", None)
    with runtime.phase("train"):
        pass


def test_finish_run_marks_the_run_completed(tmp_path):
    runtime.start_run("ctgan", "car")
    data = runtime.finish_run(str(tmp_path))
    assert data["status"] == "completed"
    saved = json.loads((tmp_path / "ctgan_car_runtime.json").read_text())
    assert saved["status"] == "completed"


def test_failed_run_is_saved_with_its_error(tmp_path):
    timer = runtime.start_run("ctgan", "car")
    timer.output_dir = str(tmp_path)
    timer.error = "ValueError: bad column"
    runtime._write_unfinished()
    saved = json.loads((tmp_path / "ctgan_car_runtime.json").read_text())
    assert saved["status"] == "failed"
    assert saved["error"] == "ValueError: bad column"


def test_run_that_stopped_early_is_marked_incomplete(tmp_path):
    timer = runtime.start_run("ctgan", "car")
    timer.output_dir = str(tmp_path)
    runtime._write_unfinished()
    saved = json.loads((tmp_path / "ctgan_car_runtime.json").read_text())
    assert saved["status"] == "incomplete"


def test_finished_run_is_not_overwritten(tmp_path):
    runtime.start_run("ctgan", "car")
    runtime.finish_run(str(tmp_path))
    runtime.current().output_dir = str(tmp_path / "other")
    runtime._write_unfinished()
    assert not (tmp_path / "other").exists()


def test_summary_records_memory_git_and_packages():
    data = runtime.RunTimer("ctgan", "car").summary()
    assert "peak_ram_gb" in data and "gpu_memory_gb" in data
    assert set(data["git"]) == {"commit", "dirty"}
    assert "numpy" in data["packages"]
    assert data["status"] == "incomplete"


def test_runner_records_the_output_folder():
    source = (BENCHMARKS / "runner.py").read_text()
    assert 'runtime.current().output_dir = paths["results_dir"]' in source
