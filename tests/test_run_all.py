"""Tests for benchmarks/run_all.py, using small fake benchmark scripts."""

import json
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import run_all  # noqa: E402

FAKE_SCRIPT = """
import json, os, sys, time
results = {results!r}
model, dataset = {model!r}, {dataset!r}
folder = os.path.join(results, dataset, model)
os.makedirs(folder, exist_ok=True)
mode = {mode!r}
if mode == "sleep":
    time.sleep(30)
if mode == "crash":
    raise ValueError("column 'age' not found")
status = "failed" if mode == "runtime_failed" else "completed"
with open(os.path.join(folder, f"{{model}}_{{dataset}}_runtime.json"), "w") as f:
    json.dump({{"status": status, "error": "boom" if status == "failed" else None,
               "peak_ram_gb": 0.5}}, f)
if mode == "ok":
    with open(os.path.join(folder, f"{{model}}_{{dataset}}_evaluation_report.json"), "w") as f:
        json.dump({{"composite_score": 0.7}}, f)
"""


@pytest.fixture
def dirs(tmp_path):
    examples = tmp_path / "examples"
    results = tmp_path / "results"
    for model in ("alpha", "beta"):
        (examples / model).mkdir(parents=True)
    return examples, results


def add_script(examples, results, model, dataset, mode="ok"):
    path = examples / model / f"run_{model}_{dataset}.py"
    path.write_text(
        FAKE_SCRIPT.format(
            results=str(results), model=model, dataset=dataset, mode=mode
        )
    )
    return path


def test_split_names():
    assert run_all.split_names("a, B ,c") == ["a", "b", "c"]
    assert run_all.split_names("") is None
    assert run_all.split_names(None) is None


def test_build_jobs_orders_small_datasets_first(dirs):
    examples, results = dirs
    for model in ("alpha", "beta"):
        for dataset in ("shuttle", "car", "adult"):
            add_script(examples, results, model, dataset)

    jobs = run_all.build_jobs(examples_dir=examples)

    assert [d for _, d in jobs] == [
        "car",
        "car",
        "adult",
        "adult",
        "shuttle",
        "shuttle",
    ]


def test_build_jobs_filters(dirs):
    examples, results = dirs
    for model in ("alpha", "beta"):
        for dataset in ("car", "adult"):
            add_script(examples, results, model, dataset)

    assert run_all.build_jobs(["alpha"], ["car"], examples_dir=examples) == [
        ("alpha", "car")
    ]
    assert run_all.build_jobs(
        exclude=["alpha"], datasets=["car"], examples_dir=examples
    ) == [("beta", "car")]


def test_build_jobs_skips_missing_scripts(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car")

    assert run_all.build_jobs(examples_dir=examples) == [("alpha", "car")]


def test_build_jobs_rejects_unknown_names(dirs):
    examples, _ = dirs

    with pytest.raises(ValueError, match="unknown model"):
        run_all.build_jobs(["nope"], examples_dir=examples)
    with pytest.raises(ValueError, match="unknown dataset"):
        run_all.build_jobs(datasets=["nope"], examples_dir=examples)


def test_should_run_skips_finished_jobs(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car")

    assert run_all.should_run(
        "alpha", "car", examples_dir=examples, results_dir=results
    )[0]

    run_all.run_job("alpha", "car", 30, examples, results)

    run, reason = run_all.should_run(
        "alpha", "car", examples_dir=examples, results_dir=results
    )
    assert not run
    assert "valid report" in reason
    assert run_all.should_run(
        "alpha", "car", force=True, examples_dir=examples, results_dir=results
    )[0]


def test_failed_job_needs_retry_flag(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car", mode="runtime_failed")
    run_all.run_job("alpha", "car", 30, examples, results)

    assert not run_all.should_run(
        "alpha", "car", examples_dir=examples, results_dir=results
    )[0]
    assert run_all.should_run(
        "alpha", "car", retry_failed=True, examples_dir=examples, results_dir=results
    )[0]


def test_run_job_completed(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car")

    record = run_all.run_job("alpha", "car", 30, examples, results)

    assert record["status"] == run_all.DONE
    assert record["error"] is None
    assert record["peak_ram_gb"] == 0.5
    assert Path(record["log"]).exists()


def test_run_job_crash_is_recorded_not_raised(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car", mode="crash")

    record = run_all.run_job("alpha", "car", 30, examples, results)

    assert record["status"] == run_all.FAILED
    assert "column 'age' not found" in record["error"]


def test_run_job_uses_runtime_failure_status(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car", mode="runtime_failed")

    record = run_all.run_job("alpha", "car", 30, examples, results)

    assert record["status"] == run_all.FAILED
    assert record["error"] == "boom"


def test_run_job_timeout_is_not_a_failure(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car", mode="sleep")

    record = run_all.run_job("alpha", "car", 1, examples, results)

    assert record["status"] == run_all.TIMEOUT
    assert "known limitation" in record["error"]


def test_stale_runtime_file_is_ignored(dirs):
    examples, results = dirs
    add_script(examples, results, "alpha", "car", mode="crash")
    folder = results / "car" / "alpha"
    folder.mkdir(parents=True)
    old = folder / "alpha_car_runtime.json"
    old.write_text(json.dumps({"status": "completed", "error": None}))
    import os

    os.utime(old, (1, 1))

    record = run_all.run_job("alpha", "car", 30, examples, results)

    assert record["status"] == run_all.FAILED


def test_summary_lists_problems():
    records = [
        {"model": "a", "dataset": "car", "status": "completed", "seconds": 60},
        {
            "model": "b",
            "dataset": "car",
            "status": "failed",
            "seconds": 6,
            "error": "Traceback\nValueError: bad | column",
        },
        {
            "model": "c",
            "dataset": "adult",
            "status": "timeout",
            "seconds": 7200,
            "error": "stopped after 7200 s",
        },
    ]

    text = run_all.render_summary(records)

    assert "3 jobs" in text
    assert "| failed | 1 |" in text
    assert "ValueError: bad / column" in text
    assert "| c | adult | timeout | 120.0 |" in text


def test_merge_records_keeps_old_and_replaces_rerun():
    old = [
        {"model": "a", "dataset": "car", "status": "failed"},
        {"model": "b", "dataset": "car", "status": "completed"},
    ]
    new = [
        {"model": "a", "dataset": "car", "status": "completed"},
        {"model": "b", "dataset": "car", "status": "skipped"},
    ]

    merged = {
        (r["model"], r["dataset"]): r["status"] for r in run_all.merge_records(old, new)
    }

    assert merged == {("a", "car"): "completed", ("b", "car"): "completed"}


def test_every_real_script_is_found():
    jobs = run_all.build_jobs()

    assert len(jobs) == 110
