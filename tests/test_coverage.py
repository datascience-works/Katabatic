"""Tests for benchmark coverage reporting and leaderboard."""

import json

import pytest

from benchmarks.coverage import (
    FAILED,
    SUCCESS,
    UNSUPPORTED,
    UNTESTED,
    build_coverage_table,
    build_leaderboard,
    get_cell_status,
    get_composite_score,
)


@pytest.fixture
def benchmark_dirs(tmp_path):
    """Create temporary benchmark example and results directories."""
    examples = tmp_path / "examples"
    results = tmp_path / "results"

    for model in ("model_a", "model_b", "model_c", "model_d"):
        model_dir = examples / model
        model_dir.mkdir(parents=True)
        (model_dir / f"run_{model}_adult.py").write_text(
            "# test benchmark script",
            encoding="utf-8",
        )

    return examples, results


def write_json(path, data):
    """Write test JSON data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_success_status(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_evaluation_report.json",
        {"composite_score": 0.85},
    )

    assert get_cell_status("model_a", "adult", examples, results) == SUCCESS


def test_failed_runtime_status(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_runtime.json",
        {"status": "failed", "error": "Training failed"},
    )

    assert get_cell_status("model_a", "adult", examples, results) == FAILED


def test_invalid_evaluation_report_is_failed(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_evaluation_report.json",
        {"composite_score": None},
    )

    assert get_cell_status("model_a", "adult", examples, results) == FAILED


def test_unsupported_without_benchmark_script(benchmark_dirs):
    examples, results = benchmark_dirs

    assert get_cell_status("model_a", "shuttle", examples, results) == UNSUPPORTED


def test_untested_with_script_but_no_results(benchmark_dirs):
    examples, results = benchmark_dirs

    assert get_cell_status("model_a", "adult", examples, results) == UNTESTED


def test_completed_runtime_without_report_is_failed(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_runtime.json",
        {"status": "completed"},
    )

    assert get_cell_status("model_a", "adult", examples, results) == FAILED


@pytest.mark.parametrize(
    "score",
    [None, "0.85", True, float("nan"), float("inf")],
)
def test_invalid_composite_scores_return_none(tmp_path, score):
    report = tmp_path / "report.json"
    write_json(report, {"composite_score": score})

    assert get_composite_score(report) is None


def test_build_coverage_table(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_evaluation_report.json",
        {"composite_score": 0.85},
    )

    table = build_coverage_table(
        examples,
        results,
        datasets=["adult", "shuttle"],
    )

    assert table["model_a"] == {
        "adult": SUCCESS,
        "shuttle": UNSUPPORTED,
    }


def test_leaderboard_ranks_by_average_score(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_evaluation_report.json",
        {"composite_score": 0.80},
    )
    write_json(
        results / "adult" / "model_b" / "model_b_adult_evaluation_report.json",
        {"composite_score": 0.90},
    )
    write_json(
        results / "adult" / "model_c" / "model_c_adult_evaluation_report.json",
        {"composite_score": 0.70},
    )

    leaderboard = build_leaderboard(
        examples,
        results,
        datasets=["adult"],
    )

    assert leaderboard[0]["model"] == "model_b"
    assert leaderboard[0]["average_score"] == pytest.approx(0.90)
    assert leaderboard[1]["model"] == "model_a"
    assert leaderboard[2]["model"] == "model_c"
    assert leaderboard[3]["average_score"] is None


def test_leaderboard_averages_successful_datasets(benchmark_dirs):
    examples, results = benchmark_dirs

    write_json(
        results / "adult" / "model_a" / "model_a_adult_evaluation_report.json",
        {"composite_score": 0.80},
    )
    write_json(
        results / "shuttle" / "model_a" / "model_a_shuttle_evaluation_report.json",
        {"composite_score": 0.60},
    )

    # Add the second dataset's benchmark script.
    (examples / "model_a" / "run_model_a_shuttle.py").write_text(
        "# test benchmark script",
        encoding="utf-8",
    )

    leaderboard = build_leaderboard(
        examples,
        results,
        datasets=["adult", "shuttle"],
    )

    assert leaderboard[0]["average_score"] == pytest.approx(0.70)
    assert leaderboard[0]["successful_runs"] == 2
    assert leaderboard[0]["datasets"] == ["adult", "shuttle"]
