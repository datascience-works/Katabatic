"""Unit tests for katabatic/evaluate/report/composite.py."""

import csv
import json

import numpy as np
import pytest

from katabatic.evaluate.report.composite import (
    DEFAULT_WEIGHTS,
    EvaluationReport,
    warn_if_dimension_sets_differ,
)


def _results(**scores):
    """Build a dimension_results dict keyed like SyntheticEvaluationPipeline does."""
    key_by_dim = {
        "fidelity": "fidelity_score",
        "utility": "utility_score",
        "diversity": "diversity_score",
        "privacy": "privacy_score",
        "consistency": "consistency_score",
        "stability": "stability_score",
    }
    return {dim: {key_by_dim[dim]: val} for dim, val in scores.items()}


def test_composite_all_dimensions_matches_default_weights():
    scores = {
        "fidelity": 0.8,
        "utility": 0.6,
        "diversity": 0.9,
        "privacy": 0.7,
        "consistency": 0.5,
        "stability": 1.0,
    }
    report = EvaluationReport(_results(**scores))

    expected = sum(scores[d] * DEFAULT_WEIGHTS[d] for d in scores)
    assert report.composite_score == pytest.approx(round(expected, 4))
    assert report.dimension_scores == scores


def test_composite_renormalises_over_missing_dimensions():
    # Only fidelity + utility present -> weights renormalised to sum to 1.
    report = EvaluationReport(_results(fidelity=0.8, utility=0.4))

    w_fid = DEFAULT_WEIGHTS["fidelity"]
    w_util = DEFAULT_WEIGHTS["utility"]
    expected = (0.8 * w_fid + 0.4 * w_util) / (w_fid + w_util)
    assert report.composite_score == pytest.approx(round(expected, 4))


def test_composite_ignores_dimension_missing_its_score_key():
    # A dimension present in dimension_results but without its score key
    # (e.g. an error stub with a different shape) contributes nothing.
    report = EvaluationReport({"fidelity": {"fidelity_score": 0.5}, "utility": {}})

    assert report.dimension_scores == {"fidelity": 0.5}
    assert report.composite_score == pytest.approx(0.5)


def test_composite_zero_when_no_scores_present():
    report = EvaluationReport({})
    assert report.dimension_scores == {}
    assert report.composite_score == 0.0


def test_custom_weights_override_defaults_partially():
    report = EvaluationReport(
        _results(fidelity=1.0, utility=0.0), weights={"fidelity": 0.9, "utility": 0.1}
    )
    assert report.composite_score == pytest.approx(0.9)


def test_save_writes_json_and_csv(tmp_path):
    report = EvaluationReport(_results(fidelity=0.8, utility=0.6))
    report.save(str(tmp_path), prefix="model_ds_")

    json_path = tmp_path / "model_ds_evaluation_report.json"
    csv_path = tmp_path / "model_ds_evaluation_summary.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["composite_score"] == report.composite_score
    assert payload["dimension_scores"] == {"fidelity": 0.8, "utility": 0.6}
    assert pytest.approx(sum(payload["weights_used"].values())) == 1.0
    assert payload["successful_dimensions"] == ["fidelity", "utility"]
    assert payload["successful_dimension_count"] == 2
    assert payload["total_dimensions"] == 6
    assert payload["failed_dimensions"] == []
    assert payload["skipped_dimensions"] == [
        "diversity",
        "privacy",
        "consistency",
        "stability",
    ]

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["Dimension", "Score", "Weight", "Weighted Score"]
    last_row = rows[-1]
    assert last_row[0] == "composite_score"
    assert float(last_row[1]) == report.composite_score


def test_save_creates_output_dir(tmp_path):
    output_dir = tmp_path / "nested" / "reports"
    report = EvaluationReport(_results(fidelity=0.5))
    report.save(str(output_dir))

    assert (output_dir / "evaluation_report.json").exists()


def test_serialisable_converts_numpy_types():
    report = EvaluationReport(_results(fidelity=0.5))
    payload = {
        "a": np.int64(3),
        "b": np.float64(1.5),
        "c": np.array([1, 2, 3]),
        "d": [np.float32(2.0), {"e": np.int32(1)}],
    }
    out = report._serialisable(payload)

    assert out == {"a": 3, "b": 1.5, "c": [1, 2, 3], "d": [2.0, {"e": 1}]}
    assert isinstance(out["a"], int)
    assert isinstance(out["b"], float)


def test_print_summary_runs_without_error(capsys):
    report = EvaluationReport(_results(fidelity=0.8, utility=0.4))
    report.print_summary()
    captured = capsys.readouterr()
    assert "COMPOSITE" in captured.out
    assert "fidelity" in captured.out


def test_print_summary_handles_zero_total_weight(capsys):
    # dimension present with a weight of 0 -> total_weight branch (norm_w=0) exercised.
    report = EvaluationReport(_results(fidelity=0.8), weights={"fidelity": 0.0})
    report.print_summary()
    captured = capsys.readouterr()
    assert "fidelity" in captured.out


def test_warns_when_models_have_different_dimension_sets(capsys):
    reports = {
        "Model A": EvaluationReport(_results(fidelity=0.8, utility=0.6)),
        "Model B": EvaluationReport(_results(fidelity=0.7, privacy=0.9)),
    }

    result = warn_if_dimension_sets_differ(reports)

    captured = capsys.readouterr()
    assert result is True
    assert "WARNING" in captured.out
    assert "Model B" in captured.out
    assert "utility" in captured.out
    assert "privacy" in captured.out


def test_no_warning_when_models_have_same_dimension_sets(capsys):
    reports = {
        "Model A": EvaluationReport(_results(fidelity=0.8, utility=0.6)),
        "Model B": EvaluationReport(_results(fidelity=0.7, utility=0.9)),
    }

    result = warn_if_dimension_sets_differ(reports)

    captured = capsys.readouterr()
    assert result is False
    assert "WARNING" not in captured.out


def test_no_warning_when_only_one_model_report(capsys):
    reports = {"Model A": EvaluationReport(_results(fidelity=0.8))}

    result = warn_if_dimension_sets_differ(reports)

    captured = capsys.readouterr()
    assert result is False
    assert captured.out == ""
