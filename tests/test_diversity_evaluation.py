"""Unit tests for katabatic/evaluate/diversity/evaluation.py."""

import pandas as pd
import pytest

from katabatic.evaluate.diversity.evaluation import DiversityEvaluation


@pytest.fixture
def mixed_data():
    real = pd.DataFrame(
        {
            "cat": ["a", "b", "c", "d"] * 10,
            "num": list(range(40)),
        }
    )
    synth = real.copy()
    return real, synth


def test_identical_data_scores_near_perfect(mixed_data):
    real, synth = mixed_data
    ev = DiversityEvaluation(
        real, synth, categorical_cols=["cat"], continuous_cols=["num"]
    )
    results = ev.evaluate()

    assert results["category_coverage"]["cat"] == 1.0
    assert results["bin_coverage"]["num"] == 1.0
    assert results["gower_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert results["diversity_score"] == pytest.approx(1.0, abs=1e-3)


def test_synthetic_missing_categories_lowers_coverage():
    real = pd.DataFrame({"cat": ["a", "b", "c", "d"] * 5})
    synth = pd.DataFrame({"cat": ["a"] * 20})
    ev = DiversityEvaluation(real, synth, categorical_cols=["cat"], continuous_cols=[])
    results = ev.evaluate()

    assert results["category_coverage"]["cat"] == pytest.approx(0.25)
    assert results["bin_coverage"] == {}
    # diversity_score also folds in gower_similarity (computed from the same
    # shared 'cat' column), so it's the mean of category coverage + gower sim.
    expected = (0.25 + results["gower_similarity"]) / 2
    assert results["diversity_score"] == pytest.approx(round(expected, 4))


def test_synthetic_missing_column_is_skipped_not_scored():
    real = pd.DataFrame({"cat": ["a", "b"] * 5, "other": ["x", "y"] * 5})
    synth = pd.DataFrame({"cat": ["a", "b"] * 5})  # 'other' missing entirely
    ev = DiversityEvaluation(
        real, synth, categorical_cols=["cat", "other"], continuous_cols=[]
    )
    results = ev.evaluate()

    assert "other" not in results["category_coverage"]
    assert "cat" in results["category_coverage"]


def test_constant_continuous_column_covered_when_synth_hits_value():
    real = pd.DataFrame({"num": [5, 5, 5, 5]})
    synth_hit = pd.DataFrame({"num": [5, 5]})
    synth_miss = pd.DataFrame({"num": [1, 2]})

    ev_hit = DiversityEvaluation(
        real, synth_hit, categorical_cols=[], continuous_cols=["num"]
    )
    ev_miss = DiversityEvaluation(
        real, synth_miss, categorical_cols=[], continuous_cols=["num"]
    )

    assert ev_hit._compute_bin_coverage()["num"] == 1.0
    assert ev_miss._compute_bin_coverage()["num"] == 0.0


def test_no_columns_at_all_scores_zero():
    real = pd.DataFrame({"id": [1, 2, 3]})
    synth = pd.DataFrame({"id": [1, 2, 3]})
    ev = DiversityEvaluation(real, synth, categorical_cols=[], continuous_cols=[])
    results = ev.evaluate()

    assert results["category_coverage"] == {}
    assert results["bin_coverage"] == {}
    assert results["gower_similarity"] is None
    assert results["diversity_score"] == 0.0


def test_auto_detect_column_types_when_not_provided(capsys):
    real = pd.DataFrame({"cat": ["a", "b"] * 5, "num": range(10)})
    synth = real.copy()
    ev = DiversityEvaluation(real, synth)

    captured = capsys.readouterr()
    assert "auto-detect" in captured.out.lower() or "WARNING" in captured.out
    results = ev.evaluate()
    assert results["diversity_score"] is not None


def test_empty_real_category_set_is_skipped():
    real = pd.DataFrame({"cat": [None, None]})
    synth = pd.DataFrame({"cat": ["a", "b"]})
    ev = DiversityEvaluation(real, synth, categorical_cols=["cat"], continuous_cols=[])
    coverage = ev._compute_category_coverage()
    assert coverage == {"avg": 0.0}


def test_empty_real_continuous_values_is_skipped():
    real = pd.DataFrame({"num": [None, None]})
    synth = pd.DataFrame({"num": [1.0, 2.0]})
    ev = DiversityEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    coverage = ev._compute_bin_coverage()
    assert coverage == {"avg": 0.0}


def test_gower_similarity_none_when_no_shared_columns():
    real = pd.DataFrame({"a": [1, 2, 3]})
    synth = pd.DataFrame({"b": [1, 2, 3]})
    ev = DiversityEvaluation(real, synth, categorical_cols=["a"], continuous_cols=[])
    assert ev._compute_gower_similarity() is None


def test_bin_coverage_skips_column_that_is_all_nan_in_real():
    real = pd.DataFrame({"num": [None, None, None]})
    synth = pd.DataFrame({"num": [1.0, 2.0]})
    ev = DiversityEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    assert ev._compute_bin_coverage() == {"avg": 0.0}


def test_gower_similarity_none_with_single_row_sample():
    # Only one row on each side -> no pairs -> upper-triangle distance arrays
    # are empty, hitting the len(real_dists) == 0 guard.
    real = pd.DataFrame({"num": [1.0]})
    synth = pd.DataFrame({"num": [2.0]})
    ev = DiversityEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    assert ev._compute_gower_similarity() is None


def test_gower_similarity_none_when_all_rows_have_nans():
    real = pd.DataFrame({"num": [None, None, None]})
    synth = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    ev = DiversityEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    assert ev._compute_gower_similarity() is None


def test_gower_similarity_respects_sample_size():
    real = pd.DataFrame({"num": list(range(100))})
    synth = pd.DataFrame({"num": list(range(100))})
    ev = DiversityEvaluation(
        real, synth, categorical_cols=[], continuous_cols=["num"], gower_sample_size=10
    )
    sim = ev._compute_gower_similarity()
    assert sim is not None
    assert 0.0 <= sim <= 1.0


def test_print_summary_runs_without_error(mixed_data, capsys):
    real, synth = mixed_data
    ev = DiversityEvaluation(
        real, synth, categorical_cols=["cat"], continuous_cols=["num"]
    )
    ev.evaluate()
    captured = capsys.readouterr()
    assert "Diversity Evaluation" in captured.out
    assert "Category Coverage" in captured.out
    assert "Bin Coverage" in captured.out
    assert "Gower distance distribution similarity" in captured.out
