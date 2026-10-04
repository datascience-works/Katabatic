"""Unit tests for katabatic/evaluate/privacy/evaluation.py."""

import pandas as pd
import pytest

from katabatic.evaluate.privacy.evaluation import PrivacyEvaluation


def test_exact_copy_synthetic_data_floors_at_one_third():
    # Documented behavior (see to_fix.md): exact-copy synthetic data can never
    # score below ~0.333, since near_dup_score is structurally pegged at 1.0
    # once exact duplicates are excluded from that component.
    real = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0]})
    synth = real.copy()
    ev = PrivacyEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    results = ev.evaluate()

    assert results["exact_duplicate_rate"] == 1.0
    assert results["nndr_score"] == 0.0
    assert results["exact_dup_score"] == 0.0
    assert results["near_dup_score"] == 1.0
    assert results["privacy_score"] == pytest.approx(1 / 3, abs=1e-3)


def test_disjoint_data_scores_high_privacy():
    real = pd.DataFrame({"num": [0.0, 1.0, 2.0, 3.0, 4.0]})
    synth = pd.DataFrame({"num": [100.0, 200.0, 300.0]})
    ev = PrivacyEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    results = ev.evaluate()

    assert results["exact_duplicate_rate"] == 0.0
    assert results["near_duplicate_rate"] == 0.0
    assert results["privacy_score"] > 0.5


def test_categorical_and_continuous_mixed_columns():
    real = pd.DataFrame(
        {"cat": ["a", "b", "c", "d", "e"], "num": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    synth = pd.DataFrame(
        {"cat": ["a", "b", "x", "y", "z"], "num": [1.0, 2.5, 30.0, 40.0, 50.0]}
    )
    ev = PrivacyEvaluation(
        real, synth, categorical_cols=["cat"], continuous_cols=["num"]
    )
    results = ev.evaluate()
    assert 0.0 <= results["privacy_score"] <= 1.0


def test_constant_continuous_column_does_not_crash_normalise():
    real = pd.DataFrame({"num": [5.0, 5.0, 5.0]})
    synth = pd.DataFrame({"num": [5.0, 5.0]})
    ev = PrivacyEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    real_out, synth_out, cat_mask = ev._normalise()
    assert (real_out == 0.0).all()
    assert cat_mask.tolist() == [False]


def test_nndr_raises_when_fewer_than_two_real_rows():
    real = pd.DataFrame({"num": [1.0]})
    synth = pd.DataFrame({"num": [1.0, 2.0]})
    ev = PrivacyEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])

    with pytest.raises(ValueError, match="at least 2 real rows"):
        ev.evaluate()


def test_sample_size_subsamples_large_synthetic_data():
    real = pd.DataFrame({"num": list(range(50))}, dtype=float)
    synth = pd.DataFrame({"num": list(range(200))}, dtype=float)
    ev = PrivacyEvaluation(
        real, synth, categorical_cols=[], continuous_cols=["num"], sample_size=20
    )
    results = ev.evaluate()
    assert 0.0 <= results["privacy_score"] <= 1.0


def test_near_duplicate_detection_within_threshold():
    real = pd.DataFrame({"num": [0.0, 100.0]})
    # 0.5 is within near_dup_threshold of the real min-max-normalised range
    synth = pd.DataFrame({"num": [0.3]})
    ev = PrivacyEvaluation(
        real,
        synth,
        categorical_cols=[],
        continuous_cols=["num"],
        near_dup_threshold=0.01,
    )
    results = ev.evaluate()
    # 0.3 normalised is 0.003, within threshold 0.01, and not an exact dup.
    assert results["near_duplicate_rate"] == 1.0
    assert results["exact_duplicate_rate"] == 0.0


def test_no_shared_columns_yields_nan_distance_and_zero_rates():
    real = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    synth = pd.DataFrame({"b": [1.0, 2.0]})
    ev = PrivacyEvaluation(real, synth, categorical_cols=[], continuous_cols=["a"])
    results = ev.evaluate()
    # No shared columns -> d=0 -> distance matrix is NaN (0/0), so neither the
    # exact- nor near-duplicate checks (which compare against 0.0) ever fire.
    assert results["exact_duplicate_rate"] == 0.0
    assert results["near_duplicate_rate"] == 0.0


def test_print_summary_warns_on_high_duplicate_rates(capsys):
    real = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    synth = real.copy()
    ev = PrivacyEvaluation(real, synth, categorical_cols=[], continuous_cols=["num"])
    ev.evaluate()
    captured = capsys.readouterr()
    assert "Privacy Evaluation" in captured.out
    assert "possible memorisation" in captured.out.lower()


@pytest.mark.parametrize("column_types", ["auto", "partial"])
def test_text_columns_are_compared_as_categories(column_types):
    """Text columns must not collapse to NaN distances, which hid every copied row."""
    real = pd.DataFrame(
        {"colour": ["red", "blue", "green", "red"], "size": ["S", "M", "L", "XL"]}
    )
    novel = pd.DataFrame({"colour": ["pink", "teal"], "size": ["XS", "XXL"]})
    synth = pd.concat([real.iloc[:2], novel], ignore_index=True)
    kwargs = {} if column_types == "auto" else {"categorical_cols": ["colour"]}
    results = PrivacyEvaluation(real, synth, **kwargs).evaluate()

    assert results["exact_duplicate_rate"] == 0.5


def test_integer_and_float_categories_compare_by_value():
    """A synthetic 13.0 must match a real 13; mixing them used to raise 'unseen labels'."""
    real = pd.DataFrame({"level": [9, 10, 13, 14]})
    synth = pd.DataFrame({"level": [13.0, 14.0, 12.5, 15.5]})
    results = PrivacyEvaluation(real, synth, categorical_cols=["level"]).evaluate()

    assert results["exact_duplicate_rate"] == 0.5
