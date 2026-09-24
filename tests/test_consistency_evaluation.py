"""Unit tests for katabatic/evaluate/consistency/evaluation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from katabatic.evaluate.consistency.evaluation import ConsistencyEvaluation


def _frame(n=20, seed=0):
    rng = np.random.default_rng(seed)
    y = [i % 2 for i in range(n)]
    return pd.DataFrame(
        {
            "num": rng.normal(size=n) + np.array(y) * 2,
            "cat": [["x", "y"][i % 2] for i in range(n)],
            "label": y,
        }
    )


def test_evaluate_full_pipeline_with_constraints():
    real = _frame(seed=1)
    synth = _frame(seed=2)
    ev = ConsistencyEvaluation(
        real,
        synth,
        target_col="label",
        constraints={"num": (-10, 10)},
        n_folds=2,
    )
    results = ev.evaluate()

    assert 0.0 <= results["discriminator_accuracy"] <= 1.0
    assert 0.0 <= results["discriminator_score"] <= 1.0
    assert results["constraint_score"] is not None
    assert results["feature_importance_spearman"] is not None
    assert 0.0 <= results["consistency_score"] <= 1.0


def test_evaluate_without_constraints_averages_two_components():
    real = _frame(seed=3)
    synth = _frame(seed=4)
    ev = ConsistencyEvaluation(real, synth, target_col="label", n_folds=2)
    results = ev.evaluate()

    assert results["constraint_violations"] == {}
    assert results["constraint_score"] is None
    expected = round(
        (results["discriminator_score"] + results["feature_importance_spearman"]) / 2,
        4,
    )
    assert results["consistency_score"] == expected


def test_discriminator_score_inverts_low_accuracy_correctly():
    real = _frame(seed=5)
    synth = _frame(seed=6)
    _ = ConsistencyEvaluation(real, synth, target_col="label", n_folds=2)

    # accuracy=0.1 means the RF distinguishes real/synth but with inverted
    # labels -- still fully detectable, score should be low (~0.2), not 1.0.
    assert round(max(0.0, 1.0 - abs(0.1 - 0.5) * 2), 4) == 0.2
    assert round(max(0.0, 1.0 - abs(0.9 - 0.5) * 2), 4) == 0.2
    assert round(max(0.0, 1.0 - abs(0.5 - 0.5) * 2), 4) == 1.0


def test_constraint_violations_detects_out_of_bounds_rows():
    real = _frame(seed=7)
    synth = real.copy()
    synth.loc[0, "num"] = 1000.0  # violates upper bound
    synth.loc[1, "num"] = -1000.0  # violates lower bound
    ev = ConsistencyEvaluation(
        real, synth, target_col="label", constraints={"num": (-10, 10)}, n_folds=2
    )
    violations = ev._compute_constraint_violations()

    assert violations["num [-10, 10]"] == pytest.approx(2 / len(synth))
    assert violations["overall_violation_rate"] == pytest.approx(2 / len(synth))


def test_constraint_violations_skips_missing_column():
    real = _frame(seed=8)
    synth = real.copy()
    ev = ConsistencyEvaluation(
        real, synth, target_col="label", constraints={"missing_col": (0, 1)}
    )
    violations = ev._compute_constraint_violations()
    assert violations == {"overall_violation_rate": 0.0}


def test_constraint_violations_one_sided_bounds():
    real = _frame(seed=9)
    synth = real.copy()
    synth.loc[0, "num"] = -1000.0
    ev = ConsistencyEvaluation(
        real, synth, target_col="label", constraints={"num": (0.0, None)}
    )
    violations = ev._compute_constraint_violations()
    assert violations["overall_violation_rate"] > 0.0

    synth2 = real.copy()
    synth2.loc[0, "num"] = 1000.0
    ev2 = ConsistencyEvaluation(
        real, synth2, target_col="label", constraints={"num": (None, 0.0)}
    )
    violations2 = ev2._compute_constraint_violations()
    assert violations2["overall_violation_rate"] > 0.0


def test_feature_importance_spearman_none_when_target_missing():
    real = _frame(seed=10)
    synth = real.copy()
    ev = ConsistencyEvaluation(real, synth, target_col="not_a_column")
    assert ev._compute_feature_importance_spearman() is None


def test_feature_importance_spearman_none_when_no_feature_columns():
    real = pd.DataFrame({"label": [0, 1, 0, 1]})
    synth = real.copy()
    ev = ConsistencyEvaluation(real, synth, target_col="label")
    assert ev._compute_feature_importance_spearman() is None


def test_feature_importance_spearman_clips_negative_correlation_to_zero(monkeypatch):
    real = _frame(seed=11)
    synth = real.copy()
    ev = ConsistencyEvaluation(real, synth, target_col="label")

    monkeypatch.setattr(
        "katabatic.evaluate.consistency.evaluation.spearmanr",
        lambda a, b: (-0.7, 0.01),
    )
    assert ev._compute_feature_importance_spearman() == 0.0


def test_encode_dataframe_with_and_without_reference():
    real = pd.DataFrame({"cat": ["a", "b", "c"]})
    other = pd.DataFrame({"cat": ["d", "e"]})
    ev = ConsistencyEvaluation(real, real.copy(), target_col="cat")

    encoded_alone = ev._encode_dataframe(real.copy())
    assert pd.api.types.is_numeric_dtype(encoded_alone["cat"])

    encoded_with_ref = ev._encode_dataframe(real.copy(), reference_df=other)
    assert pd.api.types.is_numeric_dtype(encoded_with_ref["cat"])


def test_print_summary_red_flags_high_discriminator_accuracy(capsys):
    real = _frame(seed=12)
    synth = pd.DataFrame(
        {
            "num": [1000.0] * len(real),  # trivially distinguishable
            "cat": ["z"] * len(real),
            "label": real["label"],
        }
    )
    ev = ConsistencyEvaluation(real, synth, target_col="label", n_folds=2)
    results = ev.evaluate()

    captured = capsys.readouterr()
    assert "Consistency Evaluation" in captured.out
    if results["discriminator_accuracy"] > 0.70:
        assert "RED FLAG" in captured.out


def test_print_summary_no_constraints_message(capsys):
    real = _frame(seed=13)
    synth = real.copy()
    ev = ConsistencyEvaluation(real, synth, target_col="label", n_folds=2)
    ev.evaluate()
    captured = capsys.readouterr()
    assert "No constraints defined" in captured.out
