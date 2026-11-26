"""Unit tests for katabatic/evaluate/utility/evaluation.py.

Uses n_folds=2 and small datasets throughout to keep the 5-classifier suite fast.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from katabatic.evaluate.report.composite import EvaluationReport
from katabatic.evaluate.utility.evaluation import UtilityEvaluation


def _binary_frame(n=20, seed=0):
    rng = np.random.default_rng(seed)
    y = [i % 2 for i in range(n)]
    return pd.DataFrame(
        {
            "f0": rng.normal(size=n) + np.array(y) * 2,  # separable-ish
            "f1": rng.integers(0, 3, size=n),
            "label": y,
        }
    )


def test_evaluate_numeric_binary_target_no_test_data():
    real = _binary_frame(seed=1)
    synth = _binary_frame(seed=2)
    ev = UtilityEvaluation(real, synth, target_col="label", n_folds=2)
    results = ev.evaluate()

    assert set(results["tstr"].keys()) == {"LR", "DT", "RF", "LinearSVM", "MLP"}
    assert set(results["trtr"].keys()) == set(results["tstr"].keys())
    assert "accuracy" in results["tstr"]["LR"]
    assert 0.0 <= results["utility_score"] <= 1.0


def test_evaluate_string_target_with_held_out_test_data():
    n = 24
    real = pd.DataFrame(
        {
            "size": [["low", "med", "high"][i % 3] for i in range(n)],
            "num": range(n),
            "label": [["yes", "no"][i % 2] for i in range(n)],
        }
    )
    synth = real.copy()
    test = real.copy()
    ev = UtilityEvaluation(real, synth, target_col="label", test_data=test, n_folds=2)
    results = ev.evaluate()

    assert 0.0 <= results["utility_score"] <= 1.0
    # Same data on both sides of a held-out comparison -> small delta.
    assert results["delta"]["LR"]["accuracy"] == pytest.approx(0.0, abs=0.5)


def test_encode_label_encodes_non_numeric_columns_only():
    real = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3], "label": [0, 1, 0]})
    synth = real.copy()
    ev = UtilityEvaluation(real, synth, target_col="label")
    encoded = ev._encode(real[["cat", "num"]])

    assert pd.api.types.is_numeric_dtype(encoded["cat"])
    assert encoded["num"].tolist() == [1, 2, 3]


def test_encode_uses_test_data_categories_when_present():
    real = pd.DataFrame({"cat": ["a", "b"], "label": [0, 1]})
    synth = pd.DataFrame({"cat": ["a", "b"], "label": [0, 1]})
    test = pd.DataFrame({"cat": ["c", "a"], "label": [0, 1]})
    ev = UtilityEvaluation(real, synth, target_col="label", test_data=test)
    encoded = ev._encode(test[["cat"]])
    # Should not raise despite 'c' only appearing in test_data.
    assert len(encoded) == 2


def test_tstr_raises_when_all_folds_single_class():
    real = _binary_frame()
    synth = _binary_frame()
    ev = UtilityEvaluation(real, synth, target_col="label", n_folds=2)

    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)  # single class throughout
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    with pytest.raises(ValueError, match="too class-imbalanced"):
        ev._tstr(
            LogisticRegression(max_iter=1000, random_state=42),
            X,
            y,
            X,
            y,
            cv,
            is_binary=True,
        )


def test_trtr_falls_back_to_cv_fold_without_test_data():
    real = _binary_frame(n=20, seed=3)
    synth = _binary_frame(n=20, seed=4)
    ev = UtilityEvaluation(real, synth, target_col="label", n_folds=2)
    assert ev.test_data is None

    results = ev.evaluate()
    # use_held_out False path exercised; just confirm it completed sanely.
    assert "accuracy" in results["trtr"]["LR"]


def test_score_computes_auc_for_binary_classifier():
    real = _binary_frame(n=30, seed=5)
    synth = real.copy()
    ev = UtilityEvaluation(real, synth, target_col="label")

    X = real[["f0", "f1"]].values
    y = real["label"].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    scores = ev._score(clf, X, y, is_binary=True)

    assert set(scores.keys()) == {
        "accuracy",
        "f1",
        "macro_f1",
        "per_class_recall",
        "auc",
    }


def test_score_skips_auc_for_multiclass():
    real = _binary_frame(n=30, seed=6)
    real["label"] = [i % 3 for i in range(30)]
    ev = UtilityEvaluation(real, real.copy(), target_col="label")

    X = real[["f0", "f1"]].values
    y = real["label"].values
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    scores = ev._score(clf, X, y, is_binary=False)

    assert "auc" not in scores


def test_score_handles_predict_proba_failure_gracefully(capsys):
    real = _binary_frame(n=30, seed=7)
    ev = UtilityEvaluation(real, real.copy(), target_col="label")

    X = real[["f0", "f1"]].values
    y = real["label"].values
    clf = LinearSVC(max_iter=2000, random_state=42)  # no predict_proba
    clf.fit(X, y)
    scores = ev._score(clf, X, y, is_binary=True)

    assert "auc" not in scores
    assert "accuracy" in scores
    captured = capsys.readouterr()
    assert "AUC could not be computed" in captured.out


def test_aggregate_computes_mean_std_and_drops_empty_metrics():
    real = _binary_frame()
    ev = UtilityEvaluation(real, real.copy(), target_col="label")

    fold_metrics = {"accuracy": [0.8, 1.0], "f1": [0.5, 0.7], "auc": []}
    agg = ev._aggregate(fold_metrics)

    assert agg["accuracy"] == {"mean": 0.9, "std": pytest.approx(0.1, abs=1e-9)}
    assert "auc" not in agg


def test_compute_delta_only_uses_common_metrics():
    real = _binary_frame()
    ev = UtilityEvaluation(real, real.copy(), target_col="label")

    tstr = {"LR": {"accuracy": {"mean": 0.7}}, "DT": {"accuracy": {"mean": 0.6}}}
    trtr = {"LR": {"accuracy": {"mean": 0.9}}}  # DT missing entirely
    delta = ev._compute_delta(tstr, trtr)

    assert delta["LR"]["accuracy"] == pytest.approx(0.2)
    assert delta["DT"] == {}


def test_compute_score_empty_deltas_returns_zero():
    real = _binary_frame()
    ev = UtilityEvaluation(real, real.copy(), target_col="label")
    assert ev._compute_score({}) == 0.0


def test_compute_score_clips_to_unit_interval():
    real = _binary_frame()
    ev = UtilityEvaluation(real, real.copy(), target_col="label")

    huge_positive_delta = {"LR": {"accuracy": 5.0}}
    huge_negative_delta = {"LR": {"accuracy": -5.0}}

    assert ev._compute_score(huge_positive_delta) == 0.0
    assert ev._compute_score(huge_negative_delta) == 1.0


def test_print_summary_runs_without_error(capsys):
    real = _binary_frame(seed=8)
    synth = _binary_frame(seed=9)
    ev = UtilityEvaluation(real, synth, target_col="label", n_folds=2)
    ev.evaluate()
    captured = capsys.readouterr()
    assert "Utility Evaluation" in captured.out
    assert "Classifier" in captured.out
    assert "macro_f1" in captured.out
    assert "recall (class=0)" in captured.out


@pytest.mark.parametrize(
    "labels", [[0, 1], [2, 7], [1.0, 2.0], ["majority", "minority"]]
)
def test_macro_f1_and_recall_expose_missed_minority_class(labels):
    y = np.array([labels[0]] * 90 + [labels[1]] * 10)
    X = np.zeros((len(y), 1))
    frame = pd.DataFrame({"feature": X.ravel(), "label": y})
    ev = UtilityEvaluation(frame, frame.copy(), target_col="label")
    clf = DummyClassifier(strategy="most_frequent").fit(X, y)

    scores = ev._score(clf, X, y, is_binary=True)

    assert scores["accuracy"] == pytest.approx(0.9)
    assert scores["f1"] == pytest.approx(0.85263158)
    assert scores["macro_f1"] == pytest.approx(0.47368421)
    assert scores["per_class_recall"] == {str(labels[0]): 1.0, str(labels[1]): 0.0}


@pytest.mark.parametrize("held_out", [False, True])
def test_multiclass_recalls_keep_original_labels_across_protocols_and_json(
    monkeypatch, tmp_path, held_out
):
    monkeypatch.setattr(
        "katabatic.evaluate.utility.evaluation._make_classifiers",
        lambda: {"Dummy": DummyClassifier(strategy="most_frequent")},
    )
    real = pd.DataFrame(
        {"feature": range(24), "label": ["major"] * 12 + ["minor"] * 6 + ["rare"] * 6}
    )
    synth = real.assign(label=["major"] * 12 + ["minor"] * 12)
    test = (
        pd.DataFrame(
            {
                "feature": range(18),
                "label": ["major"] * 9 + ["minor"] * 3 + ["rare"] * 3 + ["unseen"] * 3,
            }
        )
        if held_out
        else None
    )
    ev = UtilityEvaluation(real, synth, "label", test_data=test, n_folds=2)

    results = ev.evaluate()

    expected_classes = {"major", "minor", "rare"} | ({"unseen"} if held_out else set())
    for protocol in ("tstr", "trtr"):
        metrics = results[protocol]["Dummy"]
        assert set(metrics["per_class_recall"]) == expected_classes
        assert metrics["per_class_recall"]["major"] == {"mean": 1.0, "std": 0.0}
        for label in expected_classes - {"major"}:
            assert metrics["per_class_recall"][label] == {"mean": 0.0, "std": 0.0}
        assert metrics["macro_f1"] == {
            "mean": 0.1667 if held_out else 0.2222,
            "std": 0.0,
        }
    assert results["delta"]["Dummy"]["per_class_recall"] == dict.fromkeys(
        expected_classes, 0.0
    )

    report = EvaluationReport({"utility": results})
    report.save(str(tmp_path))
    saved = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert saved["full_results"]["utility"] == results
    assert saved["composite_score"] == results["utility_score"]


def test_aggregate_and_delta_report_per_class_fold_variation():
    real = _binary_frame()
    ev = UtilityEvaluation(real, real.copy(), target_col="label")
    tstr = ev._aggregate(
        {"macro_f1": [0.3, 0.5], "per_class_recall": {"minority": [0.0, 1.0]}}
    )
    trtr = ev._aggregate(
        {"macro_f1": [0.7, 0.9], "per_class_recall": {"minority": [1.0, 1.0]}}
    )

    assert tstr["per_class_recall"]["minority"] == {"mean": 0.5, "std": 0.5}
    assert trtr["per_class_recall"]["minority"] == {"mean": 1.0, "std": 0.0}
    assert ev._compute_delta({"LR": tstr}, {"LR": trtr}) == {
        "LR": {"macro_f1": 0.4, "per_class_recall": {"minority": 0.5}}
    }


def test_utility_score_includes_macro_f1_and_keeps_recall_diagnostic():
    real = _binary_frame()
    ev = UtilityEvaluation(real, real.copy(), target_col="label")
    delta = {"LR": {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.6, "auc": 0.0}}
    assert ev._compute_score(delta) == pytest.approx(0.85)

    delta["LR"]["per_class_recall"] = {f"class_{i}": 1.0 for i in range(20)}
    assert ev._compute_score(delta) == pytest.approx(0.85)


@pytest.mark.parametrize("synthetic_dtype", [int, float, str])
def test_numeric_recall_names_use_original_real_labels(monkeypatch, synthetic_dtype):
    monkeypatch.setattr(
        "katabatic.evaluate.utility.evaluation._make_classifiers",
        lambda: {"Dummy": DummyClassifier(strategy="most_frequent")},
    )
    real = _binary_frame()
    synth = real.assign(label=real["label"].astype(synthetic_dtype))

    results = UtilityEvaluation(real, synth, "label", n_folds=2).evaluate()

    for protocol in ("tstr", "trtr", "delta"):
        assert set(results[protocol]["Dummy"]["per_class_recall"]) == {"0", "1"}


def test_unicode_recall_labels_survive_limited_console_and_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "katabatic.evaluate.utility.evaluation._make_classifiers",
        lambda: {"Dummy": DummyClassifier(strategy="most_frequent")},
    )
    real = pd.DataFrame({"feature": range(12), "label": ["多数"] * 8 + ["少数"] * 4})
    output = io.BytesIO()
    stream = io.TextIOWrapper(output, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", stream)

    results = UtilityEvaluation(real, real.copy(), "label", n_folds=2).evaluate()
    EvaluationReport({"utility": results}).save(str(tmp_path))
    stream.flush()

    assert b"recall (class=\\u" in output.getvalue()
    saved = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert set(
        saved["full_results"]["utility"]["tstr"]["Dummy"]["per_class_recall"]
    ) == {"多数", "少数"}
