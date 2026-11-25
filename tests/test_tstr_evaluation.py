from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import katabatic.evaluate.tstr.evaluation as tstr_module
from katabatic.artifacts.local import LocalArtifactStore
from katabatic.artifacts.refs import DatasetRef, ModelRef
from katabatic.evaluate.tstr.evaluation import TSTREvaluation, XGBClassifier


@pytest.fixture(autouse=True)
def _isolate_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _write_split(tmp_path, x_synth, y_synth, x_test, y_test):
    synthetic_dir = tmp_path / "synthetic"
    real_test_dir = tmp_path / "real_test"
    synthetic_dir.mkdir()
    real_test_dir.mkdir()

    x_synth.to_csv(synthetic_dir / "x_synth.csv", index=False)
    pd.DataFrame({"y": y_synth}).to_csv(synthetic_dir / "y_synth.csv", index=False)
    x_test.to_csv(real_test_dir / "x_test.csv", index=False)
    pd.DataFrame({"y": y_test}).to_csv(real_test_dir / "y_test.csv", index=False)

    return str(synthetic_dir), str(real_test_dir)


@pytest.mark.parametrize("n", [40])
def test_tstr_evaluation_numeric_features(tmp_path, n):
    """Baseline test for numeric features."""
    x = pd.DataFrame({"f0": range(n), "f1": [i % 5 for i in range(n)]})
    y = [i % 2 for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    assert "LR" in results
    assert "Accuracy" in results["LR"]


def test_tstr_evaluation_categorical_features(tmp_path):
    """Regression test on categorical features."""
    categories_a = ["low", "med", "high"]
    categories_b = ["small", "big"]
    n = 40
    x = pd.DataFrame(
        {
            "size": [categories_a[i % 3] for i in range(n)],
            "price": [categories_b[i % 2] for i in range(n)],
        }
    )
    y = [i % 2 for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    assert "LR" in results
    assert "Accuracy" in results["LR"]


def test_tstr_evaluation_mixed_features(tmp_path):
    """Mixed numeric + categorical columns (e.g. the shipped adult.csv)."""
    n = 40
    x = pd.DataFrame(
        {
            "age": range(n),
            "workclass": [["Private", "Self-emp"][i % 2] for i in range(n)],
        }
    )
    y = [i % 2 for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    assert "LR" in results
    assert "Accuracy" in results["LR"]


@pytest.mark.parametrize(
    "labels",
    [["no", "yes"], ["acc", "good", "unacc", "vgood"]],
    ids=["binary", "multiclass"],
)
def test_tstr_evaluation_string_labels(tmp_path, labels):
    """Text labels (e.g. car's acc/unacc) must work for every classifier, XGBoost included."""
    n = 40
    x = pd.DataFrame({"f0": range(n), "f1": [i % 5 for i in range(n)]})
    y = [labels[i % len(labels)] for i in range(n)]

    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y, x, y)
    results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    expected = {"LR", "MLP", "RF"} | (
        {"XGBoost"} if XGBClassifier is not None else set()
    )
    assert set(results) == expected
    assert all(0.0 <= m["Accuracy"] <= 1.0 for m in results.values())
    assert all(0.0 <= m["Macro F1 Score"] <= 1.0 for m in results.values())
    assert all(set(m["Per-Class Recall"]) == set(labels) for m in results.values())
    if len(labels) == 2:
        assert all("AUC" in m for m in results.values())


@pytest.fixture
def majority_classifiers(monkeypatch):
    """Use deterministic predictions to expose missed minority classes."""
    for name in ("LogisticRegression", "MLPClassifier", "RandomForestClassifier"):
        monkeypatch.setattr(
            tstr_module, name, lambda: DummyClassifier(strategy="most_frequent")
        )
    monkeypatch.setattr(tstr_module, "XGBClassifier", None)


@pytest.mark.parametrize(
    "y_train,y_test,expected_macro,expected_recall",
    [
        ([0] * 9 + [1], [0] * 9 + [1], 9 / 19, {"0": 1.0, "1": 0.0}),
        ([0.0] * 9 + [1.0], [0] * 9 + [1], 9 / 19, {"0": 1.0, "1": 0.0}),
        ([2] * 9 + [7], [2] * 9 + [7], 9 / 19, {"2": 1.0, "7": 0.0}),
        (
            ["majority"] * 8 + ["minority", "rare"],
            ["majority"] * 8 + ["minority", "rare"],
            8 / 27,
            {"majority": 1.0, "minority": 0.0, "rare": 0.0},
        ),
        (
            ["majority"] * 8 + ["minority"] * 2,
            ["majority"] * 8 + ["minority", "missing"],
            8 / 27,
            {"majority": 1.0, "minority": 0.0, "missing": 0.0},
        ),
    ],
    ids=[
        "binary",
        "synthetic-float-real-int",
        "noncontiguous-numeric",
        "multiclass",
        "missing-synthetic-class",
    ],
)
def test_tstr_reports_macro_f1_and_minority_recall(
    tmp_path, majority_classifiers, y_train, y_test, expected_macro, expected_recall
):
    x = pd.DataFrame({"feature": range(10)})
    synthetic_dir, real_test_dir = _write_split(tmp_path, x, y_train, x, y_test)

    with pytest.warns(UserWarning, match="xgboost is not installed"):
        results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()

    for metrics in results.values():
        assert metrics["Macro F1 Score"] == pytest.approx(expected_macro)
        assert metrics["Per-Class Recall"] == expected_recall
        assert metrics["F1 Score"] > metrics["Macro F1 Score"]

    csv_path = Path("Results") / tmp_path.name / "synthetic_tstr.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lr_metrics = {
        row["Metric"]: float(row["Value"]) for row in rows if row["Model"] == "LR"
    }
    assert lr_metrics["Macro F1 Score"] == round(expected_macro, 4)
    for label, recall in expected_recall.items():
        assert lr_metrics[f"Recall (class={label})"] == recall


def test_tstr_artifacts_preserve_class_recall_and_macro_f1(
    tmp_path, majority_classifiers
):
    store = LocalArtifactStore(tmp_path / "artifacts")
    dataset_ref = DatasetRef("imbalanced", "v1")
    model_ref = ModelRef("dummy", "imbalanced", "v1", "train1")
    synthetic_path = store.open_path(model_ref.synthetic_relpath)
    real_test_path = store.open_path(dataset_ref.test_relpath)
    synthetic_path.mkdir(parents=True)
    real_test_path.mkdir(parents=True)
    x = pd.DataFrame({"feature": range(10)})
    labels = pd.DataFrame({"target": ["majority"] * 9 + ["minority"]})
    x.to_csv(synthetic_path / "x_synth.csv", index=False)
    labels.to_csv(synthetic_path / "y_synth.csv", index=False)
    x.to_csv(real_test_path / "x_test.csv", index=False)
    labels.to_csv(real_test_path / "y_test.csv", index=False)
    evaluator, eval_ref = TSTREvaluation.from_artifact(store, model_ref, dataset_ref)

    with pytest.warns(UserWarning, match="xgboost is not installed"):
        results = evaluator.evaluate()

    saved = store.load_json(eval_ref.metrics_relpath)
    assert saved == results
    assert saved["LR"]["Macro F1 Score"] == pytest.approx(9 / 19)
    assert saved["LR"]["Per-Class Recall"] == {"majority": 1.0, "minority": 0.0}
    with store.open_path(eval_ref.report_relpath).open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    lr_metrics = {
        row["Metric"]: float(row["Value"]) for row in rows if row["Model"] == "LR"
    }
    assert lr_metrics["Macro F1 Score"] == 0.4737
    assert lr_metrics["Recall (class=minority)"] == 0.0


def test_tstr_unicode_recall_csv_with_legacy_windows_console(
    tmp_path, majority_classifiers, monkeypatch
):
    x = pd.DataFrame({"feature": range(10)})
    labels = ["主流"] * 9 + ["少数"]
    synthetic_dir, real_test_dir = _write_split(tmp_path, x, labels, x, labels)
    console_bytes = io.BytesIO()
    with io.TextIOWrapper(console_bytes, encoding="cp1252") as console:
        with monkeypatch.context() as console_patch:
            console_patch.setattr(tstr_module.sys, "stdout", console)
            with pytest.warns(UserWarning, match="xgboost is not installed"):
                results = TSTREvaluation(synthetic_dir, real_test_dir).evaluate()
            console.flush()
            console_output = console_bytes.getvalue().decode("cp1252")

    assert results["LR"]["Per-Class Recall"] == {"主流": 1.0, "少数": 0.0}
    assert "Recall (class=\\u5c11\\u6570): 0.0000" in console_output
    csv_path = Path("Results") / tmp_path.name / "synthetic_tstr.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert any(
        row["Model"] == "LR"
        and row["Metric"] == "Recall (class=少数)"
        and float(row["Value"]) == 0.0
        for row in rows
    )
