"""Tests for legacy TrainTestSplitPipeline wiring (split → train paths → evaluation)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from katabatic.artifacts import LocalArtifactStore
from katabatic.evaluate.fidelity.evaluation import FidelityEvaluation
from katabatic.models.base_model import Model
from katabatic.pipeline.train_test_split.pipeline import (
    TrainTestSplitPipeline,
    _default_legacy_synthetic_dir,
)
from katabatic.utils.train_test_consistency import sanity_check_train_test


class StubEval:
    """Avoid importing full TSTREvaluation (pulls xgboost) for pipeline wiring tests."""

    def __init__(self, synthetic_dir=None, real_test_dir=None, **kwargs):
        self.synthetic_dir = synthetic_dir
        self.real_test_dir = real_test_dir

    def evaluate(self):
        assert self.synthetic_dir is not None and self.real_test_dir is not None
        assert Path(self.synthetic_dir, "x_synth.csv").is_file()
        assert Path(self.synthetic_dir, "y_synth.csv").is_file()
        assert Path(self.real_test_dir, "x_test.csv").is_file()
        assert Path(self.real_test_dir, "y_test.csv").is_file()
        return {"stub": True}


class StubGen(Model):
    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return []

    def train(self, data_dir, *args, **kwargs):
        synthetic_dir = kwargs["synthetic_dir"]
        Path(synthetic_dir).mkdir(parents=True, exist_ok=True)

        x_train = pd.read_csv(Path(data_dir) / "x_train.csv")
        y_train = pd.read_csv(Path(data_dir) / "y_train.csv")
        y_col = y_train.columns[0]
        y_series = y_train.iloc[:, 0]

        x_train.to_csv(Path(synthetic_dir) / "x_synth.csv", index=False)
        pd.DataFrame({y_col: y_series}).to_csv(
            Path(synthetic_dir) / "y_synth.csv", index=False, header=True
        )

    def evaluate(self, *args, **kwargs):
        return 0.0

    def sample(self, *args, **kwargs):
        return pd.DataFrame()


def test_default_legacy_synthetic_dir():
    m = StubGen()
    d = _default_legacy_synthetic_dir("/tmp/data/myset", m)
    parts = Path(d).parts
    assert "synthetic" in parts
    assert "myset" in parts
    assert parts[-1] == "stubgen"


def test_legacy_pipeline_defaults_paths_and_last_model(tmp_path):
    df = pd.DataFrame({"f0": range(30), "y": [0, 1] * 15})
    inp = tmp_path / "in.csv"
    df.to_csv(inp, index=False)
    out_dir = tmp_path / "split"

    pipe = TrainTestSplitPipeline(
        model=StubGen,
        evaluations=[StubEval],
        override_evaluations=True,
    )
    res = pipe.run(input_csv=str(inp), output_dir=str(out_dir), test_size=0.2, seed=0)

    assert res["message"]
    assert Path(res["output_dir"]) == out_dir
    assert Path(res["real_test_dir"]) == out_dir
    assert "synthetic" in str(res["synthetic_dir"]).replace("\\", "/")
    assert res["tstr_results"] == {"stub": True}
    assert isinstance(pipe.last_model, StubGen)


def test_legacy_pipeline_presplit(tmp_path):
    tr = tmp_path / "train.csv"
    te = tmp_path / "test.csv"
    pd.DataFrame({"f0": range(20), "y": [0, 1] * 10}).to_csv(tr, index=False)
    pd.DataFrame({"f0": range(20, 30), "y": [0, 1] * 5}).to_csv(te, index=False)
    out_dir = tmp_path / "split"

    pipe = TrainTestSplitPipeline(
        model=StubGen,
        evaluations=[StubEval],
        override_evaluations=True,
    )
    res = pipe.run(train_csv=str(tr), test_csv=str(te), output_dir=str(out_dir))

    assert Path(out_dir, "x_train.csv").is_file()
    assert Path(out_dir, "x_test.csv").is_file()
    assert res["tstr_results"] == {"stub": True}


def test_sanity_check_column_mismatch_raises():
    a = pd.DataFrame({"x": [1], "y": [0]})
    b = pd.DataFrame({"z": [1], "y": [0]})
    with pytest.raises(ValueError, match="Column mismatch"):
        sanity_check_train_test(a, b)


def test_artifact_pipeline_eval_paths_match_model_run(tmp_path):
    df = pd.DataFrame({"f0": range(40), "y": [0, 1, 2, 3] * 10})
    inp = tmp_path / "in.csv"
    df.to_csv(inp, index=False)
    root = tmp_path / "artifacts"
    store = LocalArtifactStore(root)

    pipe = TrainTestSplitPipeline(model=StubGen)
    res = pipe.run(
        input_csv=str(inp),
        dataset_name="benchds",
        artifact_store=store,
        model_name="stubgen",
        require_registered_dataset=True,
    )
    mr = res["model_ref"]
    assert re.match(r"^models/stubgen_benchds_train-\d{8}-\d{6}$", mr.root_relpath), (
        mr.root_relpath
    )
    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert ev.root_relpath == mr.root_relpath.replace("models/", "evaluations/", 1)
    assert ev.report_relpath.endswith("/tstr_report.csv")
    assert ev.metrics_relpath.endswith("/tstr_metrics.json")
    assert Path(store.open_path(ev.report_relpath)).is_file()
    reg_path = store.open_path("registry/datasets.json")
    assert reg_path.is_file()


def test_artifact_presplit_auto_registry(tmp_path):
    tr = tmp_path / "tr.csv"
    te = tmp_path / "te.csv"
    pd.DataFrame({"f0": range(24), "y": [0, 1] * 12}).to_csv(tr, index=False)
    pd.DataFrame({"f0": range(24, 30), "y": [0, 1] * 3}).to_csv(te, index=False)
    root = tmp_path / "artifacts"
    store = LocalArtifactStore(root)

    pipe = TrainTestSplitPipeline(
        model=StubGen,
        evaluations=[StubEval],
        override_evaluations=True,
    )
    res = pipe.run(
        train_csv=str(tr),
        test_csv=str(te),
        dataset_name="pdsh",
        artifact_store=store,
        model_name="stubgen",
    )
    assert res["dataset_ref"].root_relpath.startswith("datasets/pdsh/")
    # StubEval has no from_artifact; pipeline returns no EvaluationRef for it.
    assert res["evaluation_refs"][0] is None


@pytest.mark.skip(
    reason="FidelityEvaluation uuses df-based contructor which is incompatible "
    "with TrainTestSplitPipeline's directory-based eval interface."
)
def test_artifact_fidelity_evaluation_smoke(tmp_path):
    df = pd.DataFrame({"f0": range(40), "y": [0, 1] * 20})
    inp = tmp_path / "in.csv"
    df.to_csv(inp, index=False)
    store = LocalArtifactStore(tmp_path / "artifacts")

    pipe = TrainTestSplitPipeline(
        model=StubGen,
        evaluations=[FidelityEvaluation],
        override_evaluations=True,
    )
    res = pipe.run(
        input_csv=str(inp),
        dataset_name="fidsh",
        artifact_store=store,
        model_name="stubgen",
    )
    ev = res["evaluation_refs"][0]
    assert ev is not None
    assert ev.report_relpath.endswith("/fidelity_report.csv")
    metrics = store.load_json(ev.metrics_relpath)
    assert "summary" in metrics
    assert "mean_jsd" in metrics["summary"]
    assert "dcr_mean" in metrics["summary"]
