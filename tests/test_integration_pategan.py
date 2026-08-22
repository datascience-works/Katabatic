"""Integration tests for PATE-GAN (import/registry; full train is manual)."""

from __future__ import annotations

import pandas as pd
import pytest

from katabatic.artifacts import LocalArtifactStore
from katabatic.artifacts.refs import ModelRef
from katabatic.models.pategan.models import PATEGAN
from katabatic.models.pategan.utils import DataTransformer, save_metadata
from katabatic.models.registry import ModelRegistry

tensorflow = pytest.importorskip("tensorflow")
tf = tensorflow.compat.v1


@pytest.mark.integration
@pytest.mark.pategan
def test_pategan_registry_load():
    cls = ModelRegistry.load_model("pategan")
    assert cls.__name__ == "PATEGAN"
    assert hasattr(cls, "train")
    assert hasattr(cls, "sample")


@pytest.mark.integration
@pytest.mark.pategan
def test_pategan_supported_models_list():
    supported = ModelRegistry.get_supported_models()
    assert "pategan" in supported


@pytest.mark.integration
@pytest.mark.pategan
def test_pategan_load_from_ref(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")

    ref = ModelRef(
        model_name="pategan",
        dataset_name="testdata",
        dataset_version="v1",
        train_run_id="run1",
    )

    state_dir = store.open_path(ref.state_relpath)
    state_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 2.0, 3.0],
        }
    )

    transformer = DataTransformer()
    transformer.fit(df)

    model = PATEGAN(
        niter=1,
        batch_size=2,
        z_dim=2,
    )

    model.transformer = transformer
    model._build_model(2)

    training_config = {
        "niter": 1,
        "batch_size": 2,
        "learning_rate": model.learning_rate,
        "lambda_gp": model.lambda_gp,
        "z_dim": 2,
    }

    privacy_config = {
        "epsilon": model.epsilon,
        "delta": model.delta,
        "num_teachers": model.num_teachers,
        "lambda_noise": 1.0,
    }

    save_metadata(
        str(state_dir / "metadata.json"),
        transformer,
        training_config,
        privacy_config,
        model.random_state,
    )

    saver = tf.train.Saver()
    saver.save(
        model._sess,
        str(state_dir / "pategan.ckpt"),
    )

    loaded = PATEGAN.load_from_ref(store, ref)

    assert isinstance(loaded, PATEGAN)
    assert loaded._is_fitted is True
    assert loaded.transformer.column_order == transformer.column_order
