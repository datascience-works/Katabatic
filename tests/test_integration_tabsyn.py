from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from tests.conftest import require_backend

torch = require_backend("torch", "save")

from katabatic.artifacts import LocalArtifactStore  # noqa: E402
from katabatic.artifacts.refs import ModelRef  # noqa: E402
from katabatic.models.registry import ModelRegistry  # noqa: E402
from katabatic.models.tabsyn.models import TabSyn  # noqa: E402
from katabatic.models.tabsyn.utils import (  # noqa: E402
    MLPDiffusion,
    _Decoder,
    _Encoder,
    _Precond,
    _Tokenizer,
)

N_NUM = 2
CAT_SIZES = [2]
TOKEN_DIM = 4
DIM_T = 16
TRAIN_ROWS = 10


@pytest.mark.integration
@pytest.mark.tabsyn
def test_tabsyn_registry_load():
    cls = ModelRegistry.load_model("tabsyn")
    assert cls.__name__ == "TabSyn"
    for method in ("train", "sample", "load_from_ref"):
        assert hasattr(cls, method)


@pytest.mark.integration
@pytest.mark.tabsyn
def test_tabsyn_supported_models_list():
    assert "tabsyn" in ModelRegistry.get_supported_models()


@pytest.mark.integration
@pytest.mark.tabsyn
def test_tabsyn_declares_artifact_state_files():
    assert isinstance(TabSyn.ARTIFACT_STATE_FILES, (tuple, list))
    assert TabSyn.ARTIFACT_STATE_FILES == ("tabsyn_state.pkl",)


def _write_untrained_bundle(state_dir) -> None:
    """
    Write a structurally valid TabSyn artifact without training.

    Full TabSyn training is a VAE plus a diffusion model, far too slow to run on
    every CI job. This fabricates the same bundle train_tabsyn() writes, using
    freshly initialised modules. The weights are meaningless — the point is that
    load_from_ref() can reconstruct a usable model from the artifact alone.
    """
    tokenizer = _Tokenizer(n_num=N_NUM, cat_sizes=CAT_SIZES, d_token=TOKEN_DIM)
    encoder = _Encoder(d_token=TOKEN_DIM)
    decoder = _Decoder(n_num=N_NUM, cat_sizes=CAT_SIZES, d_token=TOKEN_DIM)

    # Same in_dim formula sample_tabsyn() uses: one token per column, flattened.
    d_in = (N_NUM + len(CAT_SIZES)) * TOKEN_DIM
    precond = _Precond(MLPDiffusion(d_in=d_in, dim_t=DIM_T), sigma_data=0.5)
    precond.num_steps = 2  # keep diffusion sampling fast in CI

    label_encoder = LabelEncoder().fit(np.array(["a", "b"], dtype=object))

    bundle = {
        "denoise_fn": precond.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "meta": {
            "info": {"task_type": "binclass"},
            "n_num": N_NUM,
            "cat_sizes": CAT_SIZES,
            "cat_encoders": [label_encoder],
            "token_dim": TOKEN_DIM,
            "column_order": list(range(N_NUM + len(CAT_SIZES))),
            "scaler_mean": np.zeros(N_NUM),
            "scaler_std": np.ones(N_NUM),
            "train_rows": TRAIN_ROWS,
            "denoise_dim_t": DIM_T,
            "sigma_data": 0.5,
            "num_steps": 2,
            "device": "cpu",
        },
    }
    torch.save(bundle, str(state_dir / "tabsyn_state.pkl"))


@pytest.mark.integration
@pytest.mark.tabsyn
def test_tabsyn_load_from_ref(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = ModelRef(
        model_name="tabsyn",
        dataset_name="testdata",
        dataset_version="v1",
        train_run_id="run1",
    )

    state_dir = store.open_path(ref.state_relpath)
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_untrained_bundle(state_dir)

    loaded = TabSyn.load_from_ref(store, ref)

    assert isinstance(loaded, TabSyn)
    assert loaded.is_fitted is True
    assert loaded.state is not None
    assert loaded.state.n_num == N_NUM
    assert loaded.state.cat_sizes == CAT_SIZES

    out = loaded.sample(n_samples=5)
    assert len(out) == 5
    assert list(out.columns) == ["num_0", "num_1", "cat_0"]
    assert set(out["cat_0"].unique()).issubset({"a", "b"})


@pytest.mark.integration
@pytest.mark.tabsyn
def test_tabsyn_load_from_ref_missing_state(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = ModelRef(
        model_name="tabsyn",
        dataset_name="testdata",
        dataset_version="v1",
        train_run_id="missing",
    )
    with pytest.raises(FileNotFoundError):
        TabSyn.load_from_ref(store, ref)
