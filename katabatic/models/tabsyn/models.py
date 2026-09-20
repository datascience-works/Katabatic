from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# NOTE: adjust this import if your base lives elsewhere
from katabatic.models.base_model import Model as BaseModel

from .utils import (
    TabSynConfig,
    TabSynState,
    evaluate_tabsyn,
    sample_tabsyn,
    train_tabsyn,
)

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


class TabSyn(BaseModel):
    """
    A lightweight TabSyn-style generator that learns a latent representation
    of tabular rows and trains a diffusion denoiser on those latents.

    This class ONLY uses functionality defined in tabsyn/utils.py.
    """

    ARTIFACT_STATE_FILES = ("tabsyn_state.pkl",)

    def __init__(
        self,
        *,
        # encoder / latent layout
        d_token: int = 16,
        # decoder training
        decoder_epochs: int = 50,
        decoder_batch_size: int = 2048,
        # diffusion training
        diffusion_epochs: int = 500,
        diffusion_batch_size: int = 4096,
        diffusion_hidden_dim: int = 512,
        # sampling
        diffusion_steps: int = 50,
        # misc
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 20,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.config = TabSynConfig(
            d_token=d_token,
            decoder_epochs=decoder_epochs,
            decoder_batch_size=decoder_batch_size,
            diffusion_epochs=diffusion_epochs,
            diffusion_batch_size=diffusion_batch_size,
            diffusion_hidden_dim=diffusion_hidden_dim,
            diffusion_steps=diffusion_steps,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed,
            device=device,
        )
        self.state: TabSynState | None = None

    # ---- Base hooks ---------------------------------------------------------

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        # Core libs like numpy/pandas are assumed already in katabatic.
        # Runtime DL deps for this model:
        return ["torch", "tqdm"]

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        extra_info: dict[str, Any] | None = None,
        **kwargs,
    ) -> TabSyn:
        """
        Train decoder & diffusion on the dataset located in `data_dir`, then
        materialize x_synth.csv / y_synth.csv for TSTR.

        Parameters
        ----------
        data_dir : str
            Directory containing TabSyn's preprocessed training arrays
            (X_num_train.npy / X_cat_train.npy / y_train.npy / info.json).
        synthetic_dir : str, optional
            Directory to write the generated x_synth.csv / y_synth.csv to.
            Defaults to synthetic/<dataset_name>/tabsyn.
        artifact_state_dir : str, optional
            When provided, the fitted model state is persisted here for
            later retrieval via load_from_ref().

        Returns
        -------
        TabSyn
            Trained model instance.
        """
        self.check_dependencies()
        # 1) fit model
        cfg = replace(
            self.config,
            decoder_epochs=kwargs.get("decoder_epochs", self.config.decoder_epochs),
            diffusion_epochs=kwargs.get(
                "diffusion_epochs", self.config.diffusion_epochs
            ),
            diffusion_steps=kwargs.get("diffusion_steps", self.config.diffusion_steps),
        )
        self.state = train_tabsyn(
            data_dir=data_dir,
            cfg=cfg,
            extra_info=extra_info or {},
        )
        self.is_fitted = True

        # 2) Decide where to save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir or not isinstance(synth_dir, str):
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "tabsyn")
        os.makedirs(synth_dir, exist_ok=True)

        # 3) Sample synthetic rows (defaults to number of training rows)
        df_s = self.sample(n_samples=None, return_df=True)

        # 4) Split into X / y for TSTR
        state = self.state
        n_num = state.n_num
        n_cat = len(state.cat_sizes)
        if n_cat == 0:
            # If this is a regression task, there is no categorical label to emit.
            # You can either raise or skip y_synth. TSTR expects a label, so raise:
            raise ValueError(
                "TSTR expects a label column, but no categorical columns were learned (regression task?)."
            )

        num_cols = [f"num_{i}" for i in range(n_num)]
        cat_cols = [f"cat_{i}" for i in range(n_cat)]
        # label (y) was moved to first categorical in _concat_xy
        y_col = cat_cols[0]
        # features = numerics + remaining categoricals
        X_cols = num_cols + cat_cols[1:]

        x_synth = df_s[X_cols].copy()
        y_synth = df_s[y_col]

        # Align synthetic feature names & order with real train CSV
        real_x_train_path = os.path.join(data_dir, "x_train.csv")
        try:
            real_cols = pd.read_csv(real_x_train_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                # 1) rename to real names (even if current names differ)
                x_synth.columns = real_cols
                # 2) reorder columns to match exactly (defensive; ensures identical order)
                x_synth = x_synth.reindex(columns=real_cols)
            else:
                print(
                    f"[TabSyn] Warning: feature count mismatch: "
                    f"synthetic={x_synth.shape[1]} vs real={len(real_cols)}. "
                    "Leaving synthetic column names as-is."
                )
        except Exception as e:
            print(
                f"[TabSyn] Warning: could not align feature names using {real_x_train_path}: {e}"
            )

        # 5) Write CSVs that TSTR expects
        x_path = os.path.join(synth_dir, "x_synth.csv")
        y_path = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path, index=False)
        y_synth.to_csv(y_path, index=False, header=True)
        print(f"[TabSyn] Synthetic data saved:\n  X -> {x_path}\n  y -> {y_path}")

        self._maybe_save_artifact_state(artifact_state_dir)

        return self

    def evaluate(
        self,
        *,
        data_dir: str,
        split: str = "test",
    ) -> float:
        """Return a scalar loss on the given split (lower is better)."""
        if not self.is_fitted or self.state is None:
            raise RuntimeError("Call train() before evaluate().")
        return evaluate_tabsyn(self.state, data_dir=data_dir, split=split)

    def sample(
        self,
        n_samples: int | None = None,
        return_df: bool = True,
        save_path: str | None = None,
        *args,
        **kwargs,
    ) -> np.ndarray | pd.DataFrame:
        """Generate synthetic rows. If `return_df` True, returns a DataFrame."""
        if not self.is_fitted or self.state is None:
            raise RuntimeError("Call train() before sample().")
        out = sample_tabsyn(
            self.state,
            n_samples=n_samples,
            return_df=return_df,
        )
        if save_path is not None:
            if isinstance(out, pd.DataFrame):
                out.to_csv(save_path, index=False)
            else:
                np.save(save_path, out)
        return out

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """
        Persist fitted state so the model can be rebuilt by load_from_ref().

        Called from train() when the pipeline injects artifact_state_dir, which
        it does for any model class declaring ARTIFACT_STATE_FILES.
        """
        import torch

        state = self.state
        precond = state.denoise_fn  # full _Precond, including its MLPDiffusion backbone

        os.makedirs(artifact_state_dir, exist_ok=True)
        bundle = {
            "denoise_fn": precond.state_dict(),
            "tokenizer": state.tokenizer_state,
            "encoder": state.encoder_state,
            "decoder": state.decoder_state,
            # Metadata needed to rebuild TabSynState in TabSyn.load_from_ref().
            "meta": {
                "info": state.info,
                "n_num": state.n_num,
                "cat_sizes": state.cat_sizes,
                "cat_encoders": state.cat_encoders,
                "token_dim": state.token_dim,
                "column_order": state.column_order,
                "scaler_mean": state.scaler_mean,
                "scaler_std": state.scaler_std,
                "train_rows": state.train_rows,
                "denoise_dim_t": precond.denoise_fn.dim_t,
                "sigma_data": precond.sigma_data,
                "num_steps": getattr(precond, "num_steps", 50),
                "device": str(state.device),
            },
        }
        target = os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES[0])
        torch.save(bundle, target)

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> TabSyn:
        """
        Rehydrate a trained TabSyn from a versioned artifact.
        """
        import torch

        from .utils import MLPDiffusion, TabSynState, _Precond

        state_path = cls._require_state_file(store, ref)

        bundle = torch.load(state_path, map_location="cpu", weights_only=False)  # nosec B614: loading our own saved artifact store

        if "meta" not in bundle:
            raise ValueError(
                f"TabSyn artifact at {state_path} predates the metadata bundle "
                f"and cannot be reloaded. Retrain to regenerate it."
            )

        meta = bundle["meta"]
        device = torch.device("cpu")

        # Same in_dim formula sample_tabsyn() uses: one token per column.
        n_cols = meta["n_num"] + len(meta["cat_sizes"])
        d_in = n_cols * meta["token_dim"]

        precond = _Precond(
            MLPDiffusion(d_in=d_in, dim_t=meta["denoise_dim_t"]),
            sigma_data=meta["sigma_data"],
        )
        precond.load_state_dict(bundle["denoise_fn"])
        precond.num_steps = meta["num_steps"]
        precond = precond.to(device).eval()

        instance = cls()
        instance.state = TabSynState(
            info=meta["info"],
            n_num=meta["n_num"],
            cat_sizes=meta["cat_sizes"],
            cat_encoders=meta["cat_encoders"],
            token_dim=meta["token_dim"],
            column_order=meta["column_order"],
            scaler_mean=meta["scaler_mean"],
            scaler_std=meta["scaler_std"],
            tokenizer_state=bundle["tokenizer"],
            encoder_state=bundle["encoder"],
            decoder_state=bundle["decoder"],
            denoise_fn=precond,
            device=device,
            train_rows=meta["train_rows"],
        )
        instance.is_fitted = True
        return instance
