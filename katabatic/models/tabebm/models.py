from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from katabatic.models.base_model import Model

from .utils import decode_df, encode_df, fit_schema_stats, infer_schema

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(
    X: np.ndarray | torch.Tensor | pd.DataFrame | pd.Series | None,
) -> np.ndarray | None:
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, pd.Series):
        return X.to_numpy()
    if X is None:
        return None
    raise ValueError("Unsupported input type")


class _TabEBMBackend:
    """
    NumPy-based TabEBM backend.
    Avoids torch/autograd kernel crashes.
    """

    def __init__(self, max_data_size: int = 10000):
        self.max_data_size = max_data_size

    def generate(
        self,
        X,
        y,
        num_samples: int,
        starting_point_noise_std: float = 0.01,
        sgld_step_size: float = 0.01,
        sgld_noise_std: float = 0.01,
        sgld_steps: int = 200,
        distance_negative_class: float = 5.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        X = to_numpy(X).astype(np.float32)
        y = to_numpy(y).reshape(-1)

        if X.shape[0] > self.max_data_size:
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=self.max_data_size,
                stratify=y,
                random_state=seed,
            )

        results = {}
        unique_classes = np.unique(y)

        for class_idx, cls in enumerate(unique_classes):
            X_cls = X[y == cls]

            if len(X_cls) == 0:
                continue

            X_ebm, y_ebm = self.add_surrogate_negative_samples(
                X_cls,
                distance_negative_class,
                rng,
            )

            start_idx = rng.integers(0, X_cls.shape[0], size=num_samples)
            X_sgld = X_cls[start_idx].copy()
            X_sgld += rng.normal(
                0,
                starting_point_noise_std,
                size=X_sgld.shape,
            ).astype(np.float32)

            X_real = X_ebm[y_ebm == 0]

            for _ in range(sgld_steps):
                grad = self.compute_energy_gradient(X_sgld, X_ebm, X_real)
                grad = np.clip(grad, -5.0, 5.0)
                X_sgld -= sgld_step_size * grad
                X_sgld += rng.normal(
                    0,
                    sgld_noise_std,
                    size=X_sgld.shape,
                ).astype(np.float32)
                nan_rows = np.any(np.isnan(X_sgld), axis=1)
                if nan_rows.any():
                    repl = rng.integers(0, X_cls.shape[0], size=int(nan_rows.sum()))
                    X_sgld[nan_rows] = X_cls[repl]

            results[f"class_{class_idx}"] = X_sgld.astype(np.float32)

        return results

    @staticmethod
    def compute_energy_gradient(X_synth, X_train, X_real):
        eps = 1e-8

        # Gradient of nearest-neighbour distance
        diff_all = X_synth[:, None, :] - X_train[None, :, :]
        dist_all = np.sqrt(np.sum(diff_all**2, axis=2) + eps)

        nearest_idx = np.argmin(dist_all, axis=1)
        nearest_diff = diff_all[np.arange(len(X_synth)), nearest_idx]
        nearest_dist = dist_all[np.arange(len(X_synth)), nearest_idx][:, None]

        grad_min = nearest_diff / nearest_dist

        # Gradient of mean distance to real class samples
        diff_real = X_synth[:, None, :] - X_real[None, :, :]
        dist_real = np.sqrt(np.sum(diff_real**2, axis=2) + eps)

        grad_mean = np.mean(
            diff_real / dist_real[:, :, None],
            axis=1,
        )

        return grad_min + grad_mean

    @staticmethod
    def add_surrogate_negative_samples(X, distance_negative_class, rng):
        num_features = X.shape[1]

        surrogates = []
        for _ in range(4):
            pt = rng.choice(
                [-distance_negative_class, distance_negative_class],
                size=num_features,
            )
            surrogates.append(pt)

        X_sur = np.array(surrogates, dtype=X.dtype)

        X_ebm = np.concatenate([X, X_sur], axis=0)
        y_ebm = np.concatenate(
            [
                np.zeros(X.shape[0]),
                np.ones(len(X_sur)),
            ]
        ).astype(int)

        return X_ebm, y_ebm


@dataclass
class TabEBMConfig:
    max_data_size: int = 10000
    starting_point_noise_std: float = 0.01
    sgld_step_size: float = 0.01
    sgld_noise_std: float = 0.01
    sgld_steps: int = 200
    distance_negative_class: float = 5.0
    seed: int = 42
    debug: bool = False


class TabEBMModel(Model):
    ARTIFACT_STATE_FILES = ("tabebm_state.pkl",)

    def __init__(self, target_col: str = "target", config: TabEBMConfig | None = None):
        super().__init__()
        self.check_dependencies()

        self.target_col = target_col
        self.config = config or TabEBMConfig()

        self._tabebm = _TabEBMBackend(max_data_size=self.config.max_data_size)

        self.schema_ = None
        self.col_order_ = None
        self._x_train = None
        self._y_train = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["numpy", "pandas", "sklearn", "torch"]

    def train(
        self,
        data_dir: str,
        *args,
        label_col=None,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        **kwargs,
    ) -> TabEBMModel:
        x_train = pd.read_csv(f"{data_dir}/x_train.csv")
        # Convert pandas StringDtype -> object so ordinal encoder handles them correctly
        str_cols = [
            c
            for c in x_train.columns
            if str(x_train[c].dtype).startswith("string")
            or str(x_train[c].dtype) == "str"
        ]
        if str_cols:
            x_train[str_cols] = x_train[str_cols].astype(object)
        y_train_df = pd.read_csv(f"{data_dir}/y_train.csv")

        if label_col is None:
            label_col = y_train_df.columns[0]

        self.target_col = str(label_col)
        y_train = y_train_df.iloc[:, 0]

        schema = infer_schema(x_train)
        fit_schema_stats(x_train, schema)

        self.schema_ = schema
        self.col_order_ = list(x_train.columns)
        self._x_train = x_train
        self._y_train = y_train

        self.is_fitted = True

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)

            df_s = self.sample(n_samples=len(x_train))
            df_s[self.col_order_].to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"), index=False
            )
            df_s[[self.target_col]].to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        self._maybe_save_artifact_state(artifact_state_dir)
        return self

    def sample(
        self,
        n_samples: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.schema_ is None or self.col_order_ is None:
            raise RuntimeError("TabEBMModel not fitted. Call train() first.")

        n = int(n_samples) if n_samples is not None else len(self._x_train)

        X_enc, _, _ = encode_df(self._x_train, self.schema_)

        y_raw = self._y_train.to_numpy()
        y_fact, uniques = pd.factorize(y_raw)

        classes, counts = np.unique(y_fact, return_counts=True)
        probs = counts / counts.sum()
        per_class = np.floor(probs * n).astype(int)

        remainder = n - per_class.sum()
        if remainder > 0:
            for i in np.argsort(-probs)[:remainder]:
                per_class[i] += 1

        max_per_class = int(per_class.max()) if len(per_class) else 0
        if max_per_class == 0:
            return pd.DataFrame(columns=[*self.col_order_, self.target_col])

        generated = self._tabebm.generate(
            X=X_enc,
            y=y_fact,
            num_samples=max_per_class,
            starting_point_noise_std=self.config.starting_point_noise_std,
            sgld_step_size=self.config.sgld_step_size,
            sgld_noise_std=self.config.sgld_noise_std,
            sgld_steps=self.config.sgld_steps,
            distance_negative_class=self.config.distance_negative_class,
            seed=self.config.seed,
        )

        xs = []
        ys = []

        for idx in classes:
            need = int(per_class[idx])
            if need <= 0:
                continue

            key = f"class_{int(idx)}"
            Xc = generated[key][:need]

            dfc = decode_df(Xc, self.schema_)
            dfc = dfc[self.col_order_]

            xs.append(dfc)
            ys.append(np.full(need, uniques[idx]))

        x_synth = (
            pd.concat(xs, ignore_index=True)
            if xs
            else pd.DataFrame(columns=self.col_order_)
        )
        y_synth = pd.Series(
            np.concatenate(ys) if ys else np.array([]), name=self.target_col
        )

        result = x_synth.copy()
        result[self.target_col] = y_synth.to_numpy()
        return result

    def evaluate(self, *args, **kwargs) -> float:
        """
        TabEBMModel has no meaningful standalone metric to offer.
        Use the Katabatic evaluation pipeline for cross-model metrics instead.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        raise NotImplementedError(
            "TabEBMModel.evaluate() has no meaningful standalone metric to offer. "
            "Use the Katabatic evaluation pipeline for cross-model metrics instead."
        )

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        os.makedirs(artifact_state_dir, exist_ok=True)
        state = {
            "target_col": self.target_col,
            "config": self.config,
            "schema_": self.schema_,
            "col_order_": self.col_order_,
            "x_train": self._x_train,
            "y_train": self._y_train,
        }
        state_path = os.path.join(artifact_state_dir, self.ARTIFACT_STATE_FILES[0])
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> TabEBMModel:
        state_path = cls._require_state_file(store, ref)
        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(target_col=state["target_col"], config=state["config"])
        instance.schema_ = state["schema_"]
        instance.col_order_ = state["col_order_"]
        instance._x_train = state["x_train"]
        instance._y_train = state["y_train"]
        instance.is_fitted = True
        return instance
