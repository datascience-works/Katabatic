from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

from katabatic.models.base_model import Model

from .utils import encode_categorical_onehot, decode_categorical_onehot

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

class ForestDiffusionModel(Model):
    """
    Katabatic wrapper for Forest Diffusion / Forest Flow (tree-based
    score/vector-field matching).

    Trains one regressor per (class label, noise level) to learn the
    score function (VP-diffusion branch) or vector field (flow-matching
    branch), then reverses the process to generate synthetic samples.
    """

    def __init__(
        self,
        n_t: int = 50,
        model: str = "xgboost",
        max_depth: int = 7,
        n_estimators: int = 100,
        eta: float = 0.3,
        tree_method: str = "hist",
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        subsample: float = 1.0,
        num_leaves: int = 31,
        int_indexes: Optional[List[int]] = None,
        diffusion_type: str = "flow",
        duplicate_K: int = 100,
        n_jobs: int = -1,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        eps: float = 1e-3,
        gpu_hist: bool = False,
        seed: int = 666,
        **xgboost_kwargs,
    ):
        super().__init__()

        if diffusion_type not in ("flow", "vp"):
            raise ValueError("diffusion_type must be 'flow' or 'vp'")

        self.n_t = n_t
        self.model_type = model
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta
        self.tree_method = tree_method
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.int_indexes = [] if int_indexes is None else list(int_indexes)
        self.diffusion_type = diffusion_type
        self.duplicate_K = duplicate_K
        self.n_jobs = n_jobs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps
        self.gpu_hist = gpu_hist
        self.seed = seed
        self.xgboost_kwargs = xgboost_kwargs

        # Populated during train() / _fit_internal()
        self.label_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self.encoder = None
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._n_num_cols: int = 0
        self.X_min = None
        self.X_max = None
        self.scaler = None
        self.regr = None
        self.X = None
        self.label_y = None
        self.y_vals = None
        self.y_probs = None
        self.n = None
        self.c = None

    def train(
        self,
        output_dir: str,
        label_col: Optional[str] = None,
        synthetic_dir: Optional[str] = None,
        **kwargs,
    ) -> "ForestDiffusionModel":

        x_train_df = pd.read_csv(f"{output_dir}/x_train.csv")
        y_df = pd.read_csv(f"{output_dir}/y_train.csv")

        # Store label col and feature col names before converting to numpy
        if label_col is None:
            label_col = y_df.columns[0]
        self.label_col = str(label_col)
        self._feature_cols = list(x_train_df.columns)

        y_df.columns = [str(c) for c in y_df.columns]
        y_train = y_df[self.label_col].values

        x_train, self.encoder, self._cat_cols, self._num_cols, self._n_num_cols = (
            encode_categorical_onehot(x_train_df)
        )

        x_train = x_train.astype(np.float32)
        self._fit_internal(x_train, y_train)
        self.is_fitted = True

        x_synth, y_synth = self.sample(len(x_train))

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            pd.DataFrame(y_synth, columns=[self.label_col]).to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        return self

    def _fit_internal(self, X: np.ndarray, label_y: Optional[np.ndarray] = None):
        # Drop all-NaN rows
        mask_valid = ~np.isnan(X).all(axis=1)
        X = X[mask_valid]
        if label_y is not None:
            label_y = label_y[mask_valid]

        self.X_min = np.nanmin(X, axis=0, keepdims=True)
        self.X_max = np.nanmax(X, axis=0, keepdims=True)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X = self.scaler.fit_transform(X)

        self.X = X
        self.n, self.c = X.shape

        self.label_y = label_y
        if label_y is not None:
            self.y_vals, counts = np.unique(label_y, return_counts=True)
            self.y_probs = counts / counts.sum()
        else:
            self.y_vals = np.array([0])
            self.y_probs = np.array([1.0])

        timesteps = np.linspace(1.0, self.eps, self.n_t)

        jobs = []
        for j, lab in enumerate(self.y_vals):
            mask = (
                (label_y == lab) if label_y is not None else np.ones(self.n, dtype=bool)
            )
            X_class = X[mask]

            X_dup = np.tile(X_class, (self.duplicate_K, 1))

            for step_idx, t in enumerate(timesteps):
                job_seed = self.seed + j * self.n_t + step_idx
                jobs.append((j, step_idx, t, X_dup, job_seed))

        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one_regressor)(t, X_dup, job_seed)
            for (_, _, t, X_dup, job_seed) in jobs
        )

        self.regr = [[None] * self.n_t for _ in self.y_vals]
        for (j, step_idx, _, _, _), reg in zip(jobs, fitted):
            self.regr[j][step_idx] = reg

    def _fit_one_regressor(self, t: float, X: np.ndarray, seed: int):
        """Fit one regressor at noise level `t` (Algorithm 1 / Algorithm 5
        "Forward 0 -> t" of the paper).
        """
        rng = np.random.RandomState(seed)
        Z = rng.normal(size=X.shape)
        X_t, Y_t = self._forward_0_to_t(X, Z, t)

        reg = self._make_regressor()
        reg.fit(X_t, Y_t)
        return reg

    def _forward_0_to_t(self, X: np.ndarray, Z: np.ndarray, t: float):
        """Algorithm 5 ("Forward 0 -> t") of the paper, for both branches.
        """
        if self.diffusion_type == "flow":
            X_t = (1.0 - t) * Z + t * X
            Y_t = X - Z
        else:  # "vp"
            C = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
            std = np.sqrt(np.clip(1.0 - np.exp(2.0 * C), 1e-12, None))
            X_t = np.exp(C) * X + std * Z
            Y_t = Z
        return X_t, Y_t

    def _reverse_step(self, X_t: np.ndarray, t: float, f_t: np.ndarray) -> np.ndarray:
        """Algorithm 7 ("Reverse") of the paper, for both branches."""
        h = 1.0 / self.n_t
        if self.diffusion_type == "flow":
            return X_t + h * f_t
        else:  # "vp"
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            C = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
            std = np.sqrt(np.clip(1.0 - np.exp(2.0 * C), 1e-12, None))
            score = -f_t / (std + 1e-8)
            mu = -0.5 * beta_t * X_t - beta_t * score
            Z = np.random.normal(size=X_t.shape)
            return X_t - h * mu + beta_t * np.sqrt(h) * Z

    def _predict_ensemble(self, X: np.ndarray, step: int) -> np.ndarray:
        """Class-probability-weighted average of the per-class regressor
        predictions at a given timestep index."""
        out = np.zeros_like(X)
        for j in range(len(self.y_vals)):
            out += self.y_probs[j] * self.regr[j][step].predict(X)
        return out

    def _generate_raw(self, batch_size: Optional[int] = None) -> np.ndarray:
        """Run the reverse process and return the generated matrix in the
        model's internal scaled representation (before inverse scaling,
        clipping, or categorical decoding). Used by both `generate()` and
        `evaluate()`."""
        b = self.n if batch_size is None else batch_size
        X_t = np.random.normal(size=(b, self.c))
        h = 1.0 / self.n_t
        t = 1.0
        while t > 0:
            step = int(round(t * (self.n_t - 1)))
            step = max(0, min(step, self.n_t - 1))
            f_t = self._predict_ensemble(X_t, step)
            X_t = self._reverse_step(X_t, t, f_t)
            t -= h
        return X_t

    def generate(self, batch_size: Optional[int] = None):
        X_t = self._generate_raw(batch_size)
        x = self.scaler.inverse_transform(X_t)
        x = self._clip(x)

        df_x = decode_categorical_onehot(
            x,
            self.encoder,
            self._cat_cols,
            self._num_cols,
            self._n_num_cols,
            self._feature_cols,
        )

        b = df_x.shape[0]
        labels = self.y_vals[
            np.argmax(np.random.multinomial(1, self.y_probs, size=b), axis=1)
        ]

        return df_x, labels

    def impute(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        NOTE: this implements the base algorithm only. The paper's
        extended REPAINT variant (Algorithm 4), which periodically jumps
        the trajectory backward and re-runs a few steps for better
        consistency, is not implemented here and would be a good
        follow-up once this base version has been validated.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before impute().")

        # Missingness must be captured before any string/one-hot
        # conversion, since encoding NaN as the literal string "nan"
        # would otherwise hide it from the mask.
        num_missing = X_df[self._num_cols].isna().to_numpy()
        num_vals = X_df[self._num_cols].astype(np.float32).fillna(0.0).to_numpy()

        if self._cat_cols:
            cat_missing_cols = X_df[self._cat_cols].isna()
            cat_filled = X_df[self._cat_cols].astype(object).where(
                ~cat_missing_cols, "__MISSING__"
            )
            cat_onehot = self.encoder.transform(cat_filled.astype(str))

            # Expand each categorical column's missingness across its
            # one-hot block so the mask lines up with `cat_onehot`.
            block_sizes = [len(cats) for cats in self.encoder.categories_]
            cat_missing = np.concatenate(
                [
                    np.repeat(cat_missing_cols[[col]].to_numpy(), size, axis=1)
                    for col, size in zip(self._cat_cols, block_sizes)
                ],
                axis=1,
            )

            X_raw = np.concatenate([num_vals, cat_onehot], axis=1)
            missing = np.concatenate([num_missing, cat_missing], axis=1)
        else:
            X_raw = num_vals
            missing = num_missing

        M = ~missing  # True where NON-missing
        X_filled = np.where(M, X_raw, 0.0)
        X_scaled = self.scaler.transform(X_filled)

        n_obs, d = X_scaled.shape
        Z = np.random.normal(size=(n_obs, d))
        X_t = Z.copy()
        h = 1.0 / self.n_t
        t = 1.0
        while t > 0:
            step = int(round(t * (self.n_t - 1)))
            step = max(0, min(step, self.n_t - 1))

            f_t = self._predict_ensemble(X_t, step)
            X_next = self._reverse_step(X_t, t, f_t)

            t_next = max(t - h, 0.0)
            X_true_t, _ = self._forward_0_to_t(X_scaled, Z, t_next)
            X_next = np.where(M, X_true_t, X_next)

            X_t = X_next
            t -= h

        x = self.scaler.inverse_transform(X_t)
        x = self._clip(x)
        x = np.where(M, X_raw, x)  # restore exact original observed values

        return decode_categorical_onehot(
            x,
            self.encoder,
            self._cat_cols,
            self._num_cols,
            self._n_num_cols,
            self._feature_cols,
        )

    def sample(self, n_samples: int, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")
        x_synth_df, y_synth = self.generate(batch_size=n_samples)
        return x_synth_df, y_synth

    def evaluate(self, X_real: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        NOTE: the paper's actual Wtrain/Wtest computes distance on a
        Gower-scaled representation of the *joint* mixed-type
        distribution; this is a simpler per-column marginal
        approximation computed in the model's internal scaled
        representation, and should be read as a coarser proxy rather
        than a drop-in replacement for the paper's metric.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        from scipy.stats import wasserstein_distance

        x_synth_scaled = self._generate_raw(self.n)

        if X_real is None:
            X_real = self.X

        n_cols = min(X_real.shape[1], x_synth_scaled.shape[1])
        w_dists = [
            wasserstein_distance(X_real[:, i], x_synth_scaled[:, i])
            for i in range(n_cols)
        ]
        return float(np.mean(w_dists))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_regressor(self):
        """Construct (unfitted) regressor of the configured type."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=self.seed,
            )

        if self.model_type == "lgbm":
            if LGBMRegressor is None:
                raise ImportError("lightgbm not installed")
            return LGBMRegressor(
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                random_state=self.seed,
            )

        if self.model_type == "catboost":
            if CatBoostRegressor is None:
                raise ImportError("catboost not installed")
            return CatBoostRegressor(
                iterations=self.n_estimators,
                max_depth=self.max_depth,
                loss_function="RMSE",
                silent=True,
                random_seed=self.seed,
            )

        # Default: XGBoost
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            objective="reg:squarederror",
            eta=self.eta,
            max_depth=self.max_depth,
            subsample=self.subsample,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            tree_method="gpu_hist" if self.gpu_hist else "hist",
            device="cuda" if self.gpu_hist else "cpu",
            seed=self.seed,
            **self.xgboost_kwargs,
        )

    def _clip(self, X: np.ndarray) -> np.ndarray:
        # NOTE: `int_indexes` refers to positions in the model's internal
        # (numeric-columns-first, then one-hot-categorical) representation.
        for i in self.int_indexes:
            X[:, i] = np.round(X[:, i])
        X = np.minimum(np.maximum(X, self.X_min), self.X_max)
        return X
