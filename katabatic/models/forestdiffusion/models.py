from __future__ import annotations

import os
import abc
import functools
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from katabatic.models.base_model import Model

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


# ---------------------------------------------------------------------------
# SDE framework
# ---------------------------------------------------------------------------

class SDE(abc.ABC):
    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    def reverse(self, score_fn, probability_flow=False):
        N = self.N
        T = self.T
        sde_fn = self.sde

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)           # positional — avoids kwarg mismatch
                drift = drift - (diffusion ** 2) * (
                    score * (0.5 if self.probability_flow else 1.0)
                )
                diffusion = (
                    np.zeros_like(diffusion) if self.probability_flow else diffusion
                )
                return drift, diffusion

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = np.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t * x
        diffusion = np.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = np.exp(log_mean_coeff) * x
        std = np.sqrt(1.0 - np.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return np.random.normal(size=shape)


# ---------------------------------------------------------------------------
# Predictor / sampler
# ---------------------------------------------------------------------------

class Predictor(abc.ABC):
    def __init__(self, sde, score_fn):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, h):
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn):
        super().__init__(sde, score_fn)

    def update_fn(self, x, t, h):
        z = self.sde.prior_sampling(x.shape)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x - drift * h
        x = x_mean + diffusion * np.sqrt(h) * z
        return x, x_mean


def shared_predictor_update_fn(x, t, h=None, sde=None, score_fn=None):
    predictor_obj = EulerMaruyamaPredictor(sde, score_fn)
    return predictor_obj.update_fn(x, t, h)


def get_pc_sampler(score_fn, sde, denoise=True, eps=1e-3):
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn, sde=sde, score_fn=score_fn
    )

    def pc_sampler(prior):
        x = prior
        timesteps = np.linspace(sde.T, eps, sde.N)
        h = timesteps - np.append(timesteps, 0)[1:]

        for i in range(sde.N - 1):
            x, _ = predictor_update_fn(x, timesteps[i], h[i])

        if denoise:
            _, std = sde.marginal_prob(x, eps)
            x = x + (std**2) * score_fn(x, eps)

        return x

    return pc_sampler


# ---------------------------------------------------------------------------
# ForestDiffusionModel
# ---------------------------------------------------------------------------

class ForestDiffusionModel(Model):
    """
    Katabatic wrapper for Forest Diffusion (tree-based score matching).

    Trains one regressor per (class label, diffusion timestep) to learn the
    score function, then uses an Euler-Maruyama reverse SDE to generate
    synthetic samples.
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
        cat_indexes: Optional[List[int]] = None,
        int_indexes: Optional[List[int]] = None,
        beta_min: float = 0.1,
        beta_max: float = 8.0,
        eps: float = 1e-3,
        gpu_hist: bool = False,
        seed: int = 666,
        **xgboost_kwargs,
    ):
        super().__init__()

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
        self.cat_indexes = [] if cat_indexes is None else list(cat_indexes)
        self.int_indexes = [] if int_indexes is None else list(int_indexes)
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
        self.X_min = None
        self.X_max = None
        self.scaler = None
        self.sde = None
        self.regr = None
        self.X = None
        self.label_y = None
        self.y_vals = None
        self.y_probs = None
        self.n = None
        self.c = None
        self.cols_before = None
        self.cols_after = None

    # ------------------------------------------------------------------
    # Katabatic pipeline entry point
    # ------------------------------------------------------------------

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

        if any(x_train_df.dtypes == "object"):
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            x_train = self.encoder.fit_transform(x_train_df.astype(str))
        else:
            x_train = x_train_df.to_numpy()

        x_train = x_train.astype(np.float32)
        self._fit_internal(x_train, y_train)
        self.is_fitted = True

        x_synth, y_synth = self.sample(len(x_train))

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
            pd.DataFrame(x_synth, columns=self._feature_cols).to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"), index=False
            )
            pd.DataFrame(y_synth, columns=[self.label_col]).to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        return self

    # ------------------------------------------------------------------
    # Fit: build noise schedule, train regressors
    # ------------------------------------------------------------------

    def _fit_internal(self, X: np.ndarray, label_y: Optional[np.ndarray] = None):
        np.random.seed(self.seed)

        # Drop all-NaN rows
        mask_valid = ~np.isnan(X).all(axis=1)
        X = X[mask_valid]
        if label_y is not None:
            label_y = label_y[mask_valid]

        self.X_min = np.nanmin(X, axis=0, keepdims=True)
        self.X_max = np.nanmax(X, axis=0, keepdims=True)

        if len(self.cat_indexes) > 0:
            X, self.cols_before, self.cols_after = self._dummify(X)

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

        self.sde = VPSDE(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            N=self.n_t,
        )

        timesteps = np.linspace(self.sde.T, self.eps, self.n_t)

        # One list of regressors per class label, one regressor per timestep.
        # Each regressor is fitted immediately after construction.
        self.regr = []
        for j, lab in enumerate(self.y_vals):
            mask = (label_y == lab) if label_y is not None else np.ones(self.n, dtype=bool)
            X_class = X[mask]
            regressors_for_class = []

            for step_idx, t in enumerate(timesteps):
                # Add VP noise at this timestep
                mean, std = self.sde.marginal_prob(X_class, t)
                noise = np.random.normal(size=X_class.shape)
                X_noisy = mean + std * noise

                # Score target: -noise / std  (Tweedie / DSM objective)
                score_target = -noise / (std + 1e-8)

                reg = self._make_regressor()
                # Fit one regressor predicting the flattened score target
                # per-column (multi-output via loop, matching _score_fn usage)
                reg.fit(X_noisy, score_target)
                regressors_for_class.append(reg)

            self.regr.append(regressors_for_class)

    # ------------------------------------------------------------------
    # Score function — called during reverse SDE
    # ------------------------------------------------------------------

    def _score_fn(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate the score (grad log p_t) at noisy input x and time t.

        Averages predictions across class-conditional regressors weighted
        by class probability.
        """
        X = x.reshape(-1, self.c)
        step = int(round(float(t) * (self.n_t - 1)))
        step = max(0, min(step, self.n_t - 1))

        out = np.zeros_like(X)
        for j in range(len(self.y_vals)):
            pred = self.regr[j][step].predict(X)       # (n, c)
            out += self.y_probs[j] * pred

        _, sigma = self.sde.marginal_prob(X, t)
        out = -out / (sigma + 1e-8)
        return out.reshape(x.shape)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, batch_size: Optional[int] = None) -> np.ndarray:
        b = self.n if batch_size is None else batch_size
        prior = np.random.normal(size=(b, self.c))

        sampler = get_pc_sampler(
            score_fn=self._score_fn,
            sde=self.sde,
            denoise=True,
            eps=self.eps,
        )

        x = sampler(prior).reshape(b, self.c)
        x = self.scaler.inverse_transform(x)
        x = self._clip(x)

        # Sample labels from empirical class distribution
        labels = self.y_vals[
            np.argmax(np.random.multinomial(1, self.y_probs, size=b), axis=1)
        ]

        return np.concatenate([x, labels[:, None]], axis=1)

    # ------------------------------------------------------------------
    # Base class abstract methods
    # ------------------------------------------------------------------

    def sample(self, n_samples: int, **kwargs):
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")
        synth = self.generate(batch_size=n_samples)
        x_synth = synth[:, :-1]
        y_synth = synth[:, -1]
        return x_synth, y_synth

    def evaluate(self, X_real: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        Mean column-wise KS statistic between real and synthetic features.
        Lower is better; 0 = identical marginal distributions.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        from scipy.stats import ks_2samp

        x_synth, _ = self.sample(self.n)

        if X_real is None:
            X_real = self.scaler.inverse_transform(self.X)

        n_cols = min(X_real.shape[1], x_synth.shape[1])
        ks_stats = [
            ks_2samp(X_real[:, i], x_synth[:, i]).statistic
            for i in range(n_cols)
        ]
        return float(np.mean(ks_stats))

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

    def _dummify(self, X: np.ndarray):
        df = pd.DataFrame(X)
        before = list(df.columns)
        for i in self.cat_indexes:
            df = pd.get_dummies(df, columns=[i], drop_first=True)
        after = list(df.columns)
        return df.values, before, after

    def _clip(self, X: np.ndarray) -> np.ndarray:
        for i in self.int_indexes:
            X[:, i] = np.round(X[:, i])
        X = np.minimum(np.maximum(X, self.X_min), self.X_max)
        return X
