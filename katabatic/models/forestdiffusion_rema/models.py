"""
ForestDiffusion model wrapper for Katabatic.

This implementation is aligned with the official Samsung SAIL ForestDiffusion
repository and adapts it cleanly into the Katabatic framework.

- No changes to official utils_diffusion.py
- No unsupported helper functions
- Uses VPSDE + PC sampler only
"""

import copy
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from katabatic.models.forestdiffusion.utils.diffusion import VPSDE, get_pc_sampler

# optional libraries
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


class ForestDiffusionModel:
    """
    ForestDiffusion model using tree regressors as score networks.

    Parameters follow the official ForestDiffusion implementation.
    This wrapper only adapts the API to Katabatic.
    """

    def __init__(
        self,
        X,
        label_y=None,
        n_t=50,
        model="xgboost",        
        max_depth=7,
        n_estimators=100,
        eta=0.3,
        tree_method="hist",
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=1.0,
        num_leaves=31,
        cat_indexes=None,
        int_indexes=None,
        beta_min=0.1,
        beta_max=8.0,
        eps=1e-3,
        gpu_hist=False,
        seed=666,
        **xgboost_kwargs,
    ):

        if cat_indexes is None:
            cat_indexes = []
        if int_indexes is None:
            int_indexes = []

        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

        np.random.seed(seed)

        self.seed = seed
        self.model_type = model
        self.n_t = n_t
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps
        self.cat_indexes = list(cat_indexes)
        self.int_indexes = list(int_indexes)

        #  Remove rows fully missing 
        mask_valid = ~np.isnan(X).all(axis=1)
        X = X[mask_valid]
        if label_y is not None:
            label_y = label_y[mask_valid]

        #  Save min/max for clipping 
        self.X_min = np.nanmin(X, axis=0, keepdims=True)
        self.X_max = np.nanmax(X, axis=0, keepdims=True)

        #  One-hot encode categoricals 
        if len(self.cat_indexes) > 0:
            X, self.cols_before, self.cols_after = self._dummify(X)

        #  Scale to [-1, 1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X = self.scaler.fit_transform(X)

        self.X = X
        self.n, self.c = X.shape

        #  Label conditioning 
        self.label_y = label_y
        if label_y is not None:
            self.y_vals, self.y_probs = np.unique(label_y, return_counts=True)
            self.y_probs = self.y_probs / self.y_probs.sum()
        else:
            self.y_vals = np.array([0])
            self.y_probs = np.array([1.0])

        # Build SDE
        self.sde = VPSDE(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            N=self.n_t,
        )

        #  Train regressors
        self.regr = []
        for _ in self.y_vals:
            self.regr.append([
                self._train_regressor()
                for _ in range(self.n_t)
            ])

    
    # Training
    

    def _train_regressor(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=self.seed,
            )

        if self.model_type == "lgbm":
            if LGBMRegressor is None:
                raise ImportError("lightgbm not installed")
            return LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                random_state=self.seed,
            )

        if self.model_type == "catboost":
            if CatBoostRegressor is None:
                raise ImportError("catboost not installed")
            return CatBoostRegressor(
                iterations=100,
                max_depth=7,
                loss_function="RMSE",
                silent=True,
                random_seed=self.seed,
            )

        # default: xgboost
        return xgb.XGBRegressor(
            n_estimators=100,
            objective="reg:squarederror",
            eta=0.3,
            max_depth=7,
            subsample=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            tree_method="gpu_hist" if gpu_hist else "hist",
            device="cuda" if gpu_hist else "cpu",
            seed=self.seed,
        )

    
    # Score function
   

    def _score_fn(self, t, y, mask_y):
        X = y.reshape(-1, self.c)
        out = np.zeros_like(X)

        step = int(round(t * (self.n_t - 1)))

        for j, lab in enumerate(self.y_vals):
            m = mask_y[lab]
            if m.sum() == 0:
                continue
            for k in range(self.c):
                out[m, k] = self.regr[j][step].predict(X[m])

        _, sigma = self.sde.marginal_prob(X, t)
        out = -out / sigma
        return out.reshape(-1)

    
    # Sampling
    

    def generate(self, batch_size=None):
        b = self.n if batch_size is None else batch_size
        y0 = np.random.normal(size=(b, self.c))

        labels = self.y_vals[
            np.argmax(np.random.multinomial(1, self.y_probs, size=b), axis=1)
        ]

        mask_y = {}
        for lab in self.y_vals:
            mask_y[lab] = labels == lab

        score_fn = partial(self._score_fn, mask_y=mask_y)

        sampler = get_pc_sampler(
            score_fn=score_fn,
            sde=self.sde,
            denoise=True,
            eps=self.eps,
        )

        x = sampler(y0.reshape(-1)).reshape(b, self.c)

        #  inverse transforms 
        x = self.scaler.inverse_transform(x)
        x = self._clip(x)

        if self.label_y is not None:
            x = np.concatenate([x, labels[:, None]], axis=1)

        return x

    
    # Helpers
    

    def _dummify(self, X):
        df = pd.DataFrame(X)
        before = df.columns
        for i in self.cat_indexes:
            df = pd.get_dummies(df, columns=[i], drop_first=True)
        after = df.columns
        return df.values, before, after

    def _clip(self, X):
        for i in self.int_indexes:
            X[:, i] = np.round(X[:, i])
        X = np.minimum(np.maximum(X, self.X_min), self.X_max)
        return X
