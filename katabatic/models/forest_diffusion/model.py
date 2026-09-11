from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as KatabaticModel

from .utils import (
    OneHotSpec,
    ScaleSpec,
    clip_extremes,
    euler_solve_flow,
    minmax_fit,
    minmax_inverse,
    minmax_transform,
    onehot_fit_transform,
    onehot_inverse,
    round_integer_columns,
)


# =============================================================================
# Regressor helpers
# =============================================================================
def _make_base_regressor(
    model: str,
    seed: int,
    n_jobs: int,
    max_depth: int,
    n_estimators: int,
    eta: float,
):
    """
    Create a tree-based regressor.

    We keep dependencies minimal:
    - default: sklearn RandomForestRegressor
    - optional: xgboost XGBRegressor if installed and model='xgboost'

    Args:
        model: regressor name ('random_forest' or 'xgboost' or 'gbrt').
        seed: random seed.
        n_jobs: parallel workers.
        max_depth: tree depth setting (if supported).
        n_estimators: number of trees (if supported).
        eta: learning rate (if supported).

    Returns:
        An unfitted regressor instance.
    """
    name = (model or "random_forest").lower()

    if name == "xgboost":
        try:
            import xgboost as xgb  # type: ignore

            return xgb.XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(eta),
                subsample=1.0,
                colsample_bytree=1.0,
                tree_method="hist",
                random_state=int(seed),
                n_jobs=int(n_jobs),
            )
        except Exception:
            # If xgboost is not available, fall back to sklearn.
            name = "random_forest"

    if name == "gbrt":
        # GradientBoostingRegressor is single-output; we will wrap it later.
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=int(seed))

    # Default: RandomForest supports multi-output in sklearn
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=max(50, int(n_estimators)),
        max_depth=int(max_depth),
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )


def _fit_multioutput(base, X: np.ndarray, Y: np.ndarray):
    """
    Fit a model for multi-output regression.

    Some regressors can natively fit Y with shape [n, p].
    If not, we wrap them using sklearn's MultiOutputRegressor.

    Args:
        base: unfitted regressor
        X: features [n, d]
        Y: targets [n, p]

    Returns:
        A fitted regressor (base or MultiOutputRegressor(base)).
    """
    try:
        base.fit(X, Y)
        pred = base.predict(X[:1])
        if np.asarray(pred).ndim == 2:
            return base
    except Exception:
        pass

    from sklearn.multioutput import MultiOutputRegressor

    mo = MultiOutputRegressor(base)
    mo.fit(X, Y)
    return mo


# =============================================================================
# Katabatic Model
# =============================================================================
class ForestDiffusion(KatabaticModel):
    """
    ForestDiffusion: a tree-based diffusion/flow-matching generator for tabular data.

    Katabatic Integration:
        - Reads {dataset_dir}/x_train.csv and {dataset_dir}/y_train.csv (if present)
        - Trains internal time-step regressors
        - Writes {synthetic_dir}/x_synth.csv and {synthetic_dir}/y_synth.csv

    High-level method:
        - For each time t: train regressor to predict v = (x_real - x_noise)
          from inputs (x_t, t), where x_t is a mixture of noise and real.
        - During sampling: start from noise and integrate dy/dt = f(y,t) forward.
    """

    def __init__(
        self,
        n_t: int = 50,
        duplicate_K: int = 50,
        eps: float = 1e-3,
        regressor: str = "random_forest",
        max_depth: int = 7,
        n_estimators: int = 200,
        eta: float = 0.1,
        cat_indexes: list[int] | None = None,
        int_indexes: list[int] | None = None,
        bin_indexes: list[int] | None = None,
        seed: int = 666,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Args:
            n_t: number of discrete time models (and also Euler steps).
            duplicate_K: how many noise replicates per training row.
            eps: smallest time value (avoid exactly t=0).
            regressor: 'random_forest', 'xgboost' (if installed), or 'gbrt'.
            max_depth, n_estimators, eta: regressor hyperparameters.
            cat_indexes: columns to one-hot encode (optional).
            int_indexes: columns to round to integers after generation.
            bin_indexes: binary columns (also rounded).
            seed: RNG seed.
            n_jobs: parallelism for supported regressors.

        Note:
            kwargs is accepted to match Katabatic model instantiation patterns,
            but unused here unless you extend the model.
        """
        super().__init__()
        self.n_t = int(n_t)
        self.duplicate_K = int(max(1, duplicate_K))
        self.eps = float(eps)

        self.regressor = str(regressor)
        self.max_depth = int(max_depth)
        self.n_estimators = int(n_estimators)
        self.eta = float(eta)

        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.rng = np.random.default_rng(self.seed)

        self.cat_indexes = list(cat_indexes or [])
        self.bin_indexes = list(bin_indexes or [])
        self.int_indexes = list(set(list(int_indexes or []) + self.bin_indexes))

        # Learned objects
        self.models_: list[Any] = []
        self.t_levels: np.ndarray | None = None

        # Transform specs
        self.original_dim: int | None = None
        self.onehot_spec: OneHotSpec | None = None
        self.scale_spec: ScaleSpec | None = None
        self.true_min: np.ndarray | None = None
        self.true_max: np.ndarray | None = None

        # Saving helpers
        self.x_columns: list[str] | None = None
        self.y_name: str | None = None
        self.y_empirical: np.ndarray | None = None
        self.y_probs: np.ndarray | None = None

        # Speed knobs
        self.n_epochs = int(kwargs.get("n_epochs", 3))
        self.pairs_per_epoch = int(kwargs.get("pairs_per_epoch", 50000))
        self.max_train_rows = kwargs.get("max_train_rows", 30000)
        self.n_steps = int(kwargs.get("n_steps", self.n_t))

        # single-model approach
        self.single_model = True

        # Print frequency
        self.print_every_epoch = True

    def _p(self, msg: str) -> None:
        if getattr(self, "verbose", True):
            print(f"[ForestDiffusion] {msg}", flush=True)

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """
        Katabatic uses this to check/install dependencies.

        Returns:
            List of minimal packages required to run.
        """
        return ["numpy", "pandas", "sklearn"]  # xgboost optional

    # -------------------------------------------------------------------------
    # Katabatic entrypoint: train
    # -------------------------------------------------------------------------
    def train(self, dataset_dir: str, synthetic_dir: str, **kwargs) -> ForestDiffusion:
        self._p(
            f"train() START | dataset_dir={dataset_dir} | synthetic_dir={synthetic_dir}"
        )

        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        # -------------------------
        # Load training data
        # -------------------------
        self._p(f"Loading x_train.csv: {x_train_path}")
        X_df = pd.read_csv(x_train_path)
        self.x_columns = list(X_df.columns)
        X = X_df.values.astype(float)
        self._p(f"Loaded X | shape={X.shape}")

        y_arr = None
        if os.path.exists(y_train_path):
            self._p(f"Loading y_train.csv: {y_train_path}")
            y_df = pd.read_csv(y_train_path)
            self.y_name = y_df.columns[0]
            y_arr = y_df.values.ravel()
            uniq, counts = np.unique(y_arr, return_counts=True)
            self.y_empirical = uniq
            self.y_probs = counts / counts.sum()
            self._p(f"Loaded y | shape={y_arr.shape} | unique={len(uniq)}")
        else:
            self._p("No y_train.csv found -> will write dummy y_synth.csv")

        # -------------------------
        # Optional cap for speed
        # -------------------------
        max_train_rows = kwargs.get(
            "max_train_rows", getattr(self, "max_train_rows", None)
        )
        if max_train_rows is not None and X.shape[0] > int(max_train_rows):
            self._p(f"Speed cap: sampling {max_train_rows} rows from {X.shape[0]}")
            idx = self.rng.choice(X.shape[0], size=int(max_train_rows), replace=False)
            X = X[idx]
            if y_arr is not None:
                y_arr = y_arr[idx]
            self._p(f"Capped X | shape={X.shape}")

        # -------------------------
        # Save original ranges for clipping
        # -------------------------
        self.true_min = np.nanmin(X, axis=0, keepdims=True)
        self.true_max = np.nanmax(X, axis=0, keepdims=True)
        self._p("Computed per-column min/max for clipping")

        # -------------------------
        # Encode + scale
        # -------------------------
        self.original_dim = X.shape[1]
        self._p(f"One-hot encode | cat_indexes={self.cat_indexes}")
        X_enc, self.onehot_spec = onehot_fit_transform(X, self.cat_indexes)
        self._p(f"Encoded X | shape={X_enc.shape}")

        self._p("Fit + apply MinMax scaling to [-1,1]")
        self.scale_spec = minmax_fit(X_enc)
        X_s = minmax_transform(X_enc, self.scale_spec, feature_range=(-1.0, 1.0))
        self._p(f"Scaled X | shape={X_s.shape}")

        # -------------------------
        # Build training set by resampling noisy pairs (FAST)
        # We do NOT duplicate whole dataset.
        # -------------------------
        n_epochs = int(kwargs.get("n_epochs", getattr(self, "n_epochs", 3)))
        pairs_per_epoch = int(
            kwargs.get("pairs_per_epoch", getattr(self, "pairs_per_epoch", 50000))
        )

        self._p(
            f"Training single regressor | n_epochs={n_epochs} | pairs_per_epoch={pairs_per_epoch}"
        )

        # We train v = x_real - x_noise from inputs (x_t, t)
        # where x_t = (1-t)*x_noise + t*x_real
        feat_list = []
        targ_list = []

        n_real = X_s.shape[0]
        p_out = X_s.shape[1]

        for ep in range(n_epochs):
            self._p(f"Epoch {ep + 1}/{n_epochs} | sampling training pairs...")

            # sample real rows with replacement
            ridx = self.rng.integers(low=0, high=n_real, size=pairs_per_epoch)
            X1 = X_s[ridx]  # [pairs, p_out]

            # sample noise and time
            X0 = self.rng.normal(size=(pairs_per_epoch, p_out))
            t = self.rng.uniform(low=self.eps, high=1.0, size=(pairs_per_epoch, 1))

            Xt = (1.0 - t) * X0 + t * X1
            V = X1 - X0

            # features = [Xt, t]
            X_feat = np.concatenate([Xt, t], axis=1)

            feat_list.append(X_feat)
            targ_list.append(V)

            self._p(f"Epoch {ep + 1}: built X_feat={X_feat.shape}, V={V.shape}")

        X_train = np.vstack(feat_list)
        Y_train = np.vstack(targ_list)
        self._p(
            f"Final training matrix | X_train={X_train.shape} | Y_train={Y_train.shape}"
        )

        # -------------------------
        # Fit one multi-output tree model
        # -------------------------
        self._p(f"Fitting regressor='{self.regressor}' (single model)")
        base = _make_base_regressor(
            model=self.regressor,
            seed=self.seed,
            n_jobs=self.n_jobs,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            eta=self.eta,
        )
        self.model_ = _fit_multioutput(base, X_train, Y_train)
        self._p("Regressor fitted")

        self.is_fitted = True

        # -------------------------
        # Generate synthetic
        # -------------------------
        n_steps = int(kwargs.get("n_steps", getattr(self, "n_steps", self.n_t)))
        self._p(f"Generating synthetic | n_rows={X_s.shape[0]} | n_steps={n_steps}")

        X_synth_scaled = self._generate_scaled_single(
            batch_size=X_s.shape[0], n_steps=n_steps
        )
        self._p(f"Generated scaled synth | shape={X_synth_scaled.shape}")

        self._p("Decoding + postprocess")
        X_synth = self._decode_postprocess(X_synth_scaled)
        self._p(f"Decoded synth | shape={X_synth.shape}")

        # -------------------------
        # Write outputs
        # -------------------------
        os.makedirs(synthetic_dir, exist_ok=True)
        out_x = os.path.join(synthetic_dir, "x_synth.csv")
        pd.DataFrame(X_synth, columns=self.x_columns).to_csv(out_x, index=False)
        self._p(f"Wrote {out_x}")

        out_y = os.path.join(synthetic_dir, "y_synth.csv")
        if (
            y_arr is not None
            and self.y_empirical is not None
            and self.y_probs is not None
        ):
            y_synth = self.rng.choice(
                self.y_empirical, size=X_synth.shape[0], p=self.y_probs
            )
            pd.DataFrame(y_synth, columns=[self.y_name]).to_csv(out_y, index=False)
            self._p(f"Wrote {out_y} (empirical resample)")
        else:
            pd.DataFrame(np.zeros((X_synth.shape[0], 1)), columns=["y"]).to_csv(
                out_y, index=False
            )
            self._p(f"Wrote {out_y} (dummy)")

        self._p("train() END")
        return self

    def evaluate(self, *args, **kwargs) -> float:
        """
        Katabatic evaluation is typically handled by pipeline evaluators (TSTR, etc.).
        This method is kept for interface compatibility.
        """
        return 0.0

    def sample(
        self,
        n: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows as a named pandas DataFrame."""
        if not getattr(self, "is_fitted", False):
            raise RuntimeError("Model is not trained. Call train() before sample().")

        if self.x_columns is None:
            raise RuntimeError("Training column names are missing.")

        if n <= 0:
            raise ValueError("n must be greater than zero.")

        sampling_rng = np.random.default_rng(seed) if seed is not None else self.rng

        x_scaled = self._generate_scaled_single(
            batch_size=int(n),
            n_steps=self.n_steps,
            rng=sampling_rng,
        )

        generated_values = self._decode_postprocess(x_scaled)

        return pd.DataFrame(
            generated_values,
            columns=self.x_columns,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _predict_v_single(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Predict v using the single fitted regressor (trained with (x_t, t) features).
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Single regressor not fitted (train first).")

        tcol = np.full((y.shape[0], 1), float(t), dtype=float)
        X_feat = np.concatenate([y, tcol], axis=1)
        pred = self.model_.predict(X_feat)
        return np.asarray(pred, dtype=float)

    def _generate_scaled_single(
        self,
        batch_size: int,
        n_steps: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate samples using Euler flow integration.

        Args:
            batch_size: Number of rows to generate.
            n_steps: Number of Euler integration steps.
            rng: Optional NumPy random generator.

        Returns:
            Generated samples in scaled and encoded space.
        """
        self._p(
            "_generate_scaled_single() START | "
            f"batch_size={batch_size} | n_steps={n_steps}"
        )

        if self.onehot_spec is None:
            raise RuntimeError("OneHotSpec is missing. Train the model first.")

        if not hasattr(self, "model_"):
            raise RuntimeError("Regressor is missing. Train the model first.")

        sampling_rng = rng if rng is not None else self.rng

        output_dimension = int(self.onehot_spec.out_dim)

        initial_noise = sampling_rng.normal(size=(int(batch_size), output_dimension))

        def velocity_function(
            current_values: np.ndarray,
            time_value: float,
        ) -> np.ndarray:
            return self._predict_v_single(
                current_values,
                time_value,
            )

        generated_values = euler_solve_flow(
            velocity_function,
            initial_noise,
            n_steps=int(n_steps),
        )

        self._p("_generate_scaled_single() END")

        return generated_values

    def _decode_postprocess(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert generated samples from scaled/encoded space back to original space.

        Steps:
            1) inverse min-max scale
            2) invert one-hot to restore original columns
            3) round integer/binary columns
            4) clip to training min/max bounds

        Args:
            X_scaled: array [n, p_out]

        Returns:
            Array [n, p_original]
        """
        if (
            self.scale_spec is None
            or self.onehot_spec is None
            or self.original_dim is None
        ):
            raise RuntimeError("Transform specs missing (train first).")
        if self.true_min is None or self.true_max is None:
            raise RuntimeError("Clipping ranges missing (train first).")

        X_enc = minmax_inverse(X_scaled, self.scale_spec, feature_range=(-1.0, 1.0))
        X_rec = onehot_inverse(X_enc, self.original_dim, self.onehot_spec)

        # Enforce integer/binary columns
        X_rec = round_integer_columns(X_rec, self.int_indexes)

        # Keep values within observed ranges
        X_rec = clip_extremes(X_rec, self.true_min, self.true_max)
        return X_rec
