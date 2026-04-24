from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from katabatic.models.base_model import Model
from .utils import infer_schema, fit_schema_stats, encode_df, decode_df


def to_numpy(X) -> Optional[np.ndarray]:
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, pd.Series):
        return X.to_numpy()
    if X is None:
        return None
    raise ValueError(f"Unsupported input type: {type(X)}")


class _TabEBMBackend:
    """NumPy-based TabEBM backend."""

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
        sgld_batch_size: int = 512,
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
            X_sur  = X_ebm[y_ebm == 1]

            for _ in range(sgld_steps):
                # Pre-generate noise for all samples so values are identical
                # to a non-batched run — only the gradient is chunked.
                noise = rng.normal(
                    0, sgld_noise_std, size=X_sgld.shape
                ).astype(np.float32)

                # Compute gradient in batches to keep the pairwise distance
                # matrix (batch, n_train, n_features) memory-bounded regardless
                # of how many synthetic samples are requested.
                for b_start in range(0, num_samples, sgld_batch_size):
                    b_end = min(b_start + sgld_batch_size, num_samples)
                    grad = self.compute_energy_gradient(
                        X_sgld[b_start:b_end], X_real, X_sur
                    )
                    X_sgld[b_start:b_end] -= sgld_step_size * grad

                X_sgld += noise

            results[f"class_{class_idx}"] = X_sgld.astype(np.float32)

        return results

    @staticmethod
    def compute_energy_gradient(X_synth, X_real, X_sur):
        eps = 1e-8

        # Local attraction: pull toward nearest real class neighbour
        diff_all = X_synth[:, None, :] - X_real[None, :, :]
        dist_all = np.sqrt(np.sum(diff_all ** 2, axis=2) + eps)

        nearest_idx = np.argmin(dist_all, axis=1)
        nearest_diff = diff_all[np.arange(len(X_synth)), nearest_idx]
        nearest_dist = dist_all[np.arange(len(X_synth)), nearest_idx][:, None]

        grad_min = nearest_diff / nearest_dist

        # Global centering: pull toward mean of real class samples
        grad_mean = np.mean(
            diff_all / dist_all[:, :, None],
            axis=1,
        )

        # Repulsion: push away from surrogate negative samples
        diff_sur = X_synth[:, None, :] - X_sur[None, :, :]
        dist_sur = np.sqrt(np.sum(diff_sur ** 2, axis=2) + eps)

        grad_repel = np.mean(
            diff_sur / dist_sur[:, :, None],
            axis=1,
        )

        # Subtracting grad_repel means X_sgld -= step * (-grad_repel)
        # = X_sgld += step * grad_repel → moves away from surrogates
        return grad_min + grad_mean - grad_repel

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
        y_ebm = np.concatenate([
            np.zeros(X.shape[0]),
            np.ones(len(X_sur)),
        ]).astype(int)

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
    def __init__(self, target_col: str = "target", config: Optional[TabEBMConfig] = None):
        super().__init__()
        self.target_col = target_col
        self.config = config or TabEBMConfig()

        self._tabebm = _TabEBMBackend(max_data_size=self.config.max_data_size)

        self.schema_ = None
        self.col_order_ = None
        self._y_train = None
        self._x_enc = None  # cached encoding, set at train time

    def train(self, output_dir: str, label_col=None, synthetic_dir=None,
              categorical_cols=None, continuous_cols=None, **kwargs):
        x_train = pd.read_csv(f"{output_dir}/x_train.csv")
        y_train_df = pd.read_csv(f"{output_dir}/y_train.csv")

        if label_col is None:
            label_col = y_train_df.columns[0]

        self.target_col = str(label_col)
        y_train = y_train_df.iloc[:, 0]

        schema = infer_schema(x_train)

        # Override auto-detected types with any explicitly provided column lists.
        if categorical_cols or continuous_cols:
            cat_set = set(categorical_cols or [])
            cont_set = set(continuous_cols or [])
            for col_meta in schema:
                if col_meta.name in cat_set:
                    col_meta.kind = "categorical"
                    if col_meta.categories is None:
                        col_meta.categories = sorted(
                            [str(v) for v in x_train[col_meta.name].dropna().unique()]
                        )
                elif col_meta.name in cont_set:
                    col_meta.kind = "continuous"
                    col_meta.categories = None

        fit_schema_stats(x_train, schema)

        self.schema_ = schema
        self.col_order_ = list(x_train.columns)
        self._y_train = y_train
        self._x_enc, _, _ = encode_df(x_train, schema)  # cache once

        self.is_fitted = True

        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
            synthetic_df = self.sample(len(x_train))
            x_cols = [c for c in synthetic_df.columns if c != self.target_col]
            synthetic_df[x_cols].to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"), index=False
            )
            synthetic_df[[self.target_col]].to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"), index=False
            )

        return self

    def sample(self, n: int, seed: int = None) -> pd.DataFrame:
        if not self.is_fitted or self.schema_ is None or self.col_order_ is None:
            raise RuntimeError("TabEBMModel not fitted. Call train() first.")

        effective_seed = seed if seed is not None else self.config.seed

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
            return pd.DataFrame(columns=self.col_order_ + [self.target_col])

        generated = self._tabebm.generate(
            X=self._x_enc,
            y=y_fact,
            num_samples=max_per_class,
            starting_point_noise_std=self.config.starting_point_noise_std,
            sgld_step_size=self.config.sgld_step_size,
            sgld_noise_std=self.config.sgld_noise_std,
            sgld_steps=self.config.sgld_steps,
            distance_negative_class=self.config.distance_negative_class,
            seed=effective_seed,
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

        x_synth = pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=self.col_order_)
        y_arr = np.concatenate(ys) if ys else np.array([])

        result = x_synth.copy()
        result[self.target_col] = y_arr

        return result

    def evaluate(self, *args, **kwargs):
        return None
