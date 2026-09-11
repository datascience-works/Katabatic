from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScaleSpec:
    data_min: np.ndarray
    data_max: np.ndarray


def _nanminmax(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min = np.nanmin(X, axis=0)
    x_max = np.nanmax(X, axis=0)
    same = (x_max - x_min) == 0
    x_max = x_max.copy()
    x_min = x_min.copy()
    x_max[same] = x_min[same] + 1.0
    return x_min, x_max


def minmax_fit(X: np.ndarray) -> ScaleSpec:
    x_min, x_max = _nanminmax(X)
    return ScaleSpec(data_min=x_min, data_max=x_max)


def minmax_transform(
    X: np.ndarray, spec: ScaleSpec, feature_range=(-1.0, 1.0)
) -> np.ndarray:
    lo, hi = feature_range
    X01 = (X - spec.data_min) / (spec.data_max - spec.data_min)
    return X01 * (hi - lo) + lo


def minmax_inverse(
    Xn: np.ndarray, spec: ScaleSpec, feature_range=(-1.0, 1.0)
) -> np.ndarray:
    lo, hi = feature_range
    X01 = (Xn - lo) / (hi - lo)
    return X01 * (spec.data_max - spec.data_min) + spec.data_min


# -------------------------
# One-hot (optional)
# -------------------------
@dataclass
class OneHotSpec:
    cat_indexes: list[int]
    slices: dict[int, tuple[int, int]]  # original col -> slice in expanded
    categories: dict[int, np.ndarray]  # original col -> sorted unique values
    out_dim: int


def onehot_fit_transform(
    X: np.ndarray, cat_indexes: list[int]
) -> tuple[np.ndarray, OneHotSpec]:
    if not cat_indexes:
        spec = OneHotSpec(cat_indexes=[], slices={}, categories={}, out_dim=X.shape[1])
        return X.astype(float), spec

    cat_set = set(cat_indexes)
    blocks: list[np.ndarray] = []
    slices: dict[int, tuple[int, int]] = {}
    categories: dict[int, np.ndarray] = {}
    out_col = 0

    for j in range(X.shape[1]):
        col = X[:, j]
        if j in cat_set:
            col2 = col.copy()
            nan_mask = np.isnan(col2)
            if nan_mask.any():
                fill_code = (np.nanmin(col2[~nan_mask]) - 1) if (~nan_mask).any() else 0
                col2[nan_mask] = fill_code
            uniq = np.unique(col2.astype(int))
            categories[j] = uniq
            oh = np.zeros((X.shape[0], len(uniq)), dtype=float)
            for k, v in enumerate(uniq):
                oh[:, k] = (col2.astype(int) == int(v)).astype(float)
            slices[j] = (out_col, out_col + oh.shape[1])
            out_col += oh.shape[1]
            blocks.append(oh)
        else:
            blocks.append(col.reshape(-1, 1).astype(float))
            out_col += 1

    X_out = np.concatenate(blocks, axis=1)
    spec = OneHotSpec(
        cat_indexes=list(cat_indexes),
        slices=slices,
        categories=categories,
        out_dim=X_out.shape[1],
    )
    return X_out, spec


def onehot_inverse(X_oh: np.ndarray, original_dim: int, spec: OneHotSpec) -> np.ndarray:
    if not spec.cat_indexes:
        return X_oh

    X_rec = np.zeros((X_oh.shape[0], original_dim), dtype=float)
    cat_set = set(spec.cat_indexes)

    taken = np.zeros((spec.out_dim,), dtype=bool)
    for j in spec.cat_indexes:
        s, e = spec.slices[j]
        taken[s:e] = True
    non_cat_cols = np.where(~taken)[0]

    # categorical
    for j in spec.cat_indexes:
        s, e = spec.slices[j]
        block = X_oh[:, s:e]
        idx = np.argmax(block, axis=1)
        X_rec[:, j] = spec.categories[j][idx]

    # continuous / non-cat
    k = 0
    for j in range(original_dim):
        if j in cat_set:
            continue
        X_rec[:, j] = X_oh[:, non_cat_cols[k]]
        k += 1

    return X_rec


# -------------------------
# Postprocess
# -------------------------
def round_integer_columns(X: np.ndarray, int_indexes: list[int]) -> np.ndarray:
    if not int_indexes:
        return X
    X2 = X.copy()
    for j in int_indexes:
        X2[:, j] = np.round(X2[:, j])
    return X2


def clip_extremes(X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return np.clip(X, x_min.reshape(1, -1), x_max.reshape(1, -1))


# -------------------------
# Time + solver
# -------------------------
def time_grid(n_t: int, eps: float) -> np.ndarray:
    return np.linspace(eps, 1.0, num=int(n_t), dtype=float)


def euler_solve_flow(f, y0: np.ndarray, n_steps: int) -> np.ndarray:
    y = y0.copy()
    steps = max(1, int(n_steps))
    h = 1.0 / steps
    for k in range(steps):
        t = k * h
        y = y + h * f(y, t)
    return y
