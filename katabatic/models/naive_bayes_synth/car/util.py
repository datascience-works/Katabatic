"""
Helper functions for NaiveBayesSynth: encoding categorical columns and
building/sampling from per-class conditional probability tables.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def encode_features(x: pd.DataFrame) -> tuple[np.ndarray, OrdinalEncoder]:
    """Ordinal-encode all feature columns. Returns the encoded array and the
    fitted encoder (needed later to invert sampled data back to original
    category labels)."""
    encoder = OrdinalEncoder(
        dtype=int, handle_unknown="use_encoded_value", unknown_value=-1
    )
    x_enc = encoder.fit_transform(x)
    return x_enc, encoder


def build_conditional_tables(
    x_enc: np.ndarray, y_enc: np.ndarray, num_classes: int, smoothing: float = 1.0
) -> list[np.ndarray]:
    """
    For each feature column, build a (num_classes x num_categories) table of
    P(feature=category | class), with Laplace smoothing so unseen combinations
    still get non-zero probability.

    Returns
    -------
    list of np.ndarray, one per feature column, each shaped (num_classes, num_categories)
    """
    n_features = x_enc.shape[1]
    tables = []
    for col in range(n_features):
        num_categories = int(x_enc[:, col].max()) + 1
        table = np.full((num_classes, num_categories), smoothing, dtype=float)
        for cls in range(num_classes):
            mask = y_enc == cls
            col_vals = x_enc[mask, col]
            for cat in range(num_categories):
                table[cls, cat] += np.sum(col_vals == cat)
            table[cls] /= table[cls].sum()
        tables.append(table)
    return tables


def sample_from_tables(
    class_priors: np.ndarray,
    conditional_tables: list[np.ndarray],
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample `size` synthetic rows: first sample a class per row from the class
    priors, then sample each feature independently conditioned on that class.

    Returns
    -------
    x_synth : np.ndarray of shape (size, n_features), ordinally-encoded
    y_synth : np.ndarray of shape (size,), encoded class labels
    """
    num_classes = len(class_priors)
    y_synth = rng.choice(num_classes, size=size, p=class_priors)

    n_features = len(conditional_tables)
    x_synth = np.zeros((size, n_features), dtype=int)
    for col, table in enumerate(conditional_tables):
        for cls in range(num_classes):
            row_idxs = np.where(y_synth == cls)[0]
            if len(row_idxs) == 0:
                continue
            num_categories = table.shape[1]
            x_synth[row_idxs, col] = rng.choice(
                num_categories, size=len(row_idxs), p=table[cls]
            )
    return x_synth, y_synth
