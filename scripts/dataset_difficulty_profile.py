from __future__ import annotations

import math
from importlib.resources import files
from pathlib import Path

import pandas as pd

from katabatic.datasets.profile import infer_dataset_profile

HIGH_CARDINALITY = 20  # categorical column with more unique values than this


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """NaN cells plus '?' placeholders in text columns."""
    mask = df.isna()
    for col in df.select_dtypes(exclude="number").columns:
        mask[col] = mask[col] | df[col].astype(str).str.strip().eq("?")
    return mask


def difficulty(csv_path: Path) -> dict:
    base = infer_dataset_profile(csv_path, dataset_name=csv_path.stem)
    df = pd.read_csv(csv_path)
    target = base["target_column"]

    shares = df[target].value_counts(normalize=True)
    k = len(shares)
    entropy = (
        -sum(p * math.log(p) for p in shares if p > 0) / math.log(k) if k > 1 else 0.0
    )

    miss = missing_mask(df)
    kinds = {c["name"]: c["kind"] for c in base["column_schema"]}
    features = [c for c in df.columns if c != target]
    cat_cols = [c for c in features if kinds[c] != "numeric"]
    card = {c: int(df.loc[~miss[c], c].nunique()) for c in cat_cols}

    return {
        "dataset": csv_path.stem,
        "task": base["task"],
        "rows": base["n_rows"],
        "features": base["n_features"],
        "numeric_features": len(features) - len(cat_cols),
        "categorical_features": len(cat_cols),
        "n_classes": base["n_classes"],
        "majority_baseline_acc": round(float(shares.iloc[0]), 4),
        "imbalance_ratio": round(float(shares.iloc[0] / shares.iloc[-1]), 1),
        "class_entropy_norm": round(entropy, 3),
        "missing_cells_pct": round(float(miss.to_numpy().mean() * 100), 2),
        "cols_with_missing": int(miss.any().sum()),
        "max_cardinality": max(card.values(), default=0),
        "high_card_cols": sum(v > HIGH_CARDINALITY for v in card.values()),
    }


if __name__ == "__main__":
    csvs = sorted(Path(str(files("katabatic.datasets"))).glob("*.csv"))
    out = pd.DataFrame([difficulty(p) for p in csvs])
    print(out.T.to_string(header=False))
    out.to_csv("dataset_difficulty_profile.csv", index=False)
