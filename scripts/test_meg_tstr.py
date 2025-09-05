# D:\PROJECT MEG\scripts\test_meg_tstr.py

import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

# --- ensure local "katabatic" is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from katabatic.models.meg import MEG


# ----------------------------
# Robust discretizer
# ----------------------------
def discretize_dataframe(df: pd.DataFrame, target_col: str, n_bins_default: int = 30):
    df = df.copy()
    rng = np.random.RandomState(42)

    # 1) Target to codes
    df[target_col] = df[target_col].astype("category").cat.codes
    y = df[target_col].to_numpy()

    feats = df.drop(columns=[target_col])
    num_cols = feats.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in feats.columns if c not in num_cols]

    # 2) Encode categoricals
    cat_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        feats[c] = le.fit_transform(feats[c].astype(str))
        cat_encoders[c] = le

    # 3) Adaptive binning with tie-safe jitter
    bin_edges = {}
    for c in num_cols:
        col = feats[c].astype(float).to_numpy()
        mask = ~np.isnan(col)
        uniq = np.unique(col[mask])
        n_unique = len(uniq)

        if n_unique <= 2:
            le = LabelEncoder()
            feats[c] = le.fit_transform(np.nan_to_num(col, nan=uniq[0] if n_unique else 0.0))
            cmin, cmax = np.nanmin(col), np.nanmax(col)
            edges = np.array([cmin, cmax]) if cmin != cmax else np.array([cmin, cmin + 1e-6])
            bin_edges[c] = edges
            continue

        vals, counts = np.unique(col[mask], return_counts=True)
        top_frac = counts.max() / counts.sum()
        if top_frac >= 0.9:
            n_bins_col = min(8, n_unique - 1)
        else:
            n_bins_col = min(n_bins_default, n_unique - 1, 50)

        col_work = col.copy()
        ties_ratio = 1.0 - (n_unique / max(1, mask.sum()))
        if ties_ratio > 0.5:
            jitter = rng.normal(0.0, 1e-9, size=col_work.shape)
            col_work[mask] = col_work[mask] + jitter[mask]

        try:
            kb = KBinsDiscretizer(n_bins=n_bins_col, encode="ordinal", strategy="quantile")
            feats[[c]] = kb.fit_transform(col_work.reshape(-1, 1))
            edges = kb.bin_edges_[0]
            if np.any(np.diff(edges) <= 1e-12):
                raise RuntimeError("Zero-width edges")
        except Exception:
            qs = np.linspace(0, 1, num=n_bins_col + 1)
            edges = np.quantile(col_work[mask], qs)
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([np.nanmin(col_work), np.nanmax(col_work) + 1e-6])
            feats[c] = np.clip(np.digitize(col_work, edges[1:-1], right=True), 0, len(edges) - 2)

        bin_edges[c] = edges

    feats = feats.astype(np.int64)
    X_disc = feats.to_numpy()
    cardinalities = [int(feats[c].max()) + 1 for c in feats.columns]

    def inv_num(col_name: str, bin_idx: np.ndarray) -> np.ndarray:
        edges = bin_edges[col_name]
        mids = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(bin_idx.astype(int), 0, len(mids) - 1)
        return mids[idx]

    meta = {
        "columns": feats.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_encoders": cat_encoders,
        "bin_edges": bin_edges,
        "cardinalities": cardinalities,
        "inv_num": inv_num,
    }
    return X_disc, y, meta


# ----------------------------
# Inverse helper
# ----------------------------
def inverse_discretized_to_continuous(X_disc: np.ndarray, meta):
    Xr = np.zeros_like(X_disc, dtype=np.float64)
    cols = meta["columns"]
    num_set = set(meta["num_cols"])
    for j, c in enumerate(cols):
        if c in num_set:
            Xr[:, j] = meta["inv_num"](c, X_disc[:, j])
        else:
            Xr[:, j] = X_disc[:, j].astype(np.float64)
    return Xr


# ----------------------------
# Privacy metric (DCR)
# ----------------------------
def dcr(real_cont: np.ndarray, synth_cont: np.ndarray, k: int = 1):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(real_cont)
    dists, _ = nn.kneighbors(synth_cont, return_distance=True)
    return float(np.mean(dists[:, 0])), float(np.min(dists[:, 0]))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # File chooser
    root = tk.Tk()
    root.withdraw()
    print(" Please choose your dataset CSV file...")
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not file_path:
        raise FileNotFoundError(" No file selected")
    df = pd.read_csv(file_path)

    print(f"\n Selected file: {file_path}")
    print(f"Loaded dataset with shape {df.shape}\n")

    print("Available columns:", list(df.columns))
    target_col = input("Enter the target column name: ").strip()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset")

    # Auto-sample very large datasets
    MAX_ROWS_FOR_TRAIN = 200_000
    if len(df) > MAX_ROWS_FOR_TRAIN:
        print(f"ℹ️ Dataset is large ({len(df):,}). Sampling {MAX_ROWS_FOR_TRAIN:,} rows for training.")
        df = df.sample(n=MAX_ROWS_FOR_TRAIN, random_state=42).reset_index(drop=True)

    # Discretize
    X_disc, y, meta = discretize_dataframe(df, target_col, n_bins_default=30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_disc, y, test_size=0.2, stratify=y, random_state=42
    )

    # Epoch schedule
    epochs = 50 if len(df) > 30_000 else 100
    if len(df) > 300_000:
        epochs = 30
    print(f"\n Training MEG for {epochs} epochs (rows={len(df)})..")

    # Train MEG
    G = MEG(epochs=epochs, batch_size=512, lr=1e-3, beta=1.0, cardinalities=meta["cardinalities"])
    G.fit(X_train)

    # Sample synthetic
    X_synth = G.sample(len(X_train))
    y_synth = y_train[:len(X_synth)]

    # TSTR
    print("\nTSTR Results (Train on Synthetic, Test on Real):")
    clfs = {
        "Logistic Regression": (LogisticRegression(max_iter=2000), "scaled"),
        "MLP": (MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=400, early_stopping=True), "scaled"),
        "Random Forest": (RandomForestClassifier(n_estimators=500, n_jobs=-1), "raw"),
        "XGBoost": (XGBClassifier(n_estimators=500, tree_method="hist", eval_metric="logloss", n_jobs=-1), "raw"),
    }

    scaler = StandardScaler()
    Xsyn_cont = inverse_discretized_to_continuous(X_synth, meta)
    Xte_cont = inverse_discretized_to_continuous(X_test, meta)
    Xsyn_scaled = scaler.fit_transform(Xsyn_cont)
    Xte_scaled = scaler.transform(Xte_cont)

    for name, (clf, view) in clfs.items():
        if view == "scaled":
            clf.fit(Xsyn_scaled, y_synth)
            yhat = clf.predict(Xte_scaled)
        else:
            clf.fit(Xsyn_cont, y_synth)
            yhat = clf.predict(Xte_cont)
        acc = accuracy_score(y_test, yhat)
        f1w = f1_score(y_test, yhat, average="weighted")
        print(f"{name:20s} | Accuracy={acc:.3f} | F1={f1w:.3f}")

    # Statistical similarity
    print("\nStatistical Similarity (JSD + WD):")
    jsds, wds = [], []
    for j, col in enumerate(meta["columns"]):
        r = X_test[:, j]
        s = X_synth[:, j]
        maxk = int(max(r.max(), s.max()))
        r_hist, _ = np.histogram(r, bins=maxk + 1, range=(0, maxk + 1), density=True)
        s_hist, _ = np.histogram(s, bins=maxk + 1, range=(0, maxk + 1), density=True)
        jsd = jensenshannon(r_hist + 1e-12, s_hist + 1e-12) ** 2

        if col in meta["num_cols"]:
            r_vals = meta["inv_num"](col, r)
            s_vals = meta["inv_num"](col, s)
            all_vals = np.concatenate([r_vals, s_vals])
            m, sd = all_vals.mean(), all_vals.std() + 1e-12
            r_std = (r_vals - m) / sd
            s_std = (s_vals - m) / sd
            wd = wasserstein_distance(r_std, s_std)
        else:
            wd = wasserstein_distance(r, s)

        jsds.append(jsd)
        wds.append(wd)
        print(f"{col:20s} | JSD={jsd:.3f} | WD={wd:.3f}")

    print(f"\nAverage JSD={np.mean(jsds):.3f} | Average WD={np.mean(wds):.3f}")

    # Privacy (DCR)
    sc_priv = StandardScaler()
    real_cont = sc_priv.fit_transform(inverse_discretized_to_continuous(X_disc, meta))
    synth_cont = sc_priv.transform(inverse_discretized_to_continuous(X_synth, meta))
    dcr_mean, dcr_min = dcr(real_cont, synth_cont)
    print(f"\n Privacy (DCR): mean={dcr_mean:.3f} | min={dcr_min:.3f}")

    # Save synthetic
    base = os.path.splitext(os.path.basename(file_path))[0]
    disc_path = f"synthetic_{base}.csv"
    cont_path = f"synthetic_{base}_continuous.csv"

    synth_df = pd.DataFrame(X_synth, columns=meta["columns"])
    synth_df[target_col] = y_synth
    synth_df.to_csv(disc_path, index=False)

    synth_cont_df = pd.DataFrame(Xsyn_cont, columns=meta["columns"])
    synth_cont_df[target_col] = y_synth
    synth_cont_df.to_csv(cont_path, index=False)

    print(f"\n Synthetic data saved to: {disc_path}")
    print(f" Continuous-view synthetic saved to: {cont_path}")
