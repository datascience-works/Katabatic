# katabatic/models/medgan/medgan.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
import numpy as np
import pandas as pd


from .driver import MedganCLI
from .preprocess import to_matrix_file  

ArrayLike = Union[np.ndarray, pd.DataFrame]


class Medgan:
    """
    Katabatic-compatible adapter for mp2893/medgan.

    Public API:
      - fit(X)                       : train MedGAN on a binary/count matrix
      - sample(n=None) -> np.ndarray : generate synthetic rows (n=None keeps full)
      - evaluate(real, label_col=...) -> pd.DataFrame
           TSTR (Train on Synthetic, Test on Real) with common classifiers.
    """

    def __init__(
        self,
        repo_root: str = ".",
        run_dir: Optional[str] = None,
        data_type: str = "binary",      # or "count"
        python_exec: str = "python",
        n_pretrain_epoch: int = 10,
        n_epoch: int = 20,
        batch_size: int = 128,
        save_max_keep: int = 10,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.run_dir = Path(run_dir or "runs/medgan").resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.data_type = data_type
        self.python_exec = python_exec
        self.n_pretrain_epoch = n_pretrain_epoch
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.save_max_keep = save_max_keep

        self._X_path: Optional[Path] = None
        self._ckpt_prefix: Optional[str] = None
        self._cli = MedganCLI(repo_root=str(self.repo_root), python_exec=self.python_exec)

    # ----------------------------
    # helpers
    # ----------------------------
    def _as_ndarray(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    # ----------------------------
    # public API
    # ----------------------------
    def fit(self, X: ArrayLike) -> "Medgan":
        """Train MedGAN on X (2D array-like)."""
        X = self._as_ndarray(X)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save matrix 
        self._X_path = self.run_dir / "train.binary.matrix.npy"
        to_matrix_file(X, str(self._X_path))  # your preprocess should write this path

        run_prefix = self.run_dir / "adult_medgan"

        ckpt_prefix = self._cli.train(
            matrix_path=str(self._X_path),
            run_prefix=str(run_prefix),
            data_type=self.data_type,
            n_pretrain_epoch=self.n_pretrain_epoch,
            n_epoch=self.n_epoch,
            batch_size=self.batch_size,
            save_max_keep=self.save_max_keep,
        )
        self._ckpt_prefix = ckpt_prefix
        return self

    def sample(self, n: Optional[int] = None) -> np.ndarray:
        """Generate synthetic data with the latest trained checkpoint."""
        if self._X_path is None or self._ckpt_prefix is None:
            raise RuntimeError("Call fit(X) before sample().")

        out_path = self.run_dir / "adult.synthetic.npy"
        self._cli.sample(
            matrix_path=str(self._X_path),
            out_path=str(out_path),
            model_file=str(self._ckpt_prefix),  # prefix without extension
            n_samples=n,                        # None -> driver decides / full
            data_type=self.data_type,
        )
        X_syn = np.load(out_path, allow_pickle=True)
        X_syn = np.asarray(X_syn, dtype=np.float32)
        if n is not None and X_syn.shape[0] > n:
            X_syn = X_syn[:n]
        return X_syn

    # ----------------------------
    # TSTR evaluation
    # ----------------------------
    def evaluate(
        self,
        real_data: ArrayLike,
        label_col: Optional[int] = None,
        n_syn: Optional[int] = None,        # how many synthetic rows to use; None -> all produced
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        TSTR (Train on Synthetic, Test on Real).

        Parameters
        ----------
        real_data : array-like (n_real, n_features)
            Must include the label column.
        label_col : Optional[int]
            Index of the label column. If None, we attempt a simple heuristic.
        n_syn : Optional[int]
            Number of synthetic rows to train on (sample() argument). None keeps model output full.

        Returns
        -------
        pd.DataFrame with rows per model and columns: Accuracy, F1, AUC.
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier

        # optional XGBoost (if available)
        try:
            from xgboost import XGBClassifier  # type: ignore
            _has_xgb = True
        except Exception:
            _has_xgb = False

        R = self._as_ndarray(real_data)

        # --- label column handling
        if label_col is None:
            # heuristic: pick a column that's reasonably balanced (mean ~ 0.3..0.7) and binary-ish
            means = R.mean(axis=0)
            cand = np.where((means > 0.25) & (means < 0.75))[0]
            label_col = int(cand[-1]) if cand.size else (R.shape[1] - 1)

        X_real, y_real = R[:, np.arange(R.shape[1]) != label_col], R[:, label_col].astype(int)

        # --- synthetic generation and label split (assumes same column order)
        X_syn_full = self.sample(n=n_syn)
        if X_syn_full.shape[1] != R.shape[1]:
            # If shapes don't match, assume synthetic omitted label and abort cleanly.
            raise ValueError(
                f"Synthetic shape {X_syn_full.shape} does not match real shape {R.shape}. "
                "Ensure you trained MedGAN on the full matrix including the label column."
            )
        X_syn, y_syn = X_syn_full[:, np.arange(R.shape[1]) != label_col], X_syn_full[:, label_col].astype(int)

        # --- define models
        models: Dict[str, object] = {
            "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
            "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=random_state),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=None
            ),
        }
        if _has_xgb:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=random_state, n_jobs=0
            )

        # --- train on synthetic, test on real
        rows: List[Dict[str, float]] = []
        for name, clf in models.items():
            try:
                clf.fit(X_syn, y_syn)
                y_pred = clf.predict(X_real)

                # AUC needs probabilities if possible
                if hasattr(clf, "predict_proba"):
                    y_score = clf.predict_proba(X_real)[:, 1]
                elif hasattr(clf, "decision_function"):
                    y_score = clf.decision_function(X_real)
                else:
                    # fallback: use predictions as scores (will degrade AUC)
                    y_score = y_pred.astype(float)

                acc = float(accuracy_score(y_real, y_pred))
                f1 = float(f1_score(y_real, y_pred, zero_division=0))
                # If y is a single class in test, roc_auc_score will error -> guard
                if len(np.unique(y_real)) < 2:
                    auc = float("nan")
                else:
                    auc = float(roc_auc_score(y_real, y_score))

                rows.append({"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc})

            except Exception as e:
                rows.append({"Model": name, "Accuracy": np.nan, "F1": np.nan, "AUC": np.nan})
                print(f"[WARN] {name} failed: {e}")

        return pd.DataFrame(rows, columns=["Model", "Accuracy", "F1", "AUC"])
