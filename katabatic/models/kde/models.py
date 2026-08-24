from __future__ import annotations

import json
import os

import pandas as pd

from katabatic.models.base_model import Model as BaseModel

from .kde_core import KDEModel


class KDESynthesizer(BaseModel):
    """
    Class-conditional KDE / histogram synthesizer for tabular data.

    Continuous features are modeled with a 1D Gaussian KDE per class;
    categorical features with class-conditional histograms; the class
    distribution itself is matched to the real data. No training loop,
    no GPU — fits in closed form via ``sklearn.neighbors.KernelDensity``.

    Ported from the katabatic-mentorship registry (Rishi_Goyal branch)
    and adapted to Katabatic's ``Model`` interface and artifact I/O
    conventions (see ``katabatic/models/ctgan/models.py`` for the same
    pattern in a different model).
    """

    def __init__(
        self,
        *,
        kernel: str = "gaussian",
        bandwidth: float | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cfg = {"kernel": kernel, "bandwidth": bandwidth, "seed": seed}
        self._kde: KDEModel | None = None
        self._target_col: str | None = None
        self._feature_cols: list[str] = []

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sklearn"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> KDESynthesizer:
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
            y_col = df.columns[-1]
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. Expected train_full.csv or x_train.csv/y_train.csv."
                )
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError(
                    "y_train.csv must have exactly one column (the target)."
                )
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self._target_col = str(y_col)
        self._feature_cols = [c for c in df.columns if c != self._target_col]

        categorical_cols = self._load_categorical_cols(data_dir)

        self._kde = KDEModel(
            target_col=self._target_col,
            categorical_cols=categorical_cols,
            kernel=self.cfg["kernel"],
            bandwidth=self.cfg["bandwidth"],
            random_state=self.cfg["seed"],
        )
        self._kde.fit(df)
        self.is_fitted = True

        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(
                os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "kde")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n=len(df))
        x_synth = df_s[self._feature_cols].copy()
        y_synth = df_s[[self._target_col]].copy()

        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        meta = {
            "schema": {
                "columns": df.columns.tolist(),
                "label": self._target_col,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "categorical_columns": sorted(categorical_cols) if categorical_cols else [],
            },
            "training": self.cfg,
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[KDE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
        )
        return self

    def _load_categorical_cols(self, data_dir: str) -> set[str] | None:
        """
        Prefer Katabatic's own ``info.json`` (``cat_col_idx``) when present.
        Pipeline data is often already integer-encoded, so dtype alone can't
        tell categorical codes apart from real continuous values.
        """
        info_path = os.path.join(data_dir, "info.json")
        if not os.path.exists(info_path):
            return None

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        cat_idx = info.get("cat_col_idx")
        if cat_idx is None:
            return None

        return {
            self._feature_cols[i]
            for i in cat_idx
            if i < len(self._feature_cols)
        }

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(self, n: int | None = None, *args, **kwargs) -> pd.DataFrame:
        if not self.is_fitted or self._kde is None:
            raise RuntimeError("Call train() before sample().")
        n_rows = int(n) if n is not None else 1000
        return self._kde.generate(n_rows)
