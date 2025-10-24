"""Implementation of Tabddpm model."""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union, Dict, Tuple, List
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from katebatic.models.base_model import Model

# Core TabDDPM pieces (from your provided codebase)
from tabddpm.model.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from tabddpm.model.modules import MLPDiffusion  # You can switch to ResNetDiffusion if needed


ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, Sequence]


class Tabddpm(Model):
    """Implementation of Tabddpm model.

    This wraps a mixed Gaussian+Multinomial diffusion model for tabular data.
    - `train(...)` fits the denoiser inside a diffusion process on (X, y).
    - `evaluate(...)` returns a scalar score (higher is better) based on
       negative diffusion loss over a held-out split or a few train batches.
    - `sample(n, ...)` generates n synthetic rows (DataFrame by default).
    """

    # --------------------------- configuration defaults ---------------------------

    _defaults = dict(
        steps=10_000,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        num_timesteps=1_000,
        gaussian_loss_type="mse",  # 'mse' or 'kl'
        scheduler="cosine",        # 'cosine' or 'linear'
        model_type="mlp",          # 'mlp' only here; extend to 'resnet' if you prefer
        d_layers=(256, 256, 256, 256),
        dropout=0.0,
        seed=42,
        eval_batches=50,           # how many batches to average for evaluate()
        use_ema=True,              # swap EMA weights into diffusion before sampling
    )

    # ------------------------------ lifecycle fields ------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # learned artifacts
        self._diffusion: Optional[GaussianMultinomialDiffusion] = None
        self._denoiser: Optional[torch.nn.Module] = None
        self._denoiser_ema: Optional[torch.nn.Module] = None

        # training metadata
        self._n_num: int = 0
        self._K: np.ndarray = np.array([0], dtype=int)   # per-categorical cardinalities; [0] => no categoricals
        self._is_classification: bool = True
        self._y_classes_: Optional[np.ndarray] = None
        self._y_le: Optional[LabelEncoder] = None

        # feature schema for roundtripping to DataFrame on sample()
        self._feature_names_num: List[str] = []
        self._feature_names_cat: List[str] = []
        self._cat_label_encoders: Dict[str, LabelEncoder] = {}

        # last seen class distribution (for sampling)
        self._class_dist: Optional[torch.Tensor] = None

        # cached training loader for evaluation
        self._train_loader_infinite = None  # generator of (x, {'y': y})

        # config snapshot
        self._cfg: Dict[str, Any] = dict(self._defaults)

        # fail fast on deps
        self.check_dependencies()

    # ----------------------------- dependency hints -------------------------------

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        # minimal runtime deps for the code used here
        return [
            "torch",
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "category_encoders",  # used by tabddpm.lib if you extend preprocessing
            "tabddpm",            # your local package providing diffusion/modules
        ]

    # --------------------------------- utilities ----------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _infer_task_type(y: np.ndarray) -> bool:
        """Return True if classification, False if regression."""
        # Heuristic: non-float OR few uniques => classification
        if not np.issubdtype(y.dtype, np.floating):
            return True
        uniq = np.unique(y)
        return uniq.size <= max(20, int(0.05 * len(y)))

    @staticmethod
    def _as_numpy(x: ArrayLike) -> np.ndarray:
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            return x.to_numpy()
        return np.asarray(x)

    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, 1) if x.ndim == 1 else x

    @staticmethod
    def _infinite_batches(dl: DataLoader):
        while True:
            for xb, yb in dl:
                yield xb, {"y": yb.long()}

    # --------------------------------- training -----------------------------------

    def train(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        cat_cols: Optional[Sequence[Union[int, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "Model":
        """Train the TabDDPM model on (X, y).

        Args:
            X: features (DataFrame or ndarray). Categorical columns should be
               provided via `cat_cols` (indices or names). If omitted, all
               non-numeric dtypes in a DataFrame are treated as categorical.
            y: targets (Series/ndarray). For classification, integers or strings
               are supported (strings will be label-encoded).
            cat_cols: optional positions or names identifying categorical features.
            config: optional dict to override defaults (see `self._defaults`).

        Returns:
            self
        """
        self.check_dependencies()
        if config:
            self._cfg.update({k: v for k, v in config.items() if k in self._defaults})

        self._set_seed(int(self._cfg["seed"]))

        # ---- prepare X/y, split into numeric + categorical (int-coded) ----
        X_np = self._ensure_2d(self._as_numpy(X))
        y_np = self._as_numpy(y).ravel()

        # classification or regression?
        self._is_classification = self._infer_task_type(y_np)

        # label-encode y if classification and not already int
        if self._is_classification and not np.issubdtype(y_np.dtype, np.integer):
            self._y_le = LabelEncoder().fit(y_np)
            y_enc = self._y_le.transform(y_np)
            self._y_classes_ = self._y_le.classes_
        else:
            y_enc = y_np.astype(int) if self._is_classification else y_np.astype(np.float32)
            self._y_classes_ = np.unique(y_enc) if self._is_classification else None

        # find categorical columns
        if isinstance(X, pd.DataFrame):
            all_cols = list(X.columns)
            if cat_cols is None:
                cat_cols = [c for c in all_cols if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])]
            # normalize cat_cols to names
            cat_cols = [all_cols[i] if isinstance(i, int) else i for i in (cat_cols or [])]
            num_cols = [c for c in all_cols if c not in cat_cols]
            X_num = X[num_cols].to_numpy(dtype=np.float32) if num_cols else None
            X_cat_raw = X[cat_cols] if cat_cols else None
        else:
            # ndarray path: treat `cat_cols` as integer indices
            n_features = X_np.shape[1]
            if cat_cols is None:
                cat_cols = []
            cat_idx = list(cat_cols)
            num_idx = [i for i in range(n_features) if i not in cat_idx]
            X_num = X_np[:, num_idx].astype(np.float32) if num_idx else None
            X_cat_raw = pd.DataFrame(X_np[:, cat_idx]) if cat_idx else None
            num_cols = [f"num_{i}" for i in num_idx]
            cat_cols = [f"cat_{i}" for i in cat_idx]

        # encode categoricals to 0..K-1 per column
        self._feature_names_num = list(num_cols)
        self._feature_names_cat = list(cat_cols or [])
        X_cat = None
        self._cat_label_encoders = {}
        if X_cat_raw is not None and len(self._feature_names_cat) > 0:
            enc_mats = []
            for ci, cname in enumerate(self._feature_names_cat):
                col = X_cat_raw.iloc[:, ci]
                le = LabelEncoder().fit(col.astype(str))
                self._cat_label_encoders[cname] = le
                enc_mats.append(le.transform(col.astype(str)).astype(np.int64))
            X_cat = np.stack(enc_mats, axis=1)

        # combined tensors & bookkeeping
        self._n_num = 0 if X_num is None else int(X_num.shape[1])
        if X_cat is None or X_cat.shape[1] == 0:
            self._K = np.array([0], dtype=int)
        else:
            # K_j = cardinality of j-th categorical column
            self._K = np.array([int(np.max(X_cat[:, j]) + 1) for j in range(X_cat.shape[1])], dtype=int)

        if X_num is None and X_cat is None:
            raise ValueError("X must contain at least one numeric or categorical feature.")

        if X_num is None:
            X_all = X_cat.astype(np.float32)
        elif X_cat is None:
            X_all = X_num.astype(np.float32)
        else:
            X_all = np.concatenate([X_num.astype(np.float32), X_cat.astype(np.float32)], axis=1)

        y_tensor = torch.as_tensor(y_enc, dtype=torch.long if self._is_classification else torch.float32)
        X_tensor = torch.as_tensor(X_all, dtype=torch.float32)

        # class distribution (for conditional sampling)
        if self._is_classification:
            binc = np.bincount(y_tensor.cpu().numpy().astype(int))
            self._class_dist = torch.as_tensor(binc / binc.sum(), dtype=torch.float32, device=self.device)
        else:
            self._class_dist = None

        # ---- build denoiser (MLP) + diffusion wrapper ----
        d_in = (0 if self._K.size == 1 and self._K[0] == 0 else int(self._K.sum())) + self._n_num
        is_y_cond = True  # follow your training script
        num_classes = int(len(np.unique(y_enc))) if self._is_classification else 0

        rtdl_params = dict(d_layers=list(self._cfg["d_layers"]), dropout=float(self._cfg["dropout"]))
        if self._cfg["model_type"] != "mlp":
            raise ValueError("Only 'mlp' model_type is wired here. Extend to 'resnet' if needed.")
        self._denoiser = MLPDiffusion(
            d_in=d_in,
            num_classes=num_classes,
            is_y_cond=is_y_cond,
            rtdl_params=rtdl_params,
            dim_t=128,
        ).to(self.device)

        self._denoiser_ema = self._ema_clone(self._denoiser)

        self._diffusion = GaussianMultinomialDiffusion(
            num_classes=self._K if not (self._K.size == 1 and self._K[0] == 0) else np.array([0], dtype=int),
            num_numerical_features=self._n_num,
            denoise_fn=self._denoiser,
            gaussian_loss_type=self._cfg["gaussian_loss_type"],
            num_timesteps=int(self._cfg["num_timesteps"]),
            scheduler=self._cfg["scheduler"],
            device=self.device,
        ).to(self.device).train()

        # ---- optimizer & data loader ----
        opt = torch.optim.AdamW(self._diffusion.parameters(), lr=float(self._cfg["lr"]), weight_decay=float(self._cfg["weight_decay"]))

        dl = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=int(self._cfg["batch_size"]),
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        self._train_loader_infinite = self._infinite_batches(dl)

        # ---- training loop ----
        steps = int(self._cfg["steps"])
        for step in range(steps):
            xb, out = next(self._train_loader_infinite)
            xb = xb.to(self.device, non_blocking=True)
            if self._is_classification:
                out = {"y": out["y"].to(self.device, non_blocking=True)}
            else:
                # diffusion expects y as Long if used; for regression we keep a dummy y (not used)
                out = {"y": torch.zeros(xb.shape[0], dtype=torch.long, device=self.device)}

            opt.zero_grad(set_to_none=True)
            loss_multi, loss_gauss = self._diffusion.mixed_loss(xb, out)
            loss = loss_multi + loss_gauss
            loss.backward()
            opt.step()

            if self._cfg["use_ema"]:
                self._ema_update(self._denoiser_ema.parameters(), self._denoiser.parameters(), rate=0.999)

            if (step + 1) % 100 == 0:
                lm = float(loss_multi.detach().item())
                lg = float(loss_gauss.detach().item())
                print(f"Step {step+1}/{steps} | MLoss: {lm:.4f} | GLoss: {lg:.4f}")

        if self._cfg["use_ema"]:
            # swap EMA weights into diffusion's denoiser for downstream sampling/eval
            self._diffusion._denoise_fn.load_state_dict(self._denoiser_ema.state_dict())

        self.is_fitted = True
        return self

    # -------------------------------- evaluation ----------------------------------

    def evaluate(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        *,
        batches: Optional[int] = None,
    ) -> float:
        """Return a single scalar score (higher is better).

        By default, averages the negative diffusion training loss over a small
        number of batches (uses the cached train loader if X/y are not given).

        Args:
            X, y: optional evaluation data. If omitted, uses a few train batches.
            batches: number of mini-batches to average (defaults to config.eval_batches)

        Returns:
            float score (higher is better).
        """
        if not self.is_fitted or self._diffusion is None:
            raise RuntimeError("Call train() before evaluate().")

        self._diffusion.eval()

        if X is None or y is None:
            if self._train_loader_infinite is None:
                raise ValueError("No cached loader; provide X and y to evaluate().")
            loader = self._train_loader_infinite
        else:
            # build a quick loader from provided X/y
            X_np = self._ensure_2d(self._as_numpy(X)).astype(np.float32)
            y_np = self._as_numpy(y).ravel()
            if self._is_classification and self._y_le is not None and not np.issubdtype(y_np.dtype, np.integer):
                y_np = self._y_le.transform(y_np)
            X_tensor = torch.as_tensor(X_np, dtype=torch.float32)
            y_tensor = torch.as_tensor(y_np, dtype=torch.long if self._is_classification else torch.float32)
            dl = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=int(self._cfg["batch_size"]), shuffle=False, num_workers=0)
            loader = self._infinite_batches(dl)

        n_batches = int(self._cfg["eval_batches"] if batches is None else batches)
        losses: List[float] = []
        with torch.no_grad():
            for _ in range(n_batches):
                xb, out = next(loader)
                xb = xb.to(self.device, non_blocking=True)
                out = {"y": out["y"].to(self.device, non_blocking=True)}
                lm, lg = self._diffusion.mixed_loss(xb, out)
                losses.append(float((lm + lg).item()))
        # We return negative loss as a "score" so higher is better.
        return -float(np.mean(losses))

    # ---------------------------------- sampling ----------------------------------

    def sample(
        self,
        n: int,
        *,
        batch_size: Optional[int] = None,
        as_dataframe: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Generate `n` synthetic samples.

        Args:
            n: number of rows to sample.
            batch_size: micro-batch for sampler (defaults to training batch_size).
            as_dataframe: if True, returns a DataFrame with reconstructed columns;
                          otherwise returns a NumPy array.

        Returns:
            DataFrame or ndarray of shape (n, X.shape[1]).
        """
        if not self.is_fitted or self._diffusion is None:
            raise RuntimeError("Call train() before sample().")

        self._diffusion.eval()

        bsz = int(batch_size or self._cfg["batch_size"])
        y_dist = self._class_dist if self._class_dist is not None else torch.tensor([1.0], device=self.device)

        with torch.no_grad():
            x_synth, y_synth = self._diffusion.sample_all(
                num_samples=int(n),
                batch_size=bsz,
                y_dist=y_dist,
                ddim=False,
            )

        Xn = x_synth.cpu().numpy()
        Yn = y_synth.cpu().numpy().ravel()

        if not as_dataframe:
            return Xn

        # reconstruct columns: [num ... | cat ...]
        cols = []
        data = []
        start = 0
        if self._n_num > 0:
            num_block = Xn[:, start:start + self._n_num]
            data.append(pd.DataFrame(num_block, columns=self._feature_names_num))
            start += self._n_num

        if self._K.size > 0 and not (self._K.size == 1 and self._K[0] == 0):
            cat_block = Xn[:, start:start + len(self._feature_names_cat)]
            cat_block = np.rint(cat_block).clip(min=0)  # integers (safety)
            cat_df = pd.DataFrame(cat_block, columns=self._feature_names_cat).astype(int)
            # decode back to original labels if we encoded them
            for cname in self._feature_names_cat:
                le = self._cat_label_encoders.get(cname)
                if le is not None:
                    # unseen indices safeguard
                    max_valid = len(le.classes_) - 1
                    safe_vals = np.clip(cat_df[cname].to_numpy(), 0, max_valid)
                    cat_df[cname] = le.inverse_transform(safe_vals)
            data.append(cat_df)

        X_df = pd.concat(data, axis=1) if data else pd.DataFrame()

        # add y as last column if classification (mirroring your script’s CSVs)
        if self._is_classification:
            y_out = (
                self._y_le.inverse_transform(Yn.astype(int))
                if self._y_le is not None
                else Yn.astype(int)
            )
            X_df.insert(len(X_df.columns), "label", y_out)

        return X_df

    # --------------------------------- helpers ------------------------------------

    @staticmethod
    def _ema_clone(model: torch.nn.Module) -> torch.nn.Module:
        ema = type(model)(**model.__dict__['_modules']) if False else type(model)(*[])  # placeholder to appease linters
        # Safer: deep copy state_dict into an identically constructed module
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.detach_()
        return ema

    @staticmethod
    def _ema_update(target_params, source_params, rate: float = 0.999) -> None:
        for targ, src in zip(target_params, source_params):
            targ.data.mul_(rate).add_(src.data, alpha=1.0 - rate)
