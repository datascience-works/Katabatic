from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple
import os
import json
import importlib

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from .utils import (
    infer_schema,
    fit_transformers,
    encode_df,
    decode_batch,
    build_conditioning,
    sample_conditions,
    ColumnMeta,
)


def _try_import(module: str):
    try:
        return importlib.import_module(module)
    except Exception:
        return None


class CTGANModel(BaseModel):
    """
    Internal CTGAN-style generator with conditional vectors over categoricals.
    - Encodes categoricals via one-hot; continuous via QuantileTransformer to Normal.
    - Torch backend: WGAN(-GP) with Gumbel-Softmax for categorical outputs.
    - NumPy backend: fast, dependency-light fallback for stability.
    """

    def __init__(
        self,
        *,
        epochs: int = 100,
        batch_size: int = 512,
        noise_dim: int = 128,
        generator_hidden: Tuple[int, ...] = (256, 256),
        discriminator_hidden: Tuple[int, ...] = (256, 256),
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.9),
        lambda_gp: float = 10.0,
        use_gradient_penalty: bool = False,
        clip_value: float = 0.01,
        n_critic: int = 5,
        gumbel_tau: float = 0.2,
        seed: int = 42,
        device: Optional[str] = "cpu",
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "noise_dim": noise_dim,
            "generator_hidden": list(generator_hidden),
            "discriminator_hidden": list(discriminator_hidden),
            "lr": lr,
            "betas": list(betas),
            "lambda_gp": lambda_gp,
            "use_gradient_penalty": use_gradient_penalty,
            "clip_value": clip_value,
            "n_critic": n_critic,
            "gumbel_tau": gumbel_tau,
            "seed": seed,
            "device": device,
            "backend": backend,
        }
        self.schema: List[ColumnMeta] | None = None
        self.output_order: List[str] = []
        self.cat_blocks: Dict[str, Tuple[int, int]] = {}
        self.cond_blocks: Dict[str, Tuple[int, int]] = {}
        self.empirical_probs: Dict[str, np.ndarray] = {}
        self._train_df: Optional[pd.DataFrame] = None

        # Torch-related members (only set when backend == 'torch')
        self.generator = None
        self.discriminator = None
        self._enc_dim: int = 0
        self._cond_dim: int = 0
        self._device = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        # Torch is only required for the torch backend.
        return ["torch", "sklearn"]

    def _build_torch_networks(self, torch, nn) -> None:
        in_g = self.cfg["noise_dim"] + self._cond_dim
        out_g = self._enc_dim
        in_d = self._enc_dim + self._cond_dim

        class LocalMLP(nn.Module):
            def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
                super().__init__()
                layers: List[nn.Module] = []
                last = in_dim
                for h in hidden:
                    layers += [nn.Linear(last, h), nn.ReLU()]
                    last = h
                layers.append(nn.Linear(last, out_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        self.generator = LocalMLP(
            in_g, self.cfg["generator_hidden"], out_g).to(self._device)
        self.discriminator = LocalMLP(
            in_d, self.cfg["discriminator_hidden"], 1).to(self._device)

    def _gumbelize_cats(self, logits) -> Any:
        torch = _try_import("torch")
        F = _try_import("torch.nn.functional")
        out = []
        ptr = 0
        for name in self.output_order:
            if name in self.cat_blocks:
                start, end = self.cat_blocks[name]
                size = end - start
                block = logits[:, ptr:ptr+size]
                out.append(F.gumbel_softmax(
                    block, tau=self.cfg["gumbel_tau"], hard=False, dim=1))
                ptr += size
            else:
                val = torch.tanh(logits[:, ptr:ptr+1])
                out.append(val)
                ptr += 1
        return torch.cat(out, dim=1)

    def _forward_generator(self, torch, z, cond):
        logits = self.generator(torch.cat([z, cond], dim=1))
        return self._gumbelize_cats(logits)

    def _forward_discriminator(self, torch, x_enc, cond):
        return self.discriminator(torch.cat([x_enc, cond], dim=1))

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CTGANModel":
        # Load data
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")
        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(
                    f"Could not find training data in {data_dir}. Expected train_full.csv or x_train.csv/y_train.csv.")
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError(
                    "y_train.csv must have exactly one column (the target).")
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        # Save for fallback sampling
        self._train_df = df.copy()

        # Schema & encoders
        self.schema = infer_schema(df)
        fit_transformers(df, self.schema)

        backend = (self.cfg.get("backend") or "torch").lower()
        if backend == "torch":
            torch = _try_import("torch")
            nn = _try_import("torch.nn")
            data_utils = _try_import("torch.utils.data")
            F = _try_import("torch.nn.functional")
            if torch is None or nn is None or data_utils is None or F is None:
                # Fall back if torch not available
                self.is_fitted = True
            else:
                torch.manual_seed(self.cfg["seed"])

                enc, self.cat_blocks, self.output_order = encode_df(
                    df, self.schema)
                cond_full, self.cond_blocks = build_conditioning(
                    df, self.schema)

                # Empirical probabilities per categorical column
                self.empirical_probs = {}
                for col in self.schema:
                    if col.kind == 'categorical':
                        start, end = self.cond_blocks[col.name]
                        counts = cond_full[:, start:end].sum(axis=0) + 1e-8
                        p = counts / counts.sum()
                        self.empirical_probs[col.name] = p.astype(np.float32)

                self._enc_dim = int(enc.shape[1])
                self._cond_dim = int(cond_full.shape[1])
                self._device = torch.device(self.cfg["device"] or (
                    "cuda" if torch.cuda.is_available() else "cpu"))
                self._build_torch_networks(torch, nn)

                # DataLoader
                TensorDataset = data_utils.TensorDataset
                DataLoader = data_utils.DataLoader
                enc_t = torch.tensor(enc, dtype=torch.float32)
                cond_t = torch.tensor(cond_full, dtype=torch.float32)
                loader = DataLoader(TensorDataset(
                    enc_t, cond_t), batch_size=self.cfg["batch_size"], shuffle=True, drop_last=True)

                G = self.generator
                D = self.discriminator
                g_opt = torch.optim.Adam(
                    G.parameters(), lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]))
                d_opt = torch.optim.Adam(
                    D.parameters(), lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]))

                def _wgangp_gradient_penalty(discriminator, real, fake, cond, device, lambda_gp: float = 10.0):
                    batch_size = real.size(0)
                    alpha = torch.rand(batch_size, 1, device=device)
                    alpha = alpha.expand_as(real)
                    interpolates = alpha * real + (1 - alpha) * fake
                    interpolates = torch.cat([interpolates, cond], dim=1)
                    interpolates.requires_grad_(True)
                    d_interpolates = discriminator(interpolates)
                    grads = torch.autograd.grad(
                        outputs=d_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(d_interpolates),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grads = grads[:, :real.size(1)]
                    gp = ((grads.view(batch_size, -1).norm(2, dim=1) - 1)
                          ** 2).mean() * lambda_gp
                    return gp

                G.train()
                D.train()
                for _ in range(self.cfg["epochs"]):
                    for real_batch, real_cond in loader:
                        real_batch = real_batch.to(self._device)
                        real_cond = real_cond.to(self._device)

                        # n_critic steps
                        for _ in range(self.cfg["n_critic"]):
                            z = torch.randn(real_batch.size(
                                0), self.cfg["noise_dim"], device=self._device)
                            cond_fake_np = sample_conditions(real_batch.size(
                                0), self.schema, self.cond_blocks, self.empirical_probs)
                            cond_fake = torch.tensor(
                                cond_fake_np, dtype=torch.float32, device=self._device)
                            fake_batch = self._forward_generator(
                                torch, z, cond_fake).detach()

                            d_real = self._forward_discriminator(
                                torch, real_batch, real_cond).mean()
                            d_fake = self._forward_discriminator(
                                torch, fake_batch, cond_fake).mean()
                            if self.cfg["use_gradient_penalty"]:
                                gp = _wgangp_gradient_penalty(
                                    D, real_batch, fake_batch, cond_fake, self._device, self.cfg["lambda_gp"])
                                d_loss = d_fake - d_real + gp
                            else:
                                d_loss = d_fake - d_real

                            d_opt.zero_grad(set_to_none=True)
                            d_loss.backward()
                            d_opt.step()
                            if not self.cfg["use_gradient_penalty"]:
                                for p in D.parameters():
                                    p.data.clamp_(-self.cfg["clip_value"],
                                                  self.cfg["clip_value"])

                        z = torch.randn(real_batch.size(
                            0), self.cfg["noise_dim"], device=self._device)
                        cond_fake_np = sample_conditions(real_batch.size(
                            0), self.schema, self.cond_blocks, self.empirical_probs)
                        cond_fake = torch.tensor(
                            cond_fake_np, dtype=torch.float32, device=self._device)
                        fake_batch = self._forward_generator(
                            torch, z, cond_fake)
                        g_loss = - \
                            self._forward_discriminator(
                                torch, fake_batch, cond_fake).mean()

                        g_opt.zero_grad(set_to_none=True)
                        g_loss.backward()
                        g_opt.step()

                self.is_fitted = True
        else:
            # NumPy backend: skip NN training
            self.is_fitted = True

        # Save outputs
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(
                os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "ctgan")
        os.makedirs(synth_dir, exist_ok=True)

        df_s = self.sample(n=len(df))
        label = df.columns[-1]
        x_synth = df_s[df.columns[:-1]].copy()
        y_synth = df_s[[label]].copy()

        # Align names with real X
        real_x_train_path = os.path.join(data_dir, "x_train.csv")
        try:
            real_cols = pd.read_csv(
                real_x_train_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        meta = {
            "schema": {
                "columns": df.columns.tolist(),
                "label": label,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "categorical_columns": [c.name for c in self.schema or [] if c.kind == 'categorical']
            },
            "training": self.cfg,
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[CTGAN] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")
        return self

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: Optional[int] = None,
        conditional: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.schema is None:
            raise RuntimeError("Call train() before sample().")

        backend = (self.cfg.get("backend") or "torch").lower()
        if backend == "torch" and self.generator is not None:
            torch = _try_import("torch")
            if torch is None:
                backend = "numpy"
            else:
                batch = int(n) if n is not None else 1000
                self.generator.eval()
                all_rows: List[pd.DataFrame] = []
                with torch.no_grad():
                    steps = (batch + 1023) // 1024
                    remain = batch
                    for _ in range(steps):
                        bsz = min(1024, remain)
                        remain -= bsz
                        z = torch.randn(
                            bsz, self.cfg["noise_dim"], device=self._device)

                        if conditional and len(conditional) == 1 and self.cond_blocks:
                            name, val = next(iter(conditional.items()))
                            cond = np.zeros(
                                (bsz, self._cond_dim), dtype=np.float32)
                            if name in self.cond_blocks:
                                start, end = self.cond_blocks[name]
                                cats = next(
                                    (c.categories for c in self.schema if c.name == name), [])
                                if cats:
                                    try:
                                        idx = list(cats).index(str(val))
                                        cond[:, start + idx] = 1.0
                                    except ValueError:
                                        pass
                            cond_t = torch.tensor(
                                cond, dtype=torch.float32, device=self._device)
                        else:
                            cond_np = sample_conditions(
                                bsz, self.schema, self.cond_blocks, self.empirical_probs)
                            cond_t = torch.tensor(
                                cond_np, dtype=torch.float32, device=self._device)

                        enc_fake = self._forward_generator(
                            torch, z, cond_t).cpu().numpy()
                        df_batch = decode_batch(enc_fake, self.schema)
                        all_rows.append(df_batch)

                df_out = pd.concat(all_rows, axis=0).reset_index(drop=True)
                ordered_cols = [c.name for c in self.schema]
                return df_out[ordered_cols]

        # NumPy fallback
        n_rows = int(n) if n is not None else (
            len(self._train_df) if self._train_df is not None else 1000)
        rows: Dict[str, list] = {c.name: [] for c in self.schema}
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        train_df = self._train_df if self._train_df is not None else None

        for _ in range(n_rows):
            for col in self.schema:
                if col.kind == 'categorical':
                    cats = col.categories or []
                    if train_df is not None and len(cats) > 0:
                        counts = train_df[col.name].astype(str).value_counts().reindex(
                            cats, fill_value=0).values + 1e-8
                        p = counts / counts.sum()
                        choice = rng.choice(cats, p=p)
                        rows[col.name].append(choice)
                    else:
                        choice = rng.choice(cats) if len(cats) else None
                        rows[col.name].append(choice)
                else:
                    qt = col.qt
                    if train_df is not None and qt is not None:
                        vals = train_df[[col.name]].astype(float)
                        norm = qt.transform(vals)
                        mu = float(norm.mean())
                        sigma = float(norm.std() + 1e-6)
                        z = rng.normal(mu, sigma)
                        orig = float(qt.inverse_transform(
                            np.array([[z]])).ravel()[0])
                        rows[col.name].append(orig)
                    else:
                        rows[col.name].append(0.0)

        df_out = pd.DataFrame(rows)
        ordered_cols = [c.name for c in self.schema]
        return df_out[ordered_cols]
