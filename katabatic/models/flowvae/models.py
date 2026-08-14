"""Flow-VAE model wrapper for the Katabatic train/test split pipeline."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from katabatic.models.base_model import Model as BaseModel

from .utils import FlowVAE, fit_transform_tabular, inverse_transform_tabular


class FlowVAEModel(BaseModel):
    """Tabular Flow-VAE generator for Katabatic.

    This adapts the uploaded static Flow-VAE MLP architecture into a Katabatic
    3-file model. It trains on the joined train dataframe, including the label,
    then decodes generated rows back into x_synth.csv and y_synth.csv.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        layers: int = 2,
        gate: bool = False,
        flow_type: str = "planar",
        flow_length: int = 2,
        batch_size: int = 128,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.gate = gate
        self.flow_type = flow_type
        self.flow_length = flow_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device_name = device

        self.model: Optional[FlowVAE] = None
        self.schema = None
        self.n_train = 0
        self.label_column: Optional[str] = None
        self.loss_history: list[float] = []

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "numpy", "pandas", "scikit-learn"]

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _device(self) -> torch.device:
        if self.device_name:
            return torch.device(self.device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_training_dataframe(self, data_dir: str) -> pd.DataFrame:
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            return pd.read_csv(train_full)

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(
                f"Could not find train_full.csv or x_train.csv/y_train.csv in {data_dir}."
            )

        x_train = pd.read_csv(x_path)
        y_train = pd.read_csv(y_path)
        if y_train.shape[1] != 1:
            raise ValueError("y_train.csv must contain exactly one label column.")
        return pd.concat([x_train, y_train], axis=1)

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        categorical_cols: Optional[list] = None,
        continuous_cols: Optional[list] = None,
        **kwargs,
    ) -> "FlowVAEModel":
        """Fit the Flow-VAE and save synthetic files expected by Katabatic."""
        self._set_seed()
        device = self._device()
        df = self._load_training_dataframe(data_dir)
        self.label_column = df.columns[-1]
        self.n_train = len(df)

        matrix, schema = fit_transform_tabular(
            df,
            label_column=self.label_column,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )
        self.schema = schema

        dataset = TensorDataset(torch.tensor(matrix, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=True,
            drop_last=False,
        )

        self.model = FlowVAE(
            in_dim=matrix.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            layers=self.layers,
            gate=self.gate,
            flow_type=self.flow_type,
            flow_length=self.flow_length,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(
            f"[FlowVAE] Training on {self.n_train} rows, {matrix.shape[1]} encoded dimensions, device={device}."
        )
        start = time.time()
        self.loss_history = []
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_losses: list[float] = []
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = self.model.loss(batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.loss_history.append(mean_loss)
            if epoch == 1 or epoch == self.epochs or epoch % max(1, self.epochs // 5) == 0:
                print(f"[FlowVAE] Epoch {epoch}/{self.epochs} - loss={mean_loss:.4f}")

        self.is_fitted = True
        print(f"[FlowVAE] Training finished in {time.time() - start:.2f} seconds.")

        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "flow_vae")
        os.makedirs(synth_dir, exist_ok=True)

        synth_df = self.sample(self.n_train)
        label = self.label_column
        x_synth = synth_df[[c for c in synth_df.columns if c != label]].copy()

        # CHANGED: synthetic labels are now decoded from the model output
        # instead of being resampled from the training marginal.
        #
        # The previous implementation discarded the label column produced by
        # the decoder and drew y_synth with np.random.choice from the real
        # label distribution. That reproduced the correct class marginals but
        # left no relationship between synthetic features and synthetic
        # labels, which is exactly what TSTR measures. Baseline utility on the
        # car dataset was 0.0, with all four classifiers at or below the 0.70
        # majority-class rate (LR 0.6994, MLP 0.6936, RF 0.6532, XGB 0.5983).
        #
        # synth_df already contains the label column, decoded by
        # inverse_transform_tabular via argmax over its one-hot block, and it
        # comes from the same generated rows as x_synth, so the pairing holds.

        # The label column in the training data is already integer-encoded, so
        # the decoded values from synth_df need no further remapping. The old
        # label_to_encoded dict was only needed because labels were being
        # constructed from scratch rather than decoded.
        y_synth = synth_df[[label]].copy()

        # The decoder may never emit a rare class. Report it rather than
        # silently injecting labels, since a missing class is a real property
        # of the model that affects downstream evaluation.
        missing = set(df[self.label_column].astype(str).unique()) - set(
            y_synth[label].astype(str).unique()
        )
        if missing:
            print(f"[FlowVAE] Warning: classes absent from synthetic labels: {sorted(missing)}")

        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False)

        metadata = {
            "model": "FlowVAEModel",
            "schema": asdict(schema),
            "training": {
                "n_train": self.n_train,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "layers": self.layers,
                "gate": self.gate,
                "flow_type": self.flow_type,
                "flow_length": self.flow_length,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "loss_history": self.loss_history,
            },
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[FlowVAE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")
        return self

    def sample(self, n: Optional[int] = None, *args, **kwargs) -> pd.DataFrame:
        """Generate synthetic rows decoded back to the original columns."""
        if not self.is_fitted or self.model is None or self.schema is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = int(n or self.n_train)
        device = self._device()
        self.model.eval()
        chunks = []
        with torch.no_grad():
            remaining = n_samples
            while remaining > 0:
                batch_n = min(self.batch_size, remaining)
                generated = self.model.sample(batch_n, device=device).detach().cpu().numpy()
                chunks.append(generated)
                remaining -= batch_n
        generated_matrix = np.vstack(chunks)
        synth_df = inverse_transform_tabular(generated_matrix, self.schema)
        return synth_df

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return float(self.loss_history[-1]) if self.loss_history else 0.0
