from __future__ import annotations

import importlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.refs import ModelRef
from katabatic.models.base_model import Model as BaseModel

from .utils import (
    ColumnMeta,
    build_schema,
    decode_dataframe,
    encode_dataframe,
    square_to_table,
    table_to_square,
)


def _try_import(module: str):
    """Import an optional dependency without failing at module import time."""
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _network_side(n_columns: int, minimum: int = 4) -> int:
    """Return a power-of-two square side suitable for convolutional networks."""

    side = minimum

    while side * side < n_columns:
        side *= 2

    return side


class TableGANModel(BaseModel):
    """TableGAN-style synthetic tabular data generator.

    The model converts tabular rows into square matrices and trains a
    convolutional GAN over those matrices. Generated matrices are converted
    back into rows using the fitted table schema.

    The implementation also uses feature-statistics information loss and an
    auxiliary classifier that encourages generated records to preserve the
    relationship between the feature columns and the target column.

    Katabatic's benchmark preprocessing places the target in the final column.
    """

    ARTIFACT_STATE_FILES = ("tablegan_state.pt",)

    _NON_SERIALISABLE = (
        "generator",
        "discriminator",
        "classifier",
        "_device",
    )

    def __init__(
        self,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        noise_dim: int = 100,
        base_channels: int = 64,
        information_weight: float = 1.0,
        delta_mean: float = 0.0,
        delta_std: float = 0.0,
        classifier_weight: float = 1.0,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.5, 0.999),
        seed: int = 42,
        device: str | None = "cpu",
    ) -> None:
        super().__init__()

        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if noise_dim < 1:
            raise ValueError("noise_dim must be at least 1.")

        if base_channels < 1:
            raise ValueError("base_channels must be at least 1.")

        if information_weight < 0:
            raise ValueError("information_weight cannot be negative.")

        if delta_mean < 0:
            raise ValueError("delta_mean cannot be negative.")

        if delta_std < 0:
            raise ValueError("delta_std cannot be negative.")

        if classifier_weight < 0:
            raise ValueError("classifier_weight cannot be negative.")

        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "noise_dim": noise_dim,
            "base_channels": base_channels,
            "information_weight": information_weight,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "classifier_weight": classifier_weight,
            "lr": lr,
            "betas": list(betas),
            "seed": seed,
            "device": device,
        }

        self.schema: list[ColumnMeta] | None = None
        self.columns: list[str] = []

        self._train_df: pd.DataFrame | None = None
        self._n_columns = 0
        self._side_length = 0

        self.generator = None
        self.discriminator = None
        self.classifier = None
        self._device = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch"]

    def _build_networks(self, torch, nn) -> None:
        """Construct the generator, discriminator, and auxiliary classifier."""

        noise_dim = self.cfg["noise_dim"]
        base = self.cfg["base_channels"]
        side = self._side_length

        if side < 4 or side & (side - 1):
            raise ValueError(
                "TableGAN matrix side length must be a power of two and >= 4."
            )

        n_stages = int(np.log2(side)) - 2

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()

                start_channels = base * (2 ** max(n_stages, 0))

                layers: list[nn.Module] = [
                    nn.ConvTranspose2d(
                        noise_dim,
                        start_channels,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(start_channels),
                    nn.ReLU(True),
                ]

                current_channels = start_channels

                for _ in range(n_stages):
                    next_channels = max(base, current_channels // 2)

                    layers.extend(
                        [
                            nn.ConvTranspose2d(
                                current_channels,
                                next_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(next_channels),
                            nn.ReLU(True),
                        ]
                    )

                    current_channels = next_channels

                layers.extend(
                    [
                        nn.Conv2d(
                            current_channels,
                            1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.Tanh(),
                    ]
                )

                self.net = nn.Sequential(*layers)

            def forward(self, z):
                return self.net(z)

        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()

                feature_layers: list[nn.Module] = []

                current_channels = 1
                next_channels = base
                current_side = side

                while current_side > 4:
                    feature_layers.extend(
                        [
                            nn.Conv2d(
                                current_channels,
                                next_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.LeakyReLU(0.2, inplace=True),
                        ]
                    )

                    current_channels = next_channels
                    next_channels *= 2
                    current_side //= 2

                self.features = nn.Sequential(*feature_layers)

                self.classifier = nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        1,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.Sigmoid(),
                )

            def forward(self, x, return_features=False):
                features = self.features(x)
                probability = self.classifier(features).view(-1)

                if return_features:
                    return probability, features

                return probability

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()

                feature_layers: list[nn.Module] = []

                current_channels = 1
                next_channels = base
                current_side = side

                while current_side > 4:
                    feature_layers.extend(
                        [
                            nn.Conv2d(
                                current_channels,
                                next_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.LeakyReLU(0.2, inplace=True),
                        ]
                    )

                    current_channels = next_channels
                    next_channels *= 2
                    current_side //= 2

                self.features = nn.Sequential(*feature_layers)

                self.output = nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        1,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.Tanh(),
                )

            def forward(self, x):
                return self.output(self.features(x)).view(-1)

        self.generator = Generator().to(self._device)
        self.discriminator = Discriminator().to(self._device)
        self.classifier = Classifier().to(self._device)

    def _information_loss(
        self,
        real_features,
        fake_features,
        torch,
    ):
        """Compute first- and second-order feature-statistics information loss."""

        real_flat = real_features.reshape(real_features.size(0), -1)
        fake_flat = fake_features.reshape(fake_features.size(0), -1)

        real_mean = real_flat.mean(dim=0)
        fake_mean = fake_flat.mean(dim=0)

        real_std = real_flat.std(dim=0, unbiased=False)
        fake_std = fake_flat.std(dim=0, unbiased=False)

        mean_distance = torch.norm(
            real_mean - fake_mean,
            p=2,
        )

        std_distance = torch.norm(
            real_std - fake_std,
            p=2,
        )

        mean_loss = torch.clamp(
            mean_distance - self.cfg["delta_mean"],
            min=0.0,
        )

        std_loss = torch.clamp(
            std_distance - self.cfg["delta_std"],
            min=0.0,
        )

        return mean_loss + std_loss

    def _classifier_inputs_and_targets(
        self,
        matrices,
    ):
        """Mask the target cell and return its encoded value.

        Katabatic's benchmark preprocessing places the target in the final
        column. The target cell is removed from the classifier input so that
        the classifier must infer it from the remaining generated features.
        """

        if self._n_columns < 1:
            raise RuntimeError("TableGAN schema has not been initialised.")

        if self._side_length < 1:
            raise RuntimeError("TableGAN matrix side length has not been initialised.")

        target_index = self._n_columns - 1

        target_row = target_index // self._side_length
        target_column = target_index % self._side_length

        classifier_inputs = matrices.clone()

        targets = classifier_inputs[
            :,
            0,
            target_row,
            target_column,
        ].clone()

        classifier_inputs[
            :,
            0,
            target_row,
            target_column,
        ] = 0.0

        return classifier_inputs, targets

    def _classification_loss(
        self,
        predictions,
        targets,
        torch,
    ):
        """Compute mean absolute discrepancy between predicted and target values."""

        return torch.mean(torch.abs(predictions - targets))

    @staticmethod
    def _set_requires_grad(network, requires_grad: bool) -> None:
        """Enable or disable gradients for all parameters in a network."""

        for parameter in network.parameters():
            parameter.requires_grad_(requires_grad)

    def _load_training_dataframe(self, data_dir: str) -> pd.DataFrame:
        """Load Katabatic's standard training files."""

        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            return pd.read_csv(train_full)

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(
                f"Could not find training data in {data_dir}. Expected "
                "train_full.csv or x_train.csv/y_train.csv."
            )

        x_train = pd.read_csv(x_path)
        y_train = pd.read_csv(y_path)

        if y_train.shape[1] != 1:
            raise ValueError("y_train.csv must have exactly one column (the target).")

        return pd.concat(
            [
                x_train.reset_index(drop=True),
                y_train.reset_index(drop=True),
            ],
            axis=1,
        )

    def train(
        self,
        data_dir: str,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> TableGANModel:
        """Train TableGAN on a Katabatic dataset."""

        torch = _try_import("torch")
        nn = _try_import("torch.nn")
        data_utils = _try_import("torch.utils.data")

        if torch is None or nn is None or data_utils is None:
            raise ImportError(
                "TableGAN requires PyTorch. Install the TableGAN extra first."
            )

        df = self._load_training_dataframe(data_dir)

        if df.empty:
            raise ValueError("Training data must contain at least one row.")

        categorical_cols = kwargs.get("categorical_cols")
        continuous_cols = kwargs.get("continuous_cols")

        self.schema = build_schema(
            df,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
        )

        self.columns = df.columns.tolist()
        self._train_df = df.copy()
        self._n_columns = len(self.columns)
        self._side_length = _network_side(self._n_columns)

        encoded = encode_dataframe(
            df,
            self.schema,
        )

        matrices = table_to_square(
            encoded,
            side_length=self._side_length,
        )

        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        self._device = torch.device(
            self.cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._build_networks(torch, nn)

        tensor_data = torch.tensor(
            matrices,
            dtype=torch.float32,
        )

        dataset = data_utils.TensorDataset(tensor_data)

        loader = data_utils.DataLoader(
            dataset,
            batch_size=min(
                self.cfg["batch_size"],
                len(dataset),
            ),
            shuffle=True,
            drop_last=False,
        )

        generator = self.generator
        discriminator = self.discriminator
        classifier = self.classifier

        criterion = nn.BCELoss()

        g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=self.cfg["lr"],
            betas=tuple(self.cfg["betas"]),
        )

        d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.cfg["lr"],
            betas=tuple(self.cfg["betas"]),
        )

        c_optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=self.cfg["lr"],
            betas=tuple(self.cfg["betas"]),
        )

        generator.train()
        discriminator.train()
        classifier.train()

        for _ in range(self.cfg["epochs"]):
            for (real_batch,) in loader:
                real_batch = real_batch.to(self._device)
                batch_size = real_batch.size(0)

                # -------------------------
                # Train discriminator
                # -------------------------
                self._set_requires_grad(
                    discriminator,
                    True,
                )

                discriminator.zero_grad(set_to_none=True)

                real_labels = torch.ones(
                    batch_size,
                    dtype=torch.float32,
                    device=self._device,
                )

                fake_labels = torch.zeros(
                    batch_size,
                    dtype=torch.float32,
                    device=self._device,
                )

                real_output = discriminator(real_batch)

                real_loss = criterion(
                    real_output,
                    real_labels,
                )

                noise = torch.randn(
                    batch_size,
                    self.cfg["noise_dim"],
                    1,
                    1,
                    device=self._device,
                )

                fake_batch = generator(noise)

                fake_output = discriminator(fake_batch.detach())

                fake_loss = criterion(
                    fake_output,
                    fake_labels,
                )

                d_loss = real_loss + fake_loss

                d_loss.backward()
                d_optimizer.step()

                # -------------------------
                # Train auxiliary classifier
                # -------------------------
                self._set_requires_grad(
                    classifier,
                    True,
                )

                classifier.zero_grad(set_to_none=True)

                (
                    real_classifier_inputs,
                    real_classifier_targets,
                ) = self._classifier_inputs_and_targets(real_batch)

                real_classifier_predictions = classifier(real_classifier_inputs)

                classifier_loss = self._classification_loss(
                    real_classifier_predictions,
                    real_classifier_targets,
                    torch,
                )

                classifier_loss.backward()
                c_optimizer.step()

                # -------------------------
                # Train generator
                # -------------------------
                self._set_requires_grad(
                    discriminator,
                    False,
                )

                self._set_requires_grad(
                    classifier,
                    False,
                )

                generator.zero_grad(set_to_none=True)

                generated_output, fake_features = discriminator(
                    fake_batch,
                    return_features=True,
                )

                with torch.no_grad():
                    _, real_features = discriminator(
                        real_batch,
                        return_features=True,
                    )

                adversarial_loss = criterion(
                    generated_output,
                    real_labels,
                )

                information_loss = self._information_loss(
                    real_features,
                    fake_features,
                    torch,
                )

                (
                    fake_classifier_inputs,
                    fake_classifier_targets,
                ) = self._classifier_inputs_and_targets(fake_batch)

                fake_classifier_predictions = classifier(fake_classifier_inputs)

                generator_classification_loss = self._classification_loss(
                    fake_classifier_predictions,
                    fake_classifier_targets,
                    torch,
                )

                g_loss = (
                    adversarial_loss
                    + self.cfg["information_weight"] * information_loss
                    + self.cfg["classifier_weight"] * generator_classification_loss
                )

                g_loss.backward()
                g_optimizer.step()

                self._set_requires_grad(
                    discriminator,
                    True,
                )

                self._set_requires_grad(
                    classifier,
                    True,
                )

        self.is_fitted = True

        synth_dir = synthetic_dir

        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"

            synth_dir = os.path.join(
                "synthetic",
                dataset_name,
                "tablegan",
            )

        os.makedirs(
            synth_dir,
            exist_ok=True,
        )

        synthetic_df = self.sample(len(df))

        # Katabatic's benchmark preprocessing places the target last.
        label = df.columns[-1]

        x_synth = synthetic_df[df.columns[:-1]].copy()

        y_synth = synthetic_df[[label]].copy()

        x_output = os.path.join(
            synth_dir,
            "x_synth.csv",
        )

        y_output = os.path.join(
            synth_dir,
            "y_synth.csv",
        )

        x_synth.to_csv(
            x_output,
            index=False,
        )

        y_synth.to_csv(
            y_output,
            index=False,
        )

        metadata = {
            "model": "TableGAN",
            "schema": {
                "columns": self.columns,
                "label": label,
                "dtypes": {column: str(df[column].dtype) for column in df.columns},
                "categorical_columns": [
                    meta.name for meta in self.schema if meta.kind == "categorical"
                ],
                "continuous_columns": [
                    meta.name for meta in self.schema if meta.kind == "continuous"
                ],
            },
            "training": self.cfg,
            "matrix_side_length": self._side_length,
        }

        with open(
            os.path.join(
                synth_dir,
                "metadata.json",
            ),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        artifact_state_dir = kwargs.get("artifact_state_dir")

        if artifact_state_dir:
            self._save_artifact_state(artifact_state_dir)

        return self

    def sample(
        self,
        n: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic rows from the fitted generator."""

        if not self.is_fitted or self.schema is None or self.generator is None:
            raise RuntimeError("Call train() before sample().")

        torch = _try_import("torch")

        if torch is None:
            raise ImportError("PyTorch is required to sample from TableGAN.")

        n_rows = int(n) if n is not None else 1000

        if n_rows < 1:
            raise ValueError("n must be at least 1.")

        self.generator.eval()

        generated_batches: list[np.ndarray] = []
        remaining = n_rows

        with torch.no_grad():
            while remaining > 0:
                current_batch = min(
                    remaining,
                    self.cfg["batch_size"],
                )

                noise = torch.randn(
                    current_batch,
                    self.cfg["noise_dim"],
                    1,
                    1,
                    device=self._device,
                )

                generated = self.generator(noise)

                generated_batches.append(generated.cpu().numpy())

                remaining -= current_batch

        generated_matrices = np.concatenate(
            generated_batches,
            axis=0,
        )

        encoded = square_to_table(
            generated_matrices,
            self._n_columns,
        )

        synthetic_df = decode_dataframe(
            encoded,
            self.schema,
        )

        return synthetic_df[self.columns]

    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> float:
        """Return a placeholder score.

        Katabatic's benchmark evaluation layer performs the actual synthetic
        data quality evaluation.
        """

        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        return 0.0

    def _save_artifact_state(
        self,
        artifact_state_dir: str,
    ) -> None:
        """Persist fitted TableGAN state."""

        torch = _try_import("torch")

        if torch is None:
            raise ImportError("PyTorch is required to save TableGAN artifacts.")

        os.makedirs(
            artifact_state_dir,
            exist_ok=True,
        )

        payload: dict[str, Any] = {
            "attrs": {
                key: value
                for key, value in self.__dict__.items()
                if key not in self._NON_SERIALISABLE
            },
            "generator_state": (
                self.generator.state_dict() if self.generator is not None else None
            ),
            "discriminator_state": (
                self.discriminator.state_dict()
                if self.discriminator is not None
                else None
            ),
            "classifier_state": (
                self.classifier.state_dict() if self.classifier is not None else None
            ),
        }

        path = os.path.join(
            artifact_state_dir,
            self.ARTIFACT_STATE_FILES[0],
        )

        torch.save(
            payload,
            path,
        )

    @classmethod
    def load_from_ref(
        cls,
        store: ArtifactStore,
        ref: ModelRef,
    ) -> TableGANModel:
        """Reload a fitted TableGAN model from a Katabatic artifact."""

        state_path = store.open_path(
            f"{ref.state_relpath}/{cls.ARTIFACT_STATE_FILES[0]}"
        )

        if not state_path.is_file():
            raise FileNotFoundError(f"No TableGAN state found at {state_path}.")

        torch = _try_import("torch")

        if torch is None:
            raise ImportError("PyTorch is required to load TableGAN artifacts.")

        payload = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )

        instance = cls()

        instance.__dict__.update(payload["attrs"])

        instance._device = torch.device(instance.cfg.get("device") or "cpu")

        nn = _try_import("torch.nn")

        if nn is None:
            raise ImportError("PyTorch is required to load TableGAN artifacts.")

        instance._build_networks(
            torch,
            nn,
        )

        generator_state = payload.get("generator_state")

        if generator_state is not None:
            instance.generator.load_state_dict(generator_state)

        discriminator_state = payload.get("discriminator_state")

        if discriminator_state is not None:
            instance.discriminator.load_state_dict(discriminator_state)

        classifier_state = payload.get("classifier_state")

        if classifier_state is not None:
            instance.classifier.load_state_dict(classifier_state)

        instance.generator.eval()
        instance.discriminator.eval()
        instance.classifier.eval()

        return instance
