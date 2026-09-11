"""
Production-level MedGAN implementation for the Katabatic framework.

Based on "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks"
by Choi et al. (2017) - https://arxiv.org/abs/1703.06490
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from katabatic.models.base_model import Model
from katabatic.models.medgan.utils import (
    Autoencoder,
    Discriminator,
    Generator,
    sample_noise,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MEDGAN(Model):
    """
    MedGAN: Medical Generative Adversarial Network for tabular data synthesis.

    The model consists of:
    1. Autoencoder
    2. Generator
    3. Discriminator

    Training occurs in two phases:
    1. Pre-train the autoencoder
    2. Train the GAN
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        latent_dim: int = 128,
        generator_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 128,
        generator_num_layers: int = 2,
        discriminator_num_layers: int = 2,
        ae_pretrain_epochs: int = 100,
        gan_epochs: int = 1000,
        batch_size: int = 1000,
        ae_lr: float = 1e-3,
        generator_lr: float = 1e-3,
        discriminator_lr: float = 1e-3,
        dropout: float = 0.0,
        bn_decay: float = 0.99,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.generator_hidden_dim = generator_hidden_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.generator_num_layers = generator_num_layers
        self.discriminator_num_layers = discriminator_num_layers

        self.ae_pretrain_epochs = ae_pretrain_epochs
        self.gan_epochs = gan_epochs
        self.batch_size = batch_size
        self.ae_lr = ae_lr
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.dropout = dropout
        self.bn_decay = bn_decay
        self.random_state = random_state

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self.autoencoder = None
        self.generator = None
        self.discriminator = None
        self.input_dim_ = None

        self.columns_ = None
        self.feature_columns_ = None
        self.target_col_ = None

        self.categorical_cols_ = []
        self.continuous_cols_ = []
        self.category_values_ = {}

        self.data_min_ = None
        self.data_max_ = None

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: str | None = None,
        **kwargs,
    ):
        """
        Train MedGAN using x_train.csv and y_train.csv.
        """

        if synthetic_dir is None:
            synthetic_dir = os.path.join(
                dataset_dir,
                "synthetic",
            )

        logger.info("=" * 80)
        logger.info("Training MedGAN Model")
        logger.info("=" * 80)

        x_train_path = os.path.join(
            dataset_dir,
            "x_train.csv",
        )

        y_train_path = os.path.join(
            dataset_dir,
            "y_train.csv",
        )

        X_train = pd.read_csv(x_train_path)

        logger.info(
            f"Loaded training data: {X_train.shape}"
        )

        self.feature_columns_ = X_train.columns.tolist()

        if os.path.exists(y_train_path):
            y_train = pd.read_csv(y_train_path)

            self.target_col_ = y_train.columns[0]

            df_train = pd.concat(
                [X_train, y_train],
                axis=1,
            )
        else:
            y_train = None
            df_train = X_train.copy()

        original_df = df_train.copy()

        self.columns_ = df_train.columns.tolist()

        supplied_categorical = kwargs.get(
            "categorical_cols"
        )

        supplied_continuous = kwargs.get(
            "continuous_cols"
        )

        if supplied_categorical is not None:
            self.categorical_cols_ = [
                col
                for col in supplied_categorical
                if col in df_train.columns
            ]
        else:
            self.categorical_cols_ = []

            for col in df_train.columns:
                if (
                    pd.api.types.is_object_dtype(df_train[col])
                    or pd.api.types.is_string_dtype(df_train[col])
                    or pd.api.types.is_bool_dtype(df_train[col])
                    or isinstance(
                        df_train[col].dtype,
                        pd.CategoricalDtype,
                    )
                ):
                    self.categorical_cols_.append(col)

        if (
            self.target_col_ is not None
            and self.target_col_ not in self.categorical_cols_
        ):
            self.categorical_cols_.append(
                self.target_col_
            )

        if supplied_continuous is not None:
            self.continuous_cols_ = [
                col
                for col in supplied_continuous
                if col in df_train.columns
                and col not in self.categorical_cols_
            ]
        else:
            self.continuous_cols_ = [
                col
                for col in df_train.columns
                if col not in self.categorical_cols_
            ]

        logger.info(
            f"Categorical columns: {self.categorical_cols_}"
        )

        logger.info(
            f"Continuous columns: {self.continuous_cols_}"
        )

        encoded_df = df_train.copy()

        self.category_values_ = {}

        for col in self.categorical_cols_:
            categories = (
                encoded_df[col]
                .drop_duplicates()
                .tolist()
            )

            self.category_values_[col] = categories

            mapping = {
                value: index
                for index, value in enumerate(categories)
            }

            encoded_df[col] = (
                encoded_df[col]
                .map(mapping)
                .astype(float)
            )

        for col in self.continuous_cols_:
            encoded_df[col] = pd.to_numeric(
                encoded_df[col],
                errors="coerce",
            )

            if encoded_df[col].isna().any():
                median = encoded_df[col].median()

                encoded_df[col] = (
                    encoded_df[col]
                    .fillna(median)
                )

        data = encoded_df.values.astype(
            np.float32
        )

        self.input_dim_ = data.shape[1]

        self.data_min_ = data.min(axis=0)
        self.data_max_ = data.max(axis=0)

        data_range = (
            self.data_max_
            - self.data_min_
        )

        data_range[data_range == 0] = 1

        data_normalized = (
            data - self.data_min_
        ) / data_range

        logger.info(
            "Data normalized to [0, 1] range"
        )

        logger.info(
            f"Original range: "
            f"[{data.min():.2f}, {data.max():.2f}]"
        )

        logger.info(
            f"Normalized range: "
            f"[{data_normalized.min():.2f}, "
            f"{data_normalized.max():.2f}]"
        )

        self._fit(data_normalized)

        logger.info(
            f"\nGenerating {len(data)} synthetic samples..."
        )

        synth_df = self.sample(
            len(data)
        )

        os.makedirs(
            synthetic_dir,
            exist_ok=True,
        )

        if y_train is not None:
            y_name = self.target_col_

            x_synth = synth_df[
                self.feature_columns_
            ].copy()

            y_synth = synth_df[
                [y_name]
            ].copy()

            real_classes = set(
                original_df[y_name]
                .dropna()
                .tolist()
            )

            synthetic_classes = set(
                y_synth[y_name]
                .dropna()
                .tolist()
            )

            missing_classes = (
                real_classes
                - synthetic_classes
            )

            if missing_classes:
                logger.warning(
                    "Missing classes in synthetic data: "
                    f"{missing_classes}"
                )

                logger.info(
                    "Adding one existing training "
                    "sample for missing target classes..."
                )

                for cls in missing_classes:
                    matching_rows = original_df[
                        original_df[y_name] == cls
                    ]

                    if matching_rows.empty:
                        continue

                    row = matching_rows.iloc[[0]]

                    x_synth = pd.concat(
                        [
                            x_synth,
                            row[self.feature_columns_],
                        ],
                        ignore_index=True,
                    )

                    y_synth = pd.concat(
                        [
                            y_synth,
                            row[[y_name]],
                        ],
                        ignore_index=True,
                    )

            x_synth.to_csv(
                os.path.join(
                    synthetic_dir,
                    "x_synth.csv",
                ),
                index=False,
            )

            y_synth.to_csv(
                os.path.join(
                    synthetic_dir,
                    "y_synth.csv",
                ),
                index=False,
            )

        else:
            synth_df.to_csv(
                os.path.join(
                    synthetic_dir,
                    "x_synth.csv",
                ),
                index=False,
            )

        logger.info(
            f"\nSynthetic data saved to: "
            f"{synthetic_dir}"
        )

        logger.info(
            "Training complete!"
        )

        return self

    def _fit(
        self,
        data: np.ndarray,
    ):
        """
        Internal MedGAN fit method.
        """

        self.autoencoder = Autoencoder(
            input_dim=self.input_dim_,
            encoder_dim=self.encoder_dim,
            latent_dim=self.latent_dim,
            bn_decay=self.bn_decay,
        ).to(self.device)

        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dim=self.generator_hidden_dim,
            num_layers=self.generator_num_layers,
            bn_decay=self.bn_decay,
        ).to(self.device)

        self.discriminator = Discriminator(
            input_dim=self.input_dim_,
            hidden_dim=self.discriminator_hidden_dim,
            num_layers=self.discriminator_num_layers,
            dropout=self.dropout,
        ).to(self.device)

        logger.info(
            f"\nPhase 1: Pretraining Autoencoder "
            f"for {self.ae_pretrain_epochs} epochs..."
        )

        self._pretrain_autoencoder(
            data
        )

        logger.info(
            f"\nPhase 2: Training GAN "
            f"for {self.gan_epochs} epochs..."
        )

        self._train_gan(
            data
        )

    def _pretrain_autoencoder(
        self,
        data: np.ndarray,
    ):
        """
        Pretrain the autoencoder.
        """

        optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr=self.ae_lr,
        )

        criterion = nn.BCELoss()

        dataset = torch.tensor(
            data,
            dtype=torch.float32,
        )

        n_batches = (
            len(dataset)
            + self.batch_size
            - 1
        ) // self.batch_size

        for epoch in range(
            self.ae_pretrain_epochs
        ):
            self.autoencoder.train()

            total_loss = 0

            indices = torch.randperm(
                len(dataset)
            )

            for i in range(n_batches):
                batch_idx = indices[
                    i * self.batch_size:
                    (i + 1) * self.batch_size
                ]

                batch = dataset[
                    batch_idx
                ].to(self.device)

                optimizer.zero_grad()

                x_recon, _ = self.autoencoder(
                    batch
                )

                loss = criterion(
                    x_recon,
                    batch,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (
                (epoch + 1) % 10 == 0
                or epoch == 0
                or epoch == self.ae_pretrain_epochs - 1
            ):
                avg_loss = (
                    total_loss
                    / n_batches
                )

                logger.info(
                    f"Epoch "
                    f"{epoch + 1}/"
                    f"{self.ae_pretrain_epochs}: "
                    f"AE Loss = "
                    f"{avg_loss:.6f}"
                )

    def _train_gan(
        self,
        data: np.ndarray,
    ):
        """
        Train MedGAN using the paper-style adversarial process.
        """

        optimizer_g = optim.Adam(
            list(self.generator.parameters())
            + list(self.autoencoder.decoder_layer.parameters()),
            lr=self.generator_lr,
        )

        optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
        )

        criterion = nn.BCELoss()

        dataset = torch.tensor(
            data,
            dtype=torch.float32,
        )

        n_batches = (
            len(dataset)
            + self.batch_size
            - 1
        ) // self.batch_size

        for parameter in self.autoencoder.encoder_layer.parameters():
            parameter.requires_grad = False

        for parameter in self.autoencoder.decoder_layer.parameters():
            parameter.requires_grad = True

        discriminator_steps = 2

        for epoch in range(
            self.gan_epochs
        ):
            self.generator.train()
            self.discriminator.train()

            d_loss_total = 0.0
            g_loss_total = 0.0

            indices = torch.randperm(
                len(dataset)
            )

            for i in range(
                n_batches
            ):
                batch_idx = indices[
                    i * self.batch_size:
                    (i + 1) * self.batch_size
                ]

                real_data = dataset[
                    batch_idx
                ].to(
                    self.device
                )

                batch_len = len(
                    real_data
                )

                real_labels = torch.ones(
                    batch_len,
                    1,
                    device=self.device,
                )

                fake_labels = torch.zeros(
                    batch_len,
                    1,
                    device=self.device,
                )

                current_d_loss = 0.0

                for _ in range(
                    discriminator_steps
                ):
                    optimizer_d.zero_grad()

                    d_real = self.discriminator(
                        real_data
                    )

                    d_loss_real = criterion(
                        d_real,
                        real_labels,
                    )

                    noise = sample_noise(
                        batch_len,
                        self.latent_dim,
                        self.device,
                    )

                    fake_latent = self.generator(
                        noise
                    )

                    fake_data = self.autoencoder.decode(
                        fake_latent
                    )

                    d_fake = self.discriminator(
                        fake_data.detach()
                    )

                    d_loss_fake = criterion(
                        d_fake,
                        fake_labels,
                    )

                    d_loss = (
                        d_loss_real
                        + d_loss_fake
                    )

                    d_loss.backward()
                    optimizer_d.step()

                    current_d_loss += (
                        d_loss.item()
                    )

                optimizer_g.zero_grad()

                noise = sample_noise(
                    batch_len,
                    self.latent_dim,
                    self.device,
                )

                fake_latent = self.generator(
                    noise
                )

                fake_data = self.autoencoder.decode(
                    fake_latent
                )

                d_fake = self.discriminator(
                    fake_data
                )

                g_loss = criterion(
                    d_fake,
                    real_labels,
                )

                g_loss.backward()
                optimizer_g.step()

                d_loss_total += (
                    current_d_loss
                    / discriminator_steps
                )

                g_loss_total += (
                    g_loss.item()
                )

            if (
                (epoch + 1) % 100 == 0
                or epoch == 0
                or epoch == self.gan_epochs - 1
            ):
                avg_d_loss = (
                    d_loss_total
                    / n_batches
                )

                avg_g_loss = (
                    g_loss_total
                    / n_batches
                )

                logger.info(
                    f"Epoch "
                    f"{epoch + 1}/"
                    f"{self.gan_epochs}: "
                    f"D Loss = "
                    f"{avg_d_loss:.6f}, "
                    f"G Loss = "
                    f"{avg_g_loss:.6f}"
                )

    def sample(
        self,
        n: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Returns
        -------
        pd.DataFrame
            Synthetic data using the original
            column names and categorical values.
        """

        if (
            self.autoencoder is None
            or self.generator is None
        ):
            raise RuntimeError(
                "Model must be trained before sampling"
            )

        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        cuda_states = None

        if torch.cuda.is_available():
            cuda_states = (
                torch.cuda.get_rng_state_all()
            )

        try:
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        seed
                    )

            self.autoencoder.eval()
            self.generator.eval()

            with torch.no_grad():
                noise = sample_noise(
                    n,
                    self.latent_dim,
                    self.device,
                )

                fake_latent = self.generator(
                    noise
                )

                synthetic_data_normalized = (
                    self.autoencoder.decode(
                        fake_latent
                    )
                )

                synthetic_data_normalized = (
                    synthetic_data_normalized
                    .cpu()
                    .numpy()
                )

            data_range = (
                self.data_max_
                - self.data_min_
            )

            synthetic_data = (
                synthetic_data_normalized
                * data_range
                + self.data_min_
            )

            synthetic_df = pd.DataFrame(
                synthetic_data,
                columns=self.columns_,
            )

            for col in self.categorical_cols_:
                categories = (
                    self.category_values_.get(
                        col
                    )
                )

                if categories is None:
                    continue

                if len(categories) == 0:
                    continue

                codes = np.rint(
                    synthetic_df[col]
                    .to_numpy()
                ).astype(int)

                codes = np.clip(
                    codes,
                    0,
                    len(categories) - 1,
                )

                synthetic_df[col] = [
                    categories[code]
                    for code in codes
                ]

            return synthetic_df

        finally:
            np.random.set_state(
                numpy_state
            )

            torch.random.set_rng_state(
                torch_state
            )

            if (
                cuda_states is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(
                    cuda_states
                )

    def evaluate(self):
        """
        Evaluation is handled by
        SyntheticEvaluationPipeline.
        """
        pass