# """
# Production-level MedGAN implementation for the Katabatic framework.

# Based on "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks"
# by Choi et al. (2017) - https://arxiv.org/abs/1703.06490
# """

# import logging
# import os

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from katabatic.models.base_model import Model
# from katabatic.models.medgan.utils import (
#     Autoencoder,
#     Discriminator,
#     Generator,
#     sample_noise,
# )

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class MEDGAN(Model):
#     """
#     MedGAN: Medical Generative Adversarial Network for tabular data synthesis.

#     This model consists of three main components:
#     1. Autoencoder (AE) - Compresses high-dimensional binary/count data
#     2. Generator (G) - Generates synthetic data in the AE's latent space
#     3. Discriminator (D) - Distinguishes real from synthetic latent representations

#     The model is trained in two phases:
#     1. Pre-train the autoencoder on real data
#     2. Train the GAN (Generator + Discriminator) in the latent space
#     """

#     def __init__(
#         self,
#         # Architecture hyperparameters
#         encoder_dim: int = 128,
#         latent_dim: int = 128,
#         generator_hidden_dim: int = 128,
#         discriminator_hidden_dim: int = 128,
#         generator_num_layers: int = 2,
#         discriminator_num_layers: int = 2,
#         # Training hyperparameters
#         ae_pretrain_epochs: int = 10, #100,
#         gan_epochs: int = 10, #1000,
#         batch_size: int = 1000,
#         ae_lr: float = 1e-3,
#         generator_lr: float = 1e-3,
#         discriminator_lr: float = 1e-3,
#         # Regularization
#         dropout: float = 0.1,
#         bn_decay: float = 0.99,
#         # Other
#         random_state: int = 42,
#         device: str | None = None,
#     ):
#         super().__init__()

#         # Hyperparameters
#         self.encoder_dim = encoder_dim
#         self.latent_dim = latent_dim
#         self.generator_hidden_dim = generator_hidden_dim
#         self.discriminator_hidden_dim = discriminator_hidden_dim
#         self.generator_num_layers = generator_num_layers
#         self.discriminator_num_layers = discriminator_num_layers

#         self.ae_pretrain_epochs = ae_pretrain_epochs
#         self.gan_epochs = gan_epochs
#         self.batch_size = batch_size
#         self.ae_lr = ae_lr
#         self.generator_lr = generator_lr
#         self.discriminator_lr = discriminator_lr

#         self.dropout = dropout
#         self.bn_decay = bn_decay
#         self.random_state = random_state

#         # Device
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)

#         # Set random seeds
#         torch.manual_seed(random_state)
#         np.random.seed(random_state)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(random_state)

#         # Models (will be initialized in train)
#         self.autoencoder = None
#         self.generator = None
#         self.discriminator = None
#         self.input_dim_ = None

#     def train(self, dataset_dir: str, synthetic_dir: str | None = None, **kwargs):
#         """
#         Train MedGAN model following Katabatic framework.

#         Args:
#             dataset_dir: Directory containing x_train.csv and y_train.csv
#             synthetic_dir: Directory to save synthetic data (defaults to
#                 ``{dataset_dir}/synthetic`` when omitted, e.g. legacy pipeline).
#             **kwargs: Additional arguments
#         """
#         if synthetic_dir is None:
#             synthetic_dir = os.path.join(dataset_dir, "synthetic")
#         logger.info("=" * 80)
#         logger.info("Training MedGAN Model")
#         logger.info("=" * 80)

#         # Load training data
#         x_train_path = os.path.join(dataset_dir, "x_train.csv")
#         y_train_path = os.path.join(dataset_dir, "y_train.csv")

#         X_train = pd.read_csv(x_train_path)
#         logger.info(f"Loaded training data: {X_train.shape}")

#         # Combine with y_train if exists
#         if os.path.exists(y_train_path):
#             y_train = pd.read_csv(y_train_path)
#             df_train = pd.concat([X_train, y_train], axis=1)
#         else:
#             df_train = X_train

#         # Convert categorical columns to numeric codes for MedGAN only
#         for col in df_train.columns:
#             if df_train[col].dtype == "object":
#                 df_train[col] = (
#                     df_train[col]
#                     .astype("category")
#                     .cat.codes
#         )

#         # Convert to numpy and normalize to [0, 1]
#         data = df_train.values.astype(np.float32)
#         self.input_dim_ = data.shape[1]

#         # Store min/max for denormalization
#         self.data_min_ = data.min(axis=0)
#         self.data_max_ = data.max(axis=0)

#         # Normalize to [0, 1] range (required for BCE loss)
#         data_range = self.data_max_ - self.data_min_
#         data_range[data_range == 0] = 1  # Avoid division by zero
#         data_normalized = (data - self.data_min_) / data_range

#         logger.info("Data normalized to [0, 1] range")
#         logger.info(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
#         logger.info(
#             f"Normalized range: [{data_normalized.min():.2f}, {data_normalized.max():.2f}]"
#         )

#         # Train the model
#         self._fit(data_normalized)

#         # Generate synthetic data
#         logger.info(f"\nGenerating {len(data)} synthetic samples...")
#         synth_data = self.sample(len(data))

#         # Round categorical columns to integers
#         # Assume all columns are categorical/discrete for tabular data
#         synth_data = np.round(synth_data)

#         # Save synthetic data
#         os.makedirs(synthetic_dir, exist_ok=True)

#         if os.path.exists(y_train_path):
#             # Split back into X and y
#             y_name = y_train.columns[0]
#             x_synth = pd.DataFrame(synth_data[:, :-1], columns=X_train.columns)
#             y_synth = pd.DataFrame(synth_data[:, -1:], columns=[y_name])

#             # Ensure all training classes are present in synthetic data
#             unique_train_classes = np.unique(df_train[y_name].values)
#             unique_synth_classes = np.unique(y_synth[y_name].values)
#             missing_classes = set(unique_train_classes) - set(unique_synth_classes)

#             if missing_classes:
#                 logger.warning(f"Missing classes in synthetic data: {missing_classes}")
#                 logger.info("Adding dummy samples to ensure all classes are present...")

#                 # Add one sample for each missing class
#                 for cls in missing_classes:
#                     # Find a training sample with this class
#                     cls_idx = np.where(df_train[y_name].values == cls)[0][0]
#                     dummy_row = df_train.iloc[cls_idx : cls_idx + 1].values

#                     # Append to synthetic data
#                     dummy_x = pd.DataFrame(dummy_row[:, :-1], columns=X_train.columns)
#                     dummy_y = pd.DataFrame(dummy_row[:, -1:], columns=[y_name])
#                     x_synth = pd.concat([x_synth, dummy_x], ignore_index=True)
#                     y_synth = pd.concat([y_synth, dummy_y], ignore_index=True)

#                 logger.info(f"Added {len(missing_classes)} dummy samples")

#             # Convert to int to ensure proper class labels
#             for col in X_train.columns:
#                 x_synth[col] = x_synth[col].astype(int)
#             y_synth[y_name] = y_synth[y_name].astype(int)

#             x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
#             y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)
#         else:
#             synth_df = pd.DataFrame(synth_data, columns=df_train.columns)
#             # Convert all columns to int
#             for col in synth_df.columns:
#                 synth_df[col] = synth_df[col].astype(int)
#             synth_df.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)

#         logger.info(f"\nSynthetic data saved to: {synthetic_dir}")
#         logger.info("Training complete!")

#         return self

#     def _fit(self, data: np.ndarray):
#         """Internal fit method."""
#         # Initialize models
#         self.autoencoder = Autoencoder(
#             input_dim=self.input_dim_,
#             encoder_dim=self.encoder_dim,
#             latent_dim=self.latent_dim,
#             bn_decay=self.bn_decay,
#         ).to(self.device)

#         self.generator = Generator(
#             latent_dim=self.latent_dim,
#             hidden_dim=self.generator_hidden_dim,
#             num_layers=self.generator_num_layers,
#             bn_decay=self.bn_decay,
#         ).to(self.device)

#         self.discriminator = Discriminator(
#             latent_dim=self.latent_dim,
#             hidden_dim=self.discriminator_hidden_dim,
#             num_layers=self.discriminator_num_layers,
#             dropout=self.dropout,
#         ).to(self.device)

#         # Phase 1: Pretrain Autoencoder
#         logger.info(
#             f"\nPhase 1: Pretraining Autoencoder for {self.ae_pretrain_epochs} epochs..."
#         )
#         self._pretrain_autoencoder(data)

#         # Phase 2: Train GAN
#         logger.info(f"\nPhase 2: Training GAN for {self.gan_epochs} epochs...")
#         self._train_gan(data)

#     def _pretrain_autoencoder(self, data: np.ndarray):
#         """Pretrain the autoencoder."""
#         optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
#         criterion = nn.BCELoss()

#         dataset = torch.tensor(data, dtype=torch.float32)
#         n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

#         for epoch in range(self.ae_pretrain_epochs):
#             self.autoencoder.train()
#             total_loss = 0

#             indices = torch.randperm(len(dataset))
#             for i in range(n_batches):
#                 batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
#                 batch = dataset[batch_idx].to(self.device)

#                 optimizer.zero_grad()
#                 x_recon, _ = self.autoencoder(batch)
#                 loss = criterion(x_recon, batch)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()

#             if (epoch + 1) % 10 == 0 or epoch == 0:
#                 avg_loss = total_loss / n_batches
#                 logger.info(
#                     f"Epoch {epoch + 1}/{self.ae_pretrain_epochs}: AE Loss = {avg_loss:.6f}"
#                 )

#     def _train_gan(self, data: np.ndarray):
#         """Train the GAN in the latent space."""
#         optimizer_g = optim.Adam(self.generator.parameters(), lr=self.generator_lr)
#         optimizer_d = optim.Adam(
#             self.discriminator.parameters(), lr=self.discriminator_lr
#         )
#         criterion = nn.BCELoss()

#         dataset = torch.tensor(data, dtype=torch.float32)
#         n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

#         self.autoencoder.eval()  # Freeze autoencoder

#         for epoch in range(self.gan_epochs):
#             self.generator.train()
#             self.discriminator.train()

#             d_loss_total = 0
#             g_loss_total = 0

#             indices = torch.randperm(len(dataset))
#             for i in range(n_batches):
#                 batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
#                 real_data = dataset[batch_idx].to(self.device)
#                 batch_len = len(real_data)

#                 # Get real latent representations
#                 with torch.no_grad():
#                     real_latent = self.autoencoder.encode(real_data)

#                 # Train Discriminator
#                 optimizer_d.zero_grad()

#                 # Real samples
#                 real_labels = torch.ones(batch_len, 1, device=self.device)
#                 d_real = self.discriminator(real_latent)
#                 d_loss_real = criterion(d_real, real_labels)

#                 # Fake samples
#                 noise = sample_noise(batch_len, self.latent_dim, self.device)
#                 fake_latent = self.generator(noise)
#                 fake_labels = torch.zeros(batch_len, 1, device=self.device)
#                 d_fake = self.discriminator(fake_latent.detach())
#                 d_loss_fake = criterion(d_fake, fake_labels)

#                 d_loss = d_loss_real + d_loss_fake
#                 d_loss.backward()
#                 optimizer_d.step()

#                 # Train Generator
#                 optimizer_g.zero_grad()

#                 noise = sample_noise(batch_len, self.latent_dim, self.device)
#                 fake_latent = self.generator(noise)
#                 d_fake = self.discriminator(fake_latent)
#                 # Generator wants D to predict "real"
#                 g_loss = criterion(d_fake, real_labels)

#                 g_loss.backward()
#                 optimizer_g.step()

#                 d_loss_total += d_loss.item()
#                 g_loss_total += g_loss.item()

#             if (epoch + 1) % 100 == 0 or epoch == 0:
#                 avg_d_loss = d_loss_total / n_batches
#                 avg_g_loss = g_loss_total / n_batches
#                 logger.info(
#                     f"Epoch {epoch + 1}/{self.gan_epochs}: D Loss = {avg_d_loss:.6f}, G Loss = {avg_g_loss:.6f}"
#                 )

#     def sample(self, n: int) -> np.ndarray:
#         """
#         Generate synthetic samples.

#         Args:
#             n: Number of samples to generate

#         Returns:
#             Synthetic data as numpy array (denormalized to original range)
#         """
#         if self.autoencoder is None or self.generator is None:
#             raise RuntimeError("Model must be trained before sampling")

#         self.autoencoder.eval()
#         self.generator.eval()

#         with torch.no_grad():
#             # Generate noise and pass through generator
#             noise = sample_noise(n, self.latent_dim, self.device)
#             fake_latent = self.generator(noise)

#             # Decode latent representation to data space (normalized [0, 1])
#             synthetic_data_normalized = self.autoencoder.decode(fake_latent)
#             synthetic_data_normalized = synthetic_data_normalized.cpu().numpy()

#         # Denormalize to original range
#         data_range = self.data_max_ - self.data_min_
#         synthetic_data = synthetic_data_normalized * data_range + self.data_min_

#         return synthetic_data

#     def evaluate(self):
#         """Evaluate is handled by the pipeline's TSTREvaluation."""

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
    2. Train the GAN in latent space
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        latent_dim: int = 128,
        generator_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 128,
        generator_num_layers: int = 2,
        discriminator_num_layers: int = 2,
        ae_pretrain_epochs: int = 10,  # Original default: 100
        gan_epochs: int = 10,  # Original default: 1000
        batch_size: int = 1000,
        ae_lr: float = 1e-3,
        generator_lr: float = 1e-3,
        discriminator_lr: float = 1e-3,
        dropout: float = 0.1,
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

        # Information required to reconstruct proper DataFrames
        self.columns_ = None
        self.feature_columns_ = None
        self.target_col_ = None

        # MedGAN-specific data representation
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

        # Combine X and y
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

        # Keep original representation for later decoding
        original_df = df_train.copy()

        self.columns_ = df_train.columns.tolist()

        # ---------------------------------------------------------
        # Identify categorical / continuous columns
        # ---------------------------------------------------------

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

        # Classification target must remain discrete
        if (
            self.target_col_ is not None
            and self.target_col_
            not in self.categorical_cols_
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

        # ---------------------------------------------------------
        # Encode categorical columns inside MedGAN only
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Ensure continuous columns are numeric
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Convert to NumPy
        # ---------------------------------------------------------

        data = encoded_df.values.astype(
            np.float32
        )

        self.input_dim_ = data.shape[1]

        # ---------------------------------------------------------
        # Normalisation
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Train model
        # ---------------------------------------------------------

        self._fit(data_normalized)

        # ---------------------------------------------------------
        # Generate synthetic data
        # ---------------------------------------------------------

        logger.info(
            f"\nGenerating {len(data)} synthetic samples..."
        )

        synth_df = self.sample(
            len(data)
        )

        # ---------------------------------------------------------
        # Save synthetic data
        # ---------------------------------------------------------

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

            # Ensure all target classes are represented
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
            latent_dim=self.latent_dim,
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
        Train GAN in latent space.
        """

        optimizer_g = optim.Adam(
            self.generator.parameters(),
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

        self.autoencoder.eval()

        for epoch in range(
            self.gan_epochs
        ):
            self.generator.train()
            self.discriminator.train()

            d_loss_total = 0
            g_loss_total = 0

            indices = torch.randperm(
                len(dataset)
            )

            for i in range(n_batches):
                batch_idx = indices[
                    i * self.batch_size:
                    (i + 1) * self.batch_size
                ]

                real_data = dataset[
                    batch_idx
                ].to(self.device)

                batch_len = len(
                    real_data
                )

                with torch.no_grad():
                    real_latent = (
                        self.autoencoder.encode(
                            real_data
                        )
                    )

                # -----------------------------
                # Train discriminator
                # -----------------------------

                optimizer_d.zero_grad()

                real_labels = torch.ones(
                    batch_len,
                    1,
                    device=self.device,
                )

                d_real = self.discriminator(
                    real_latent
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

                fake_labels = torch.zeros(
                    batch_len,
                    1,
                    device=self.device,
                )

                d_fake = self.discriminator(
                    fake_latent.detach()
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

                # -----------------------------
                # Train generator
                # -----------------------------

                optimizer_g.zero_grad()

                noise = sample_noise(
                    batch_len,
                    self.latent_dim,
                    self.device,
                )

                fake_latent = self.generator(
                    noise
                )

                d_fake = self.discriminator(
                    fake_latent
                )

                g_loss = criterion(
                    d_fake,
                    real_labels,
                )

                g_loss.backward()
                optimizer_g.step()

                d_loss_total += (
                    d_loss.item()
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

        # Save current RNG states
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

            # Denormalize
            data_range = (
                self.data_max_
                - self.data_min_
            )

            synthetic_data = (
                synthetic_data_normalized
                * data_range
                + self.data_min_
            )

            # Convert to DataFrame
            synthetic_df = pd.DataFrame(
                synthetic_data,
                columns=self.columns_,
            )

            # Decode categorical columns
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
            # Restore random states
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