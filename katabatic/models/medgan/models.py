"""
Production-level MedGAN implementation for the Katabatic framework.

Based on:
"Generating Multi-label Discrete Patient Records using Generative
Adversarial Networks"
Choi et al. (2017)
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


logger = logging.getLogger(__name__)


class MEDGAN(Model):
    """
    MedGAN: Medical Generative Adversarial Network.

    Training contains two stages:

    1. Pre-train the autoencoder.
    2. Train the generator, decoder and discriminator.

    The class supports both:

        model.fit(X, y)

    and Katabatic pipeline training:

        model.train(dataset_dir)
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
        data_type: str = "binary",
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__()

        if data_type not in {"binary", "count"}:
            raise ValueError(
                "data_type must be either 'binary' or 'count'"
            )

        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim

        self.generator_hidden_dim = (
            generator_hidden_dim
        )

        self.discriminator_hidden_dim = (
            discriminator_hidden_dim
        )

        self.generator_num_layers = (
            generator_num_layers
        )

        self.discriminator_num_layers = (
            discriminator_num_layers
        )

        self.ae_pretrain_epochs = (
            ae_pretrain_epochs
        )

        self.gan_epochs = gan_epochs
        self.batch_size = batch_size

        self.ae_lr = ae_lr
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.dropout = dropout
        self.bn_decay = bn_decay
        self.data_type = data_type
        self.random_state = random_state

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        torch.manual_seed(
            self.random_state
        )

        np.random.seed(
            self.random_state
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.random_state
            )

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

    # ================================================================
    # Public fit interface
    # ================================================================

    def fit(
        self,
        X,
        y=None,
        categorical_cols=None,
        continuous_cols=None,
    ):
        """
        Fit MedGAN directly using feature and target data.

        Parameters
        ----------
        X
            pandas DataFrame or array-like feature data.

        y
            Optional Series, DataFrame or array-like target.

        categorical_cols
            Optional list of categorical feature names.

        continuous_cols
            Optional list of continuous feature names.

        Returns
        -------
        self
        """

        # ------------------------------------------------------------
        # Convert X to DataFrame
        # ------------------------------------------------------------

        if isinstance(X, pd.DataFrame):

            X_train = X.copy()

        elif isinstance(X, pd.Series):

            X_train = X.to_frame()

        else:

            X_array = np.asarray(X)

            if X_array.ndim == 1:

                X_array = X_array.reshape(
                    -1,
                    1,
                )

            if X_array.ndim != 2:

                raise ValueError(
                    "X must be two-dimensional"
                )

            X_train = pd.DataFrame(
                X_array,
                columns=[
                    f"feature_{i}"
                    for i in range(
                        X_array.shape[1]
                    )
                ],
            )

        self.feature_columns_ = (
            X_train.columns.tolist()
        )

        # ------------------------------------------------------------
        # Prepare target
        # ------------------------------------------------------------

        if y is not None:

            if isinstance(y, pd.DataFrame):

                if y.shape[1] != 1:

                    raise ValueError(
                        "y must contain exactly "
                        "one target column"
                    )

                y_train = y.copy()

            elif isinstance(y, pd.Series):

                target_name = (
                    y.name
                    if y.name is not None
                    else "target"
                )

                y_train = (
                    y.rename(
                        target_name
                    )
                    .to_frame()
                )

            else:

                y_array = np.asarray(y)

                if y_array.ndim == 2:

                    if y_array.shape[1] != 1:

                        raise ValueError(
                            "y must contain "
                            "one target column"
                        )

                    y_array = (
                        y_array.reshape(-1)
                    )

                elif y_array.ndim != 1:

                    raise ValueError(
                        "y must be one-dimensional"
                    )

                y_train = pd.DataFrame(
                    {
                        "target": y_array
                    }
                )

            self.target_col_ = (
                y_train.columns[0]
            )

            df_train = pd.concat(
                [
                    X_train.reset_index(
                        drop=True
                    ),
                    y_train.reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            )

        else:

            y_train = None

            self.target_col_ = None

            df_train = X_train.copy()

        self.columns_ = (
            df_train.columns.tolist()
        )

        # ------------------------------------------------------------
        # Determine categorical columns
        # ------------------------------------------------------------

        if categorical_cols is not None:

            self.categorical_cols_ = [
                col
                for col
                in categorical_cols
                if col in df_train.columns
            ]

        else:

            self.categorical_cols_ = []

            for col in df_train.columns:

                series = df_train[col]

                if (
                    pd.api.types.is_object_dtype(
                        series
                    )
                    or
                    pd.api.types.is_string_dtype(
                        series
                    )
                    or
                    pd.api.types.is_bool_dtype(
                        series
                    )
                    or isinstance(
                        series.dtype,
                        pd.CategoricalDtype,
                    )
                ):

                    self.categorical_cols_.append(
                        col
                    )

        # Target must remain discrete
        # for classification datasets.

        if (
            self.target_col_ is not None
            and self.target_col_
            not in self.categorical_cols_
        ):

            self.categorical_cols_.append(
                self.target_col_
            )

        # ------------------------------------------------------------
        # Determine continuous columns
        # ------------------------------------------------------------

        if continuous_cols is not None:

            self.continuous_cols_ = [
                col
                for col
                in continuous_cols
                if (
                    col in df_train.columns
                    and col
                    not in self.categorical_cols_
                )
            ]

        else:

            self.continuous_cols_ = [
                col
                for col
                in df_train.columns
                if col
                not in self.categorical_cols_
            ]

        logger.info(
            "Categorical columns: %s",
            self.categorical_cols_,
        )

        logger.info(
            "Continuous columns: %s",
            self.continuous_cols_,
        )

        # ------------------------------------------------------------
        # Encode categorical data
        # ------------------------------------------------------------

        encoded_df = df_train.copy()

        self.category_values_ = {}

        for col in self.categorical_cols_:

            categories = (
                encoded_df[col]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            if len(categories) == 0:

                raise ValueError(
                    f"Categorical column "
                    f"'{col}' contains "
                    f"no valid values"
                )

            self.category_values_[col] = (
                categories
            )

            mapping = {
                value: index
                for index, value
                in enumerate(categories)
            }

            encoded_df[col] = (
                encoded_df[col]
                .map(mapping)
            )

            if encoded_df[col].isna().any():

                raise ValueError(
                    f"Unable to encode "
                    f"categorical column '{col}'"
                )

            encoded_df[col] = (
                encoded_df[col]
                .astype(float)
            )

        # ------------------------------------------------------------
        # Continuous columns
        # ------------------------------------------------------------

        for col in self.continuous_cols_:

            encoded_df[col] = (
                pd.to_numeric(
                    encoded_df[col],
                    errors="coerce",
                )
            )

            if (
                encoded_df[col]
                .isna()
                .all()
            ):

                raise ValueError(
                    f"Column '{col}' "
                    f"contains no numeric values"
                )

            if (
                encoded_df[col]
                .isna()
                .any()
            ):

                median = (
                    encoded_df[col]
                    .median()
                )

                encoded_df[col] = (
                    encoded_df[col]
                    .fillna(median)
                )

        if encoded_df.isna().any().any():

            missing_columns = (
                encoded_df.columns[
                    encoded_df.isna().any()
                ].tolist()
            )

            raise ValueError(
                "Missing values remain in "
                f"{missing_columns}"
            )

        # ------------------------------------------------------------
        # Convert to NumPy
        # ------------------------------------------------------------

        data = (
            encoded_df
            .to_numpy(
                dtype=np.float32
            )
        )

        if len(data) < 2:

            raise ValueError(
                "MedGAN requires at least "
                "two training samples"
            )

        self.input_dim_ = (
            data.shape[1]
        )

        # ------------------------------------------------------------
        # Store range for inverse transformation
        # ------------------------------------------------------------

        self.data_min_ = (
            data.min(axis=0)
        )

        self.data_max_ = (
            data.max(axis=0)
        )

        data_range = (
            self.data_max_
            - self.data_min_
        )

        safe_range = (
            data_range.copy()
        )

        safe_range[
            safe_range == 0
        ] = 1.0

        # ------------------------------------------------------------
        # Normalize
        # ------------------------------------------------------------

        data_normalized = (
            data
            - self.data_min_
        ) / safe_range

        logger.info(
            "Data normalized to [0, 1] range"
        )

        logger.info(
            "Original range: "
            "[%.2f, %.2f]",
            float(data.min()),
            float(data.max()),
        )

        logger.info(
            "Normalized range: "
            "[%.2f, %.2f]",
            float(
                data_normalized.min()
            ),
            float(
                data_normalized.max()
            ),
        )

        # ------------------------------------------------------------
        # Train model
        # ------------------------------------------------------------

        self._fit(
            data_normalized
        )

        return self

    # ================================================================
    # Katabatic pipeline interface
    # ================================================================

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: str | None = None,
        **kwargs,
    ):
        """
        Train using Katabatic dataset directories.

        Expected files:

            x_train.csv
            y_train.csv
        """

        if synthetic_dir is None:

            synthetic_dir = (
                os.path.join(
                    dataset_dir,
                    "synthetic",
                )
            )

        x_train_path = (
            os.path.join(
                dataset_dir,
                "x_train.csv",
            )
        )

        y_train_path = (
            os.path.join(
                dataset_dir,
                "y_train.csv",
            )
        )

        if not os.path.exists(
            x_train_path
        ):

            raise FileNotFoundError(
                "Expected file not found: "
                f"{x_train_path}"
            )

        X_train = pd.read_csv(
            x_train_path
        )

        logger.info(
            "=" * 80
        )

        logger.info(
            "Training MedGAN Model"
        )

        logger.info(
            "=" * 80
        )

        logger.info(
            "Loaded training data: %s",
            X_train.shape,
        )

        if os.path.exists(
            y_train_path
        ):

            y_train = pd.read_csv(
                y_train_path
            )

            if y_train.shape[1] != 1:

                raise ValueError(
                    "y_train.csv must "
                    "contain exactly "
                    "one target column"
                )

        else:

            y_train = None

        # Use common fit implementation

        self.fit(
            X_train,
            y_train,
            categorical_cols=kwargs.get(
                "categorical_cols"
            ),
            continuous_cols=kwargs.get(
                "continuous_cols"
            ),
        )

        # ------------------------------------------------------------
        # Generate synthetic data
        # ------------------------------------------------------------

        logger.info(
            "Generating %s "
            "synthetic samples...",
            len(X_train),
        )

        synth_df = self.sample(
            len(X_train)
        )

        # ------------------------------------------------------------
        # Save synthetic data
        # ------------------------------------------------------------

        os.makedirs(
            synthetic_dir,
            exist_ok=True,
        )

        if self.target_col_ is not None:

            x_synth = (
                synth_df[
                    self.feature_columns_
                ]
                .copy()
            )

            y_synth = (
                synth_df[
                    [self.target_col_]
                ]
                .copy()
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
            "Synthetic data saved to: %s",
            synthetic_dir,
        )

        logger.info(
            "Training complete!"
        )

        return self

    # ================================================================
    # Internal model training
    # ================================================================

    def _fit(
        self,
        data: np.ndarray,
    ):
        """
        Internal MedGAN training.
        """

        # ------------------------------------------------------------
        # Autoencoder
        # ------------------------------------------------------------

        self.autoencoder = Autoencoder(
            input_dim=self.input_dim_,
            encoder_dim=self.encoder_dim,
            latent_dim=self.latent_dim,
            bn_decay=self.bn_decay,
            data_type=self.data_type,
        ).to(
            self.device
        )

        # ------------------------------------------------------------
        # Generator
        # ------------------------------------------------------------

        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dim=(
                self.generator_hidden_dim
            ),
            num_layers=(
                self.generator_num_layers
            ),
            bn_decay=self.bn_decay,
        ).to(
            self.device
        )

        # ------------------------------------------------------------
        # Discriminator
        # ------------------------------------------------------------

        self.discriminator = Discriminator(
            input_dim=self.input_dim_,
            hidden_dim=(
                self.discriminator_hidden_dim
            ),
            num_layers=(
                self.discriminator_num_layers
            ),
            dropout=self.dropout,
        ).to(
            self.device
        )

        logger.info(
            "Phase 1: Pretraining "
            "Autoencoder for %s epochs...",
            self.ae_pretrain_epochs,
        )

        self._pretrain_autoencoder(
            data
        )

        logger.info(
            "Phase 2: Training GAN "
            "for %s epochs...",
            self.gan_epochs,
        )

        self._train_gan(
            data
        )

        return self

    # ================================================================
    # Autoencoder pretraining
    # ================================================================

    def _pretrain_autoencoder(
        self,
        data: np.ndarray,
    ):
        """
        Pretrain the autoencoder.

        Binary:
            BCE loss.

        Count:
            MSE loss.
        """

        optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr=self.ae_lr,
        )

        if self.data_type == "binary":

            criterion = nn.BCELoss()

        elif self.data_type == "count":

            criterion = nn.MSELoss()

        else:

            raise ValueError(
                "data_type must be "
                "either 'binary' or 'count'"
            )

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

            total_loss = 0.0

            indices = torch.randperm(
                len(dataset)
            )

            for i in range(
                n_batches
            ):

                batch_idx = indices[
                    i * self.batch_size:
                    (i + 1)
                    * self.batch_size
                ]

                batch = dataset[
                    batch_idx
                ].to(
                    self.device
                )

                if len(batch) == 0:
                    continue

                optimizer.zero_grad()

                x_recon, _ = (
                    self.autoencoder(
                        batch
                    )
                )

                loss = criterion(
                    x_recon,
                    batch,
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                )

            if (
                (epoch + 1) % 10 == 0
                or epoch == 0
                or epoch
                == self.ae_pretrain_epochs - 1
            ):

                avg_loss = (
                    total_loss
                    / n_batches
                )

                logger.info(
                    "Epoch %s/%s: "
                    "AE Loss = %.6f",
                    epoch + 1,
                    self.ae_pretrain_epochs,
                    avg_loss,
                )

    # ================================================================
    # GAN training
    # ================================================================

    def _train_gan(
        self,
        data: np.ndarray,
    ):
        """
        Train MedGAN using the paper-style
        adversarial process.
        """

        # ------------------------------------------------------------
        # Generator + decoder optimizer
        # ------------------------------------------------------------

        optimizer_g = optim.Adam(
            list(
                self.generator.parameters()
            )
            + list(
                self.autoencoder
                .decoder_layer
                .parameters()
            ),
            lr=self.generator_lr,
        )

        # ------------------------------------------------------------
        # Discriminator optimizer
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Freeze encoder
        # ------------------------------------------------------------

        for parameter in (
            self.autoencoder
            .encoder_layer
            .parameters()
        ):

            parameter.requires_grad = False

        # ------------------------------------------------------------
        # Decoder remains trainable
        # ------------------------------------------------------------

        for parameter in (
            self.autoencoder
            .decoder_layer
            .parameters()
        ):

            parameter.requires_grad = True

        # MedGAN paper uses k = 2
        discriminator_steps = 2

        for epoch in range(
            self.gan_epochs
        ):

            self.generator.train()
            self.discriminator.train()

            d_loss_total = 0.0
            g_loss_total = 0.0

            used_batches = 0

            indices = torch.randperm(
                len(dataset)
            )

            for i in range(
                n_batches
            ):

                batch_idx = indices[
                    i * self.batch_size:
                    (i + 1)
                    * self.batch_size
                ]

                real_data = dataset[
                    batch_idx
                ].to(
                    self.device
                )

                batch_len = len(
                    real_data
                )

                # BatchNorm needs more
                # than one sample.

                if batch_len < 2:
                    continue

                used_batches += 1

                real_labels = (
                    torch.ones(
                        batch_len,
                        1,
                        device=self.device,
                    )
                )

                fake_labels = (
                    torch.zeros(
                        batch_len,
                        1,
                        device=self.device,
                    )
                )

                current_d_loss = 0.0

                # ----------------------------------------------------
                # Train discriminator twice
                # ----------------------------------------------------

                for _ in range(
                    discriminator_steps
                ):

                    optimizer_d.zero_grad()

                    # Real data

                    d_real = (
                        self.discriminator(
                            real_data
                        )
                    )

                    d_loss_real = (
                        criterion(
                            d_real,
                            real_labels,
                        )
                    )

                    # Synthetic data

                    noise = sample_noise(
                        batch_len,
                        self.latent_dim,
                        self.device,
                    )

                    fake_latent = (
                        self.generator(
                            noise
                        )
                    )

                    fake_data = (
                        self.autoencoder
                        .decode(
                            fake_latent
                        )
                    )

                    d_fake = (
                        self.discriminator(
                            fake_data.detach()
                        )
                    )

                    d_loss_fake = (
                        criterion(
                            d_fake,
                            fake_labels,
                        )
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

                # ----------------------------------------------------
                # Generator + decoder update
                # ----------------------------------------------------

                optimizer_g.zero_grad()

                noise = sample_noise(
                    batch_len,
                    self.latent_dim,
                    self.device,
                )

                fake_latent = (
                    self.generator(
                        noise
                    )
                )

                fake_data = (
                    self.autoencoder
                    .decode(
                        fake_latent
                    )
                )

                d_fake = (
                    self.discriminator(
                        fake_data
                    )
                )

                # Non-saturating GAN objective:
                # Generator wants D(fake) = 1

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

            if used_batches == 0:

                raise RuntimeError(
                    "No usable GAN batches. "
                    "At least two samples "
                    "are required in a batch."
                )

            if (
                (epoch + 1) % 100 == 0
                or epoch == 0
                or epoch
                == self.gan_epochs - 1
            ):

                avg_d_loss = (
                    d_loss_total
                    / used_batches
                )

                avg_g_loss = (
                    g_loss_total
                    / used_batches
                )

                logger.info(
                    "Epoch %s/%s: "
                    "D Loss = %.6f, "
                    "G Loss = %.6f",
                    epoch + 1,
                    self.gan_epochs,
                    avg_d_loss,
                    avg_g_loss,
                )

    # ================================================================
    # Sampling
    # ================================================================

    def sample(
        self,
        n: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Returns
        -------
        pandas.DataFrame
            Synthetic records using original
            column names and category values.
        """

        if n <= 0:

            raise ValueError(
                "n must be greater than 0"
            )

        if (
            self.autoencoder is None
            or self.generator is None
        ):

            raise RuntimeError(
                "Model must be trained "
                "before sampling"
            )

        if (
            self.columns_ is None
            or self.data_min_ is None
            or self.data_max_ is None
        ):

            raise RuntimeError(
                "Model training metadata "
                "is missing"
            )

        # ------------------------------------------------------------
        # Save random states
        # ------------------------------------------------------------

        numpy_state = (
            np.random.get_state()
        )

        torch_state = (
            torch.random
            .get_rng_state()
        )

        cuda_states = None

        if torch.cuda.is_available():

            cuda_states = (
                torch.cuda
                .get_rng_state_all()
            )

        try:

            # --------------------------------------------------------
            # Optional deterministic sample
            # --------------------------------------------------------

            if seed is not None:

                np.random.seed(seed)

                torch.manual_seed(seed)

                if torch.cuda.is_available():

                    torch.cuda.manual_seed_all(
                        seed
                    )

            self.autoencoder.eval()
            self.generator.eval()

            # --------------------------------------------------------
            # Generate
            # --------------------------------------------------------

            with torch.no_grad():

                noise = sample_noise(
                    n,
                    self.latent_dim,
                    self.device,
                )

                fake_latent = (
                    self.generator(
                        noise
                    )
                )

                synthetic_normalized = (
                    self.autoencoder
                    .decode(
                        fake_latent
                    )
                )

                synthetic_normalized = (
                    synthetic_normalized
                    .cpu()
                    .numpy()
                )

            # --------------------------------------------------------
            # Denormalize
            # --------------------------------------------------------

            data_range = (
                self.data_max_
                - self.data_min_
            )

            synthetic_data = (
                synthetic_normalized
                * data_range
                + self.data_min_
            )

            synthetic_df = (
                pd.DataFrame(
                    synthetic_data,
                    columns=self.columns_,
                )
            )

            # --------------------------------------------------------
            # Decode categorical columns
            # --------------------------------------------------------

            for col in (
                self.categorical_cols_
            ):

                categories = (
                    self.category_values_
                    .get(col)
                )

                if (
                    categories is None
                    or len(categories) == 0
                ):

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

            # --------------------------------------------------------
            # Restore random states
            # --------------------------------------------------------

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

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self):
        """
        Evaluation is handled by
        SyntheticEvaluationPipeline.
        """
        pass