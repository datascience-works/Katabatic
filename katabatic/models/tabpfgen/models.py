"""
TabPFGen model implementation for the Katabatic framework.

Based on:
Ma et al. (2024) - "TabPFGen: Tabular Data Generation with TabPFN"
https://arxiv.org/abs/2406.05216

Implementation reference:
https://github.com/sebhaan/TabPFGen

"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from tabpfn import TabPFNClassifier, TabPFNRegressor

try:
    from katabatic.models.base_model import Model
except ImportError:
    class Model:
        """Fallback base class when running outside the Katabatic package."""
        pass


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.preprocessing._encoders"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabPFGen:
    """
    TabPFGen: TabPFN-guided synthetic tabular data generator.

    TabPFGen is an energy-based approach for synthetic tabular data generation.
    The original paper uses a frozen pre-trained TabPFN model as part of the
    energy function and updates synthetic samples using SGLD-style sampling.

    This Katabatic implementation follows the same practical idea:
    1. Scale the real training features.
    2. Initialise synthetic samples from the real training distribution.
    3. Refine synthetic samples through SGLD-style updates.
    4. Use TabPFN to assign synthetic labels.
    5. Save synthetic outputs in Katabatic CSV format.

    Note:
        TabPFGen does not train for epochs like GAN-based models.
        The main generation control parameter is `n_sgld_steps`.
    """

    def __init__(
        self,
        n_sgld_steps: int = 300,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        device: str = "auto",
        random_state: int = 42,
    ):
        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.device = self._infer_device(device)

        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _infer_device(self, device: Optional[str]) -> torch.device:
        """
        Select the computation device.

        Args:
            device: "auto", "cpu", or "cuda".

        Returns:
            torch.device selected for generation.
        """
        if device is None or device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _compute_energy(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a differentiable energy score for synthetic samples.

        In the TabPFGen paper, class-conditional energy is linked to the
        TabPFN output for the selected synthetic label. In this practical
        Katabatic wrapper, a distance-based differentiable surrogate is used
        to keep synthetic samples close to the real training distribution.

        Lower energy means the synthetic sample is closer to real samples,
        especially samples from the same class.
        """
        distances = torch.cdist(x_synth, x_train)

        # General realism term: distance to the nearest real training sample.
        min_distances, _ = distances.min(dim=1)

        # Class-aware term: average distance to real samples with same label.
        class_mask = y_synth.unsqueeze(1) == y_train.unsqueeze(0)
        class_counts = class_mask.float().sum(dim=1).clamp(min=1.0)

        class_distances = (distances * class_mask.float()).sum(dim=1) / class_counts

        return min_distances + class_distances

    def _sgld_step(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one SGLD-style update step.

        SGLD updates synthetic samples using:
        - the gradient of the energy function
        - Gaussian noise for exploration and diversity
        """
        x_synth = x_synth.clone().detach().requires_grad_(True)

        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        grad = torch.autograd.grad(energy.sum(), x_synth, allow_unused=True)[0]

        if grad is None:
            grad = torch.zeros_like(x_synth)

        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.sgld_step_size)

        return (
            x_synth
            - self.sgld_step_size * grad
            + self.sgld_noise_scale * noise
        ).detach()

    def generate_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        balance_classes: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for classification datasets.

        Args:
            X_train: Real training features.
            y_train: Real training labels.
            n_samples: Number of synthetic rows to generate.
            balance_classes: Whether to generate approximately balanced classes.

        Returns:
            X_synth, y_synth as NumPy arrays.
        """
        logger.info("Starting TabPFGen classification generation")
        logger.info(f"SGLD steps: {self.n_sgld_steps}")
        logger.info(f"Requested synthetic samples: {n_samples}")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)

        X_scaled = self.scaler.fit_transform(X_train)

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train_t = torch.tensor(y_encoded, device=self.device, dtype=torch.long)

        classes = np.unique(y_encoded)
        x_synth_list = []
        y_synth_list = []

        if balance_classes:
            base_count = n_samples // len(classes)
            remainder = n_samples % len(classes)

            for i, cls in enumerate(classes):
                class_count = base_count + (1 if i < remainder else 0)
                idx = np.where(y_encoded == cls)[0]

                sample_idx = np.random.choice(idx, size=class_count, replace=True)

                x_init = (
                    x_train[sample_idx]
                    + torch.randn(
                        class_count,
                        X_train.shape[1],
                        device=self.device
                    ) * 0.01
                )

                y_init = torch.full(
                    (class_count,),
                    int(cls),
                    device=self.device,
                    dtype=torch.long
                )

                x_synth_list.append(x_init)
                y_synth_list.append(y_init)

            x_synth = torch.cat(x_synth_list, dim=0)
            y_synth = torch.cat(y_synth_list, dim=0)

        else:
            x_synth = torch.randn(
                n_samples,
                X_train.shape[1],
                device=self.device,
                dtype=torch.float32
            ) * 0.01

            y_synth = torch.randint(
                low=0,
                high=len(classes),
                size=(n_samples,),
                device=self.device,
                dtype=torch.long
            )

        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train_t)

            if (step + 1) % 100 == 0 or step == 0:
                logger.info(f"SGLD step {step + 1}/{self.n_sgld_steps}")

        x_synth_np = x_synth.detach().cpu().numpy()

        # TabPFN assigns final labels to generated samples.
        # TabPFN is fitted to the real training data.
        clf = TabPFNClassifier(device=self.device.type)
        clf.fit(X_scaled, y_train)

        if balance_classes:
            # Keep the assigned synthetic labels when class balancing is enabled.
            # This prevents synthetic labels from collapsing into the majority class,
            # which is important for imbalanced datasets such as car and nursery.
            y_synth_np = label_encoder.inverse_transform(
                y_synth.detach().cpu().numpy().astype(int)
            )
        else:
            # If class balancing is disabled, let TabPFN assign final labels.
            probs = clf.predict_proba(x_synth_np)
            y_pred_indices = probs.argmax(axis=1)
            y_synth_np = clf.classes_[y_pred_indices]

        X_synth = self.scaler.inverse_transform(x_synth_np)

        logger.info("Classification synthetic generation complete")

        return X_synth, y_synth_np

    def generate_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for regression datasets.

        Args:
            X_train: Real training features.
            y_train: Real target values.
            n_samples: Number of synthetic rows to generate.

        Returns:
            X_synth, y_synth as NumPy arrays.
        """
        logger.info("Starting TabPFGen regression generation")
        logger.info(f"SGLD steps: {self.n_sgld_steps}")

        X_scaled = self.scaler.fit_transform(X_train)

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_placeholder = torch.zeros(len(X_scaled), device=self.device)

        x_synth = torch.randn(
            n_samples,
            X_train.shape[1],
            device=self.device,
            dtype=torch.float32
        )

        y_synth_placeholder = torch.zeros(n_samples, device=self.device)

        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(
                x_synth,
                y_synth_placeholder,
                x_train,
                y_placeholder,
            )

            if (step + 1) % 100 == 0 or step == 0:
                logger.info(f"SGLD step {step + 1}/{self.n_sgld_steps}")

        x_synth_np = x_synth.detach().cpu().numpy()

        regressor = TabPFNRegressor(device=self.device.type)
        regressor.fit(X_scaled, y_train)

        y_synth = regressor.predict(x_synth_np)
        X_synth = self.scaler.inverse_transform(x_synth_np)

        logger.info("Regression synthetic generation complete")

        return X_synth, y_synth


class TabPFGenModel(Model):
    """
    Katabatic wrapper for TabPFGen.

    This wrapper connects the TabPFGen generator to the Katabatic pipeline.
    It expects the pipeline to provide:
    - x_train.csv
    - y_train.csv

    It writes:
    - x_synth.csv
    - y_synth.csv
    - synthetic.csv
    """

    def __init__(
        self,
        task: str = "classification",
        n_sgld_steps: int = 300,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        balance_classes: bool = True,
        device: str = "auto",
        y_col: Optional[str] = None,
        random_state: int = 42,
    ):
        super().__init__()

        self.task = task
        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.balance_classes = balance_classes
        self.device = device
        self.y_col = y_col
        self.random_state = random_state

        # Populated by train(); used by sample()
        self._X_train_encoded = None
        self._y_train = None
        self._feature_cols = None
        self._label_col = None
        self._cat_cols = []
        self._ordinal_encoder = None

    def _make_generator(self, seed: int) -> TabPFGen:
        return TabPFGen(
            n_sgld_steps=self.n_sgld_steps,
            sgld_step_size=self.sgld_step_size,
            sgld_noise_scale=self.sgld_noise_scale,
            device=self.device,
            random_state=seed,
        )

    def _decode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round ordinal-encoded floats to valid indices then inverse-transform to strings."""
        if not self._cat_cols or self._ordinal_encoder is None:
            return df
        cat_data = df[self._cat_cols].values.copy()
        for i in range(len(self._cat_cols)):
            n_cats = len(self._ordinal_encoder.categories_[i])
            cat_data[:, i] = np.round(cat_data[:, i]).clip(0, n_cats - 1)
        df = df.copy()
        df[self._cat_cols] = self._ordinal_encoder.inverse_transform(cat_data)
        return df

    def train(
        self,
        output_dir: str,
        *args,
        categorical_cols: list = None,
        continuous_cols: list = None,
        **kwargs,
    ):
        """
        Generate synthetic data using Katabatic pipeline outputs.

        Args:
            output_dir: Directory containing x_train.csv and y_train.csv.
            categorical_cols: Categorical column names. When provided, these are
                used for ordinal encoding instead of dtype-based auto-detection.
            continuous_cols: Continuous column names (informational; categorical_cols
                takes precedence for encoding decisions).
            **kwargs: Additional keyword arguments. May include synthetic_dir.

        Returns:
            Dictionary containing paths to generated synthetic files.
        """
        logger.info("=" * 80)
        logger.info("Running TabPFGenModel")
        logger.info("=" * 80)

        output_dir = Path(output_dir)
        synthetic_dir = Path(kwargs.get("synthetic_dir", output_dir / "synthetic"))
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        x_train_path = output_dir / "x_train.csv"
        y_train_path = output_dir / "y_train.csv"

        if not x_train_path.exists():
            raise FileNotFoundError(f"Missing required file: {x_train_path}")
        if not y_train_path.exists():
            raise FileNotFoundError(f"Missing required file: {y_train_path}")

        X_train_df = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)

        y_train = y_train_df.squeeze().to_numpy()

        # Use explicitly provided categorical_cols when available; fall back to
        # dtype-based detection so the model works without them if needed.
        if categorical_cols is not None:
            cat_cols = [c for c in categorical_cols if c in X_train_df.columns]
        else:
            cat_cols = [
                c for c in X_train_df.columns
                if not pd.api.types.is_numeric_dtype(X_train_df[c])
            ]

        if cat_cols:
            ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_encoded = X_train_df.copy()
            X_encoded[cat_cols] = ordinal_encoder.fit_transform(X_train_df[cat_cols])
            X_train = X_encoded.to_numpy().astype(float)
        else:
            ordinal_encoder = None
            X_train = X_train_df.to_numpy().astype(float)

        self._cat_cols = cat_cols
        self._ordinal_encoder = ordinal_encoder
        self._X_train_encoded = X_train
        self._y_train = y_train
        self._feature_cols = X_train_df.columns.tolist()

        logger.info(f"Loaded X_train shape: {X_train_df.shape}")
        logger.info(f"Loaded y_train length: {len(y_train)}")
        logger.info(f"Task type: {self.task}")
        logger.info(f"Output directory: {synthetic_dir}")

        gen = self._make_generator(self.random_state)

        if self.task == "classification":
            X_syn, y_syn = gen.generate_classification(
                X_train=X_train,
                y_train=y_train,
                n_samples=len(y_train),
                balance_classes=self.balance_classes,
            )
            label_col = self.y_col or y_train_df.columns[0] if len(y_train_df.columns) > 0 else "class"

        elif self.task == "regression":
            X_syn, y_syn = gen.generate_regression(
                X_train=X_train,
                y_train=y_train,
                n_samples=len(y_train),
            )
            label_col = self.y_col or y_train_df.columns[0] if len(y_train_df.columns) > 0 else "target"

        else:
            raise ValueError(
                "Invalid task type. Expected 'classification' or 'regression'."
            )

        self._label_col = label_col

        x_synth_df = self._decode_categoricals(
            pd.DataFrame(X_syn, columns=X_train_df.columns)
        )
        y_synth_df = pd.DataFrame({label_col: y_syn})

        synthetic_full_df = x_synth_df.copy()
        synthetic_full_df[label_col] = y_syn

        x_synth_path = synthetic_dir / "x_synth.csv"
        y_synth_path = synthetic_dir / "y_synth.csv"
        synthetic_csv_path = synthetic_dir / "synthetic.csv"

        x_synth_df.to_csv(x_synth_path, index=False)
        y_synth_df.to_csv(y_synth_path, index=False)
        synthetic_full_df.to_csv(synthetic_csv_path, index=False)

        logger.info(f"Saved x_synth.csv: {x_synth_path}")
        logger.info(f"Saved y_synth.csv: {y_synth_path}")
        logger.info(f"Saved synthetic.csv: {synthetic_csv_path}")
        logger.info("TabPFGen generation complete")

        return {
            "synthetic_csv": str(synthetic_csv_path),
            "x_synth_csv": str(x_synth_path),
            "y_synth_csv": str(y_synth_path),
            "n_sgld_steps": self.n_sgld_steps,
            "random_state": self.random_state,
        }

    def sample(self, n_samples: int, seed: int = None, **kwargs) -> pd.DataFrame:
        """
        Generate n_samples synthetic rows using the training data stored by train().

        Accepts an optional seed so the stability evaluator can produce
        multiple independent runs with reproducible variance.
        """
        if self._X_train_encoded is None:
            raise RuntimeError("Call train() before sample().")

        gen = self._make_generator(seed if seed is not None else self.random_state)

        if self.task == "classification":
            X_syn, y_syn = gen.generate_classification(
                X_train=self._X_train_encoded,
                y_train=self._y_train,
                n_samples=n_samples,
                balance_classes=self.balance_classes,
            )
        elif self.task == "regression":
            X_syn, y_syn = gen.generate_regression(
                X_train=self._X_train_encoded,
                y_train=self._y_train,
                n_samples=n_samples,
            )
        else:
            raise ValueError(f"Invalid task: {self.task!r}")

        synth_df = self._decode_categoricals(
            pd.DataFrame(X_syn, columns=self._feature_cols)
        )
        synth_df[self._label_col] = y_syn
        return synth_df
