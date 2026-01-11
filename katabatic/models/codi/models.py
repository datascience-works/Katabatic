"""
CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis

This implementation follows the paper:
"CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis"
https://arxiv.org/abs/2304.12654

CoDi uses two co-evolving diffusion models:
- One for continuous features
- One for categorical features
With contrastive learning to ensure coherent generation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, r2_score

from katabatic.models.base_model import Model
from katabatic.models.codi.utils import (
    infer_schema,
    encode_dataframe,
    decode_dataframe,
    save_metadata,
    load_metadata,
    set_global_seed,
    GaussianDiffusionTrainer,
    GaussianDiffusionSampler,
    MultinomialDiffusion,
    TabularUNet,
    get_device
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CODI(Model):
    """
    CoDi model for tabular synthetic data generation.

    Uses co-evolving diffusion models for continuous and categorical features
    with contrastive learning.
    """

    def __init__(
        self,
        # Diffusion hyperparameters
        n_steps: int = 50,
        beta_1: float = 0.00001,
        beta_T: float = 0.02,

        # Network architecture
        encoder_dim_con: Tuple[int, ...] = (64, 128, 256),
        encoder_dim_dis: Tuple[int, ...] = (64, 128, 256),
        nf_con: int = 16,
        nf_dis: int = 64,
        activation: str = 'relu',

        # Training hyperparameters
        epochs: int = 30,
        batch_size: int = 512,
        lr_con: float = 2e-3,
        lr_dis: float = 2e-3,
        grad_clip: float = 1.0,

        # Contrastive learning
        lambda_con: float = 0.2,
        lambda_dis: float = 0.2,

        # Other
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize CoDi model.

        Args:
            n_steps: Number of diffusion timesteps
            beta_1: Start beta value for diffusion schedule
            beta_T: End beta value for diffusion schedule
            encoder_dim_con: Encoder dimensions for continuous model
            encoder_dim_dis: Encoder dimensions for discrete model
            nf_con: Number of features in continuous embeddings
            nf_dis: Number of features in discrete embeddings
            activation: Activation function ('relu', 'elu', 'lrelu', 'swish')
            epochs: Training epochs
            batch_size: Training batch size
            lr_con: Learning rate for continuous model
            lr_dis: Learning rate for discrete model
            grad_clip: Gradient clipping value
            lambda_con: Weight for contrastive loss (continuous)
            lambda_dis: Weight for contrastive loss (discrete)
            random_state: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.n_steps = n_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.encoder_dim_con = encoder_dim_con
        self.encoder_dim_dis = encoder_dim_dis
        self.nf_con = nf_con
        self.nf_dis = nf_dis
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_con = lr_con
        self.lr_dis = lr_dis
        self.grad_clip = grad_clip
        self.lambda_con = lambda_con
        self.lambda_dis = lambda_dis
        self.random_state = random_state
        self.device = get_device(device)

        # Will be set during training
        self.schema_ = None
        self.model_con_ = None
        self.model_dis_ = None
        self.trainer_con_ = None
        self.trainer_dis_ = None
        self.sampler_con_ = None
        self.num_classes_ = None

        set_global_seed(random_state)

    def train(self, dataset_dir: str, synthetic_dir: str, **kwargs) -> "CODI":
        """
        Train the CoDi model.

        Args:
            dataset_dir: Directory containing x_train.csv and y_train.csv
            synthetic_dir: Directory to save synthetic data
            **kwargs: Additional arguments (passed from pipeline)

        Returns:
            self: Trained model instance
        """
        logger.info("=" * 80)
        logger.info("Training CoDi Model")
        logger.info("=" * 80)

        # Load training data
        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"Training data not found: {x_train_path}")

        X_train = pd.read_csv(x_train_path)

        # y_train is optional
        y_train = None
        if os.path.exists(y_train_path):
            y_train = pd.read_csv(y_train_path)
            if isinstance(y_train, pd.DataFrame) and len(y_train.columns) == 1:
                y_train = y_train.iloc[:, 0]

        # Combine X and y for full data processing
        if y_train is not None:
            y_name = y_train.name if hasattr(y_train, 'name') else 'target'
            df_train = X_train.copy()
            df_train[y_name] = y_train
        else:
            df_train = X_train.copy()

        logger.info(f"Loaded training data: {df_train.shape}")

        # Infer schema and encode data
        self.schema_ = infer_schema(df_train)
        encoded_data, self.schema_ = encode_dataframe(df_train, self.schema_)

        logger.info(f"Schema: {len(self.schema_['continuous_columns'])} continuous, "
                    f"{len(self.schema_['categorical_columns'])} categorical columns")

        # Split into continuous and categorical
        con_cols = self.schema_['continuous_columns']
        cat_cols = self.schema_['categorical_columns']

        # Handle continuous-only, categorical-only, or mixed data
        if len(con_cols) > 0:
            data_con = encoded_data[con_cols].values.astype(np.float32)
        else:
            # If no continuous columns, create a dummy one
            data_con = np.zeros((len(encoded_data), 1), dtype=np.float32)

        if len(cat_cols) > 0:
            data_cat = encoded_data[cat_cols].values.astype(np.float32)
            self.num_classes_ = np.array([self.schema_['column_info'][col]['size']
                                          for col in cat_cols])
        else:
            # If no categorical columns, create a dummy one
            data_cat = np.zeros((len(data_con), 1), dtype=np.float32)
            self.num_classes_ = np.array([1])

        self.has_continuous_ = len(con_cols) > 0
        self.has_categorical_ = len(cat_cols) > 0

        # Calculate one-hot encoded dimension for categorical data
        if self.has_categorical_:
            onehot_dim_cat = int(np.sum(self.num_classes_))
        else:
            onehot_dim_cat = 1

        # Build models (use one-hot dimension for categorical)
        self._build_models(data_con.shape[1], onehot_dim_cat)

        # Train the model
        self._fit(data_con, data_cat)

        # Generate synthetic data
        logger.info(f"\nGenerating {len(df_train)} synthetic samples...")
        # Keep data encoded for evaluation
        df_synth = self.sample(len(df_train), return_encoded=True)

        # Save synthetic data
        os.makedirs(synthetic_dir, exist_ok=True)

        # Ensure categorical columns are integers and within valid range
        cat_cols = self.schema_['categorical_columns']
        for col in cat_cols:
            if col in df_synth.columns:
                col_info = self.schema_['column_info'][col]
                max_val = col_info['size'] - 1
                # Clip, round, and convert to int
                df_synth[col] = df_synth[col].clip(
                    0, max_val).round().astype(int)

                # Ensure all classes appear at least once (important for classifiers)
                existing_classes = set(df_synth[col].unique())
                all_classes = set(range(col_info['size']))
                missing_classes = all_classes - existing_classes

                if missing_classes:
                    logger.warning(
                        f"Column '{col}': Adding missing classes {missing_classes}")
                    # Add one sample for each missing class at the end
                    for missing_class in sorted(missing_classes):
                        # Create a row with this class (copy last row and change this column)
                        new_row = df_synth.iloc[-1:].copy()
                        new_row[col] = missing_class
                        df_synth = pd.concat(
                            [df_synth, new_row], ignore_index=True)

        if y_train is not None:
            y_name = y_train.name if hasattr(y_train, 'name') else 'target'
            x_synth = df_synth.drop(columns=[y_name])
            y_synth = df_synth[[y_name]]

            x_synth.to_csv(os.path.join(
                synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(
                synthetic_dir, "y_synth.csv"), index=False)
        else:
            df_synth.to_csv(os.path.join(
                synthetic_dir, "x_synth.csv"), index=False)

        # Save metadata
        metadata = {
            'schema': self.schema_,
            'hyperparameters': {
                'n_steps': self.n_steps,
                'beta_1': self.beta_1,
                'beta_T': self.beta_T,
                'encoder_dim_con': self.encoder_dim_con,
                'encoder_dim_dis': self.encoder_dim_dis,
                'nf_con': self.nf_con,
                'nf_dis': self.nf_dis,
                'activation': self.activation,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'lr_con': self.lr_con,
                'lr_dis': self.lr_dis,
                'random_state': self.random_state
            }
        }

        # Save schema for data encoding/decoding
        save_metadata(self.schema_, os.path.join(synthetic_dir, 'schema.json'))

        # Save full metadata (schema + hyperparameters)
        import json
        with open(os.path.join(synthetic_dir, 'metadata.json'), 'w') as f:
            # Create serializable version
            metadata_serializable = {
                'hyperparameters': metadata['hyperparameters'],
                'schema': {
                    'continuous_columns': self.schema_['continuous_columns'],
                    'categorical_columns': self.schema_['categorical_columns'],
                    'column_order': self.schema_['column_order']
                }
            }
            json.dump(metadata_serializable, f, indent=2)

        logger.info(f"\nSynthetic data saved to: {synthetic_dir}")
        logger.info("Training complete!")

        return self

    def _build_models(self, dim_con: int, dim_cat: int):
        """Build the continuous and categorical diffusion models."""
        logger.info(f"Building models: con_dim={dim_con}, cat_dim={dim_cat}")

        # Continuous model
        self.model_con_ = TabularUNet(
            input_dim=dim_con,
            cond_dim=dim_cat,
            output_dim=dim_con,
            hidden_dims=list(self.encoder_dim_con),
            time_embed_dim=self.nf_con,
            activation=self.activation
        ).to(self.device)

        self.trainer_con_ = GaussianDiffusionTrainer(
            self.model_con_, self.beta_1, self.beta_T, self.n_steps
        ).to(self.device)

        self.sampler_con_ = GaussianDiffusionSampler(
            self.model_con_, self.beta_1, self.beta_T, self.n_steps
        ).to(self.device)

        # Discrete model
        self.model_dis_ = TabularUNet(
            input_dim=dim_cat,
            cond_dim=dim_con,
            output_dim=dim_cat,
            hidden_dims=list(self.encoder_dim_dis),
            time_embed_dim=self.nf_dis,
            activation=self.activation
        ).to(self.device)

        self.trainer_dis_ = MultinomialDiffusion(
            self.model_dis_, self.num_classes_, self.beta_1, self.beta_T, self.n_steps
        ).to(self.device)

        num_params_con = sum(p.numel() for p in self.model_con_.parameters())
        num_params_dis = sum(p.numel() for p in self.model_dis_.parameters())
        logger.info(f"Continuous model params: {num_params_con:,}")
        logger.info(f"Discrete model params: {num_params_dis:,}")

    def _fit(self, data_con: np.ndarray, data_cat: np.ndarray):
        """Fit the model using co-evolving training with contrastive learning."""
        logger.info(f"\nTraining for {self.epochs} epochs...")

        # Optimizers
        optim_con = torch.optim.Adam(
            self.model_con_.parameters(), lr=self.lr_con)
        optim_dis = torch.optim.Adam(
            self.model_dis_.parameters(), lr=self.lr_dis)

        # Convert to tensors
        data_con_tensor = torch.from_numpy(data_con).float()
        data_cat_tensor = torch.from_numpy(data_cat).float()

        # Training loop
        n_samples = len(data_con)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for epoch in range(self.epochs):
            self.model_con_.train()
            self.model_dis_.train()

            # Shuffle data
            indices = torch.randperm(n_samples)
            data_con_shuffled = data_con_tensor[indices]
            data_cat_shuffled = data_cat_tensor[indices]

            epoch_loss_con = 0.0
            epoch_loss_dis = 0.0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)

                batch_con = data_con_shuffled[start_idx:end_idx].to(
                    self.device)
                batch_cat = data_cat_shuffled[start_idx:end_idx].to(
                    self.device)

                # Train step
                loss_con, loss_dis = self._train_step(
                    batch_con, batch_cat, optim_con, optim_dis
                )

                epoch_loss_con += loss_con
                epoch_loss_dis += loss_dis

            avg_loss_con = epoch_loss_con / n_batches
            avg_loss_dis = epoch_loss_dis / n_batches

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                            f"loss_con={avg_loss_con:.4f}, loss_dis={avg_loss_dis:.4f}")

    def _train_step(
        self,
        x_con: torch.Tensor,
        x_cat: torch.Tensor,
        optim_con: torch.optim.Optimizer,
        optim_dis: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Single training step with contrastive learning."""
        batch_size = x_con.shape[0]

        # Sample timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)

        loss_con_total = torch.tensor(0.0, device=self.device)
        loss_dis_total = torch.tensor(0.0, device=self.device)

        # === Continuous diffusion loss (if we have continuous features) ===
        if self.has_continuous_:
            noise_con = torch.randn_like(x_con)
            x_t_con = self.trainer_con_.make_x_t(x_con, t, noise_con)
            eps_pred = self.model_con_(x_t_con, t, x_cat)
            loss_con_diff = F.mse_loss(eps_pred, noise_con)

            # Contrastive learning (simplified)
            neg_indices = torch.randperm(batch_size)
            x_cat_neg = x_cat[neg_indices]

            eps_pos = self.model_con_(x_t_con, t, x_cat)
            eps_neg = self.model_con_(x_t_con, t, x_cat_neg)

            loss_con_contrast = torch.relu(
                F.mse_loss(eps_neg, noise_con, reduction='none').mean() -
                F.mse_loss(eps_pos, noise_con, reduction='none').mean() + 1.0
            )

            loss_con_total = loss_con_diff + self.lambda_con * loss_con_contrast

            # Backward pass - continuous
            optim_con.zero_grad()
            loss_con_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_con_.parameters(), self.grad_clip)
            optim_con.step()

        # === Discrete diffusion loss (if we have categorical features) ===
        if self.has_categorical_:
            # Convert to log-space one-hot
            x_cat_onehot = self._to_onehot(x_cat)
            log_x_start = torch.log(x_cat_onehot.float().clamp(min=1e-30))
            x_t_cat = self.trainer_dis_.q_sample(log_x_start, t)
            kl, _ = self.trainer_dis_.compute_Lt(
                log_x_start, x_t_cat, t, x_con)
            kl_prior = self.trainer_dis_.kl_prior(log_x_start)
            loss_dis_total = (kl + kl_prior).mean()

            # Backward pass - discrete
            optim_dis.zero_grad()
            loss_dis_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_dis_.parameters(), self.grad_clip)
            optim_dis.step()

        return loss_con_total.item(), loss_dis_total.item()

    def _to_onehot(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Convert categorical data to one-hot encoding."""
        if x_cat.shape[1] == 1 and self.num_classes_[0] == 1:
            # Dummy categorical
            return torch.ones_like(x_cat)

        onehot_list = []
        for i, n_classes in enumerate(self.num_classes_):
            col = x_cat[:, i].long()
            onehot = torch.zeros(len(col), n_classes, device=x_cat.device)
            onehot.scatter_(1, col.unsqueeze(1), 1.0)
            onehot_list.append(onehot)

        return torch.cat(onehot_list, dim=1)

    def sample(
        self,
        n: int,
        conditional: Optional[Dict[str, Any]] = None,
        return_encoded: bool = False
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Args:
            n: Number of samples to generate
            conditional: Optional conditional information (not yet implemented)
            return_encoded: If True, return encoded data with consecutive integers

        Returns:
            DataFrame of synthetic samples (encoded or decoded based on return_encoded)
        """
        if self.schema_ is None:
            raise ValueError("Model must be trained before sampling")

        if self.has_continuous_:
            self.model_con_.eval()
        if self.has_categorical_:
            self.model_dis_.eval()

        with torch.no_grad():
            # Start from random noise
            if self.has_continuous_:
                x_T_con = torch.randn(
                    n, self.model_con_.input_dim, device=self.device)
            else:
                # Dummy continuous data for categorical-only
                x_T_con = torch.zeros(n, 1, device=self.device)

            # For categorical, start from uniform distribution
            if self.has_categorical_:
                x_T_cat_shape = (n, self.model_dis_.input_dim)
                log_x_T_cat = torch.log(
                    torch.ones(x_T_cat_shape, device=self.device) /
                    self.num_classes_[0]
                )
            else:
                # Dummy categorical data for continuous-only
                log_x_T_cat = torch.zeros(n, 1, device=self.device)

            # Reverse diffusion process
            x_0_con, x_0_cat = self._sample_reverse(x_T_con, log_x_T_cat)

        # Convert back to numpy
        samples_con = x_0_con.cpu().numpy()
        samples_cat = x_0_cat.cpu().numpy()

        # Combine continuous and categorical
        con_cols = self.schema_['continuous_columns']
        cat_cols = self.schema_['categorical_columns']

        # Create DataFrame with encoded values
        df_encoded = pd.DataFrame()

        if self.has_continuous_:
            for i, col in enumerate(con_cols):
                df_encoded[col] = samples_con[:, i]

        if self.has_categorical_:
            for i, col in enumerate(cat_cols):
                df_encoded[col] = samples_cat[:, i]

        # Return encoded or decoded data based on flag
        if return_encoded:
            return df_encoded
        else:
            # Decode to original space
            df_decoded = decode_dataframe(df_encoded, self.schema_)
            return df_decoded

    def _sample_reverse(
        self,
        x_T_con: torch.Tensor,
        log_x_T_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverse diffusion sampling process."""
        x_t_con = x_T_con
        x_t_cat = log_x_T_cat

        for t_step in reversed(range(self.n_steps)):
            t = torch.full((x_t_con.shape[0],), t_step,
                           device=self.device, dtype=torch.long)

            # Sample continuous (if we have continuous features)
            if self.has_continuous_:
                mean, log_var = self.sampler_con_.p_mean_variance(
                    x_t_con, t, x_t_cat)
                if t_step > 0:
                    noise = torch.randn_like(x_t_con)
                else:
                    noise = 0
                x_t_con = mean + torch.exp(0.5 * log_var) * noise
                x_t_con = torch.clamp(x_t_con, -1.0, 1.0)

            # Sample categorical (if we have categorical features)
            if self.has_categorical_:
                x_t_cat = self.trainer_dis_.p_sample(x_t_cat, t, x_t_con)

        # Convert categorical from log-space to indices
        if self.has_categorical_:
            cat_indices = self._from_onehot_log(x_t_cat)
        else:
            cat_indices = torch.zeros(len(x_t_con), 1, device=x_t_con.device)

        return x_t_con, cat_indices

    def _from_onehot_log(self, log_x: torch.Tensor) -> torch.Tensor:
        """Convert log one-hot back to categorical indices."""
        if self.num_classes_[0] == 1:
            return torch.zeros(len(log_x), 1, device=log_x.device)

        indices_list = []
        start_idx = 0
        for n_classes in self.num_classes_:
            end_idx = start_idx + n_classes
            col_logits = log_x[:, start_idx:end_idx]
            col_indices = torch.argmax(col_logits, dim=1, keepdim=True)
            indices_list.append(col_indices.float())
            start_idx = end_idx

        return torch.cat(indices_list, dim=1)

    def evaluate(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        model: str = "lr",
        metrics: Optional[list] = None,
        task: Optional[str] = None,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, float]:
        """
        Quick model-centric TSTR evaluation.

        Args:
            x: Feature DataFrame
            y: Target Series
            model: Model type ('lr', 'mlp', 'rf', 'xgb')
            metrics: List of metrics to compute
            task: Task type ('classification' or 'regression')
            random_state: Random seed
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric scores
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        # Infer task if not provided
        if task is None:
            unique_values = y.nunique()
            task = 'classification' if unique_values < 20 else 'regression'

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select model
        if task == 'classification':
            if model == "lr":
                clf = LogisticRegression(
                    max_iter=1000, random_state=random_state)
            elif model == "mlp":
                clf = MLPClassifier(hidden_layer_sizes=(
                    100,), max_iter=1000, random_state=random_state)
            elif model == "rf":
                clf = RandomForestClassifier(
                    n_estimators=100, random_state=random_state)
            else:
                raise ValueError(f"Unknown model: {model}")

            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            if model == "lr":
                reg = LinearRegression()
            elif model == "mlp":
                reg = MLPRegressor(hidden_layer_sizes=(
                    100,), max_iter=1000, random_state=random_state)
            elif model == "rf":
                reg = RandomForestRegressor(
                    n_estimators=100, random_state=random_state)
            else:
                raise ValueError(f"Unknown model: {model}")

            reg.fit(X_train_scaled, y_train)
            y_pred = reg.predict(X_test_scaled)

            results = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2))
            }

        return results
