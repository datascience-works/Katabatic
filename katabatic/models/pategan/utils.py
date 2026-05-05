import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import LabelEncoder


def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy and tensorflow."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer schema from a dataframe including column types and metadata.

    Args:
        df: Input dataframe

    Returns:
        Dict containing schema information
    """
    schema = {
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'column_info': {}
    }

    for col in df.columns:
        col_info = {'type': None, 'metadata': {}}

        if df[col].dtype in ['object', 'category']:
            col_info['type'] = 'categorical'
            col_info['metadata'] = {
                'categories': df[col].astype(str).unique().tolist()
            }
        elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
            col_info['type'] = 'discrete'
            col_info['metadata'] = {
                'min': int(df[col].min()),
                'max': int(df[col].max())
            }
        else:  # float types
            col_info['type'] = 'continuous'
            col_info['metadata'] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }

        schema['column_info'][col] = col_info

    return schema


class DataTransformer:
    """
    Handles encoding/decoding of tabular data for PATE-GAN.

    Maintains original column order and types for schema fidelity.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.schema: Optional[Dict] = None
        self.column_order: List[str] = []
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> 'DataTransformer':
        """
        Fit transformers on the data.

        Args:
            df: Input dataframe to fit on

        Returns:
            self
        """
        self.schema = infer_schema(df)
        self.column_order = df.columns.tolist()

        df_transformed = df.copy()

        # Encode categorical columns
        for col in self.column_order:
            col_info = self.schema['column_info'][col]

            if col_info['type'] == 'categorical':
                le = LabelEncoder()
                df_transformed[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Convert to numpy and normalize to [0, 1]
        data_array = df_transformed.values.astype(float)
        self.min_vals = np.min(data_array, axis=0)
        self.max_vals = np.max(data_array, axis=0)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dataframe to normalized numpy array.

        Args:
            df: Input dataframe

        Returns:
            Normalized numpy array in [0, 1]
        """
        df_transformed = df.copy()

        # Encode categorical columns
        for col in self.column_order:
            if col in self.label_encoders:
                df_transformed[col] = self.label_encoders[col].transform(
                    df[col].astype(str))

        # Convert to numpy and normalize
        data_array = df_transformed[self.column_order].values.astype(float)
        normalized = (data_array - self.min_vals) / \
            (self.max_vals - self.min_vals + 1e-8)

        return normalized

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """
        Convert normalized array back to original feature space.

        Args:
            data: Normalized numpy array

        Returns:
            DataFrame in original feature space with correct dtypes
        """
        # Denormalize
        denormalized = data * \
            (self.max_vals - self.min_vals + 1e-8) + self.min_vals

        # Create dataframe
        df = pd.DataFrame(denormalized, columns=self.column_order)

        # Decode categorical columns
        for col in self.column_order:
            col_info = self.schema['column_info'][col]

            if col in self.label_encoders:
                # Round to nearest integer and clip to valid range
                int_vals = np.round(df[col]).astype(int)
                int_vals = np.clip(int_vals, 0, len(
                    self.label_encoders[col].classes_) - 1)
                df[col] = self.label_encoders[col].inverse_transform(int_vals)
            elif col_info['type'] == 'discrete':
                # Round continuous values for discrete columns
                df[col] = np.round(df[col]).astype(int)

        return df


class PrivacyMechanism:
    """
    Implements PATE privacy mechanism with noisy teacher aggregation.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, num_teachers: int = 10):
        """
        Initialize privacy mechanism.

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter (typically 1e-5 or 1e-6)
            num_teachers: Number of teacher discriminators
        """
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers

        # Calculate noise scale (lambda) for Gaussian mechanism
        # Note: Original paper has a bug in lambda calculation
        # Correct formula: lambda = sqrt(2 * log(1.25 / delta)) / epsilon
        self.lambda_noise = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def add_gaussian_noise(self, votes: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to teacher votes for differential privacy.

        Args:
            votes: Binary votes from teachers (shape: [batch_size])

        Returns:
            Noisy aggregated votes (shape: [batch_size])
        """
        noise = np.random.normal(
            loc=0.0, scale=self.lambda_noise, size=votes.shape)
        noisy_votes = votes + noise
        return noisy_votes





