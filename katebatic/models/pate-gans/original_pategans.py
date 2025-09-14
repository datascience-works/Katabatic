import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import sys
import importlib.util
from utils import *

class PateGans:
    def __init__(self, epsilon=1.0, delta=1e-5, num_teachers=10):
        """
        Simple PATE-GAN generator using the original implementation
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter 
            num_teachers: Number of teacher models. Fewer teachers for small dataset
        """
        self.epsilon = epsilon
        self.delta = delta
        self.num_teachers = num_teachers
        self.model = None
        self.label_encoders = {}  # Store encoders for later decoding
        
        # Set random seed
        seed = 13
        np.random.seed(seed)
    
    def fit_generate(self, data, n_samples=None, target_column=None):
        """
        Train PATE-GAN and generate synthetic data
        
        Args:
            data: pandas DataFrame or path to CSV file
            n_samples: Number of samples to generate
            target_column: name of target column
            
        Returns:
            pandas DataFrame with synthetic data
        """
        # Load data if path provided
        if isinstance(data, str):
            if data.endswith('.gz'):
                df = pd.read_csv(data, compression='gzip')
            else:
                df = pd.read_csv(data)
        else:
            df = data.copy()
        
        print(f"Training on dataset with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Store original column names and dtypes for later reconstruction
        self.original_columns = df.columns.tolist()
        self.original_dtypes = df.dtypes.to_dict()
        
        # Convert all non-numeric columns to numeric using label encoding
        df_numeric = df.copy()
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                print(f"Encoding categorical column: {column}")
                le = LabelEncoder()
                df_numeric[column] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le
        
        # Convert to numpy array (all numeric now)
        df_numeric = df_numeric.astype(float)
        
        print(f"Preprocessed data shape: {df_numeric.shape}")
        print(f"Data types after preprocessing: {df_numeric.dtypes.tolist()}")
        
        # Get data shape for X_shape parameter
        n_records, n_features = df_numeric.shape
        
        # Set up model parameters  
        kwargs = {
            "epsilon": self.epsilon,
            "delta": int(-np.log10(self.delta)),  # Original wants -log10(delta)
            "num_teachers": self.num_teachers,
            "X_shape": (n_records, n_features)  # Add the required X_shape parameter
        }
        
        # Initialize model
        print("Initializing PATE-GAN...")
        self.model = PG_ORIGINAL(**kwargs)
        
        # Train the model
        print("Training PATE-GAN...")
        self.model.fit(df_numeric) 
        print("Training completed!")
        
        # Generate synthetic data
        if n_samples is None:
            n_samples = len(df)
        
        print(f"Generating {n_samples} synthetic samples...")
        synthetic_data = self.model.generate(n_samples)
        print("Generation completed!")
        
        # Convert back to DataFrame with original column names
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.original_columns)
        
        # Decode categorical columns back to original values
        for column, encoder in self.label_encoders.items():
            # Round to nearest integer for categorical data
            synthetic_df[column] = np.round(synthetic_df[column]).astype(int)
            # Clip values to valid range
            min_val, max_val = 0, len(encoder.classes_) - 1
            synthetic_df[column] = np.clip(synthetic_df[column], min_val, max_val)
            # Decode back to original categories
            synthetic_df[column] = encoder.inverse_transform(synthetic_df[column])
        
        # Ensure numeric columns have appropriate types
        for column in self.original_columns:
            if column not in self.label_encoders:
                if self.original_dtypes[column] in ['int64', 'int32']:
                    synthetic_df[column] = np.round(synthetic_df[column]).astype(int)
        
        return synthetic_df

