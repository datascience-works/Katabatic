import pandas as pd
import numpy as np
import json
import os
import shutil
import subprocess
import sys
import pickle
from typing import Tuple, List, Dict, Any, Optional, Union
import tempfile
import warnings
import glob
import time

class DatasetProcessor:
    """Complete dataset processor for CoDi with automatic fixes and validation"""
    
    def __init__(self, categorical_threshold: int = 20, numeric_categorical_threshold: float = 0.05):
        self.categorical_threshold = categorical_threshold
        self.numeric_categorical_threshold = numeric_categorical_threshold
    
    def auto_detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically detect continuous and categorical columns with improved logic"""
        continuous_cols = []
        categorical_cols = []
        
        for col in df.columns:
            col_data = df[col].dropna()  # Remove NaN for analysis
            unique_count = col_data.nunique()
            total_count = len(col_data)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Check if column contains only integers (potential categorical)
            is_integer_like = False
            if pd.api.types.is_numeric_dtype(col_data):
                is_integer_like = col_data.apply(lambda x: float(x).is_integer()).all()
            
            # Enhanced decision logic
            if pd.api.types.is_numeric_dtype(col_data):
                # Special case: floating point values that are actually discrete
                if is_integer_like and (unique_count <= self.categorical_threshold or unique_ratio < self.numeric_categorical_threshold):
                    categorical_cols.append(col)
                # Special case: many decimal values suggest continuous
                elif not is_integer_like and unique_count > self.categorical_threshold:
                    continuous_cols.append(col)
                # Default numeric logic
                elif unique_count <= self.categorical_threshold or unique_ratio < self.numeric_categorical_threshold:
                    categorical_cols.append(col)
                else:
                    continuous_cols.append(col)
            else:
                # Non-numeric -> categorical
                categorical_cols.append(col)
        
        return continuous_cols, categorical_cols
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced preprocessing with better missing value handling"""
        df_processed = df.copy()
        categorical_mappings = {}
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col].fillna(mode_val[0], inplace=True)
                    else:
                        df_processed[col].fillna('unknown', inplace=True)
        
        # Encode categorical variables with proper indexing
        for col in df_processed.columns:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                unique_vals = sorted(df_processed[col].unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                categorical_mappings[col] = {
                    'mapping': mapping,
                    'reverse_mapping': {idx: val for val, idx in mapping.items()}
                }
                df_processed[col] = df_processed[col].map(mapping)
        
        return df_processed, categorical_mappings
    
    def validate_and_fix_categorical_data(self, data: np.ndarray, columns: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Validate and fix categorical columns to ensure proper 0-based indexing"""
        fixed_data = data.copy()
        fixed_columns = [col.copy() for col in columns]
        
        for i, col in enumerate(fixed_columns):
            if col['type'] == 'categorical':
                col_data = fixed_data[:, i].astype(int)
                unique_vals = sorted(np.unique(col_data))
                
                # Check if values are properly 0-based
                expected_range = list(range(len(unique_vals)))
                if unique_vals != expected_range:
                    # Create mapping to fix indexing
                    mapping = {old_val: new_val for new_val, old_val in enumerate(unique_vals)}
                    
                    # Apply mapping
                    for old_val, new_val in mapping.items():
                        fixed_data[fixed_data[:, i] == old_val, i] = new_val
                    
                    # Update column metadata
                    col['size'] = len(unique_vals)
                    col['i2s'] = [str(val) for val in unique_vals]
        
        return fixed_data, fixed_columns
    
    def create_codi_format(self, dataset_name: str, train_data: np.ndarray, test_data: np.ndarray, 
                          column_names: List[str], con_idx: List[int], dis_idx: List[int], 
                          categorical_mappings: Dict) -> Dict:
        """Create CoDi-compatible format with validation"""
        
        # Validate and fix categorical data
        all_data = np.vstack([train_data, test_data])
        
        # Create initial columns structure
        columns = []
        for i, col_name in enumerate(column_names):
            if i in con_idx:
                col_data = all_data[:, i]
                columns.append({
                    "name": col_name,
                    "type": "continuous",
                    "min": float(np.min(col_data)),
                    "max": float(np.max(col_data))
                })
            else:
                col_data = all_data[:, i].astype(int)
                unique_vals = sorted(np.unique(col_data))
                
                # Create i2s mapping
                if col_name in categorical_mappings:
                    reverse_mapping = categorical_mappings[col_name]['reverse_mapping']
                    i2s = [str(reverse_mapping.get(idx, str(idx))) for idx in unique_vals]
                else:
                    i2s = [str(val) for val in unique_vals]
                
                columns.append({
                    "name": col_name,
                    "type": "categorical",
                    "size": len(unique_vals),
                    "i2s": i2s
                })
        
        # Fix categorical data indexing
        fixed_train, fixed_columns = self.validate_and_fix_categorical_data(train_data, columns)
        fixed_test, _ = self.validate_and_fix_categorical_data(test_data, columns)
        
        # Determine problem type
        last_col_idx = len(column_names) - 1
        if last_col_idx in dis_idx:
            last_col_name = column_names[last_col_idx]
            if last_col_name in categorical_mappings:
                num_classes = len(categorical_mappings[last_col_name]['mapping'])
            else:
                num_classes = len(np.unique(all_data[:, last_col_idx]))
            
            problem_type = "binary_classification" if num_classes == 2 else "multiclass_classification"
        else:
            problem_type = "regression"
        
        # Save fixed data
        os.makedirs('tabular_datasets', exist_ok=True)
        np.savez(f'tabular_datasets/{dataset_name}.npz', train=fixed_train, test=fixed_test)
        
        # Create metadata
        codi_meta = {
            "columns": fixed_columns,
            "problem_type": problem_type
        }
        
        with open(f'tabular_datasets/{dataset_name}.json', 'w') as f:
            json.dump(codi_meta, f, indent=2)
        
        return codi_meta
    
    def process_dataset(self, csv_path: str, dataset_name: str, 
                       force_continuous: Optional[List[str]] = None,
                       force_categorical: Optional[List[str]] = None,
                       test_split: float = 0.2,
                       verbose: bool = False) -> Dict[str, Any]:
        """Complete dataset processing pipeline"""
        
        force_continuous = force_continuous or []
        force_categorical = force_categorical or []
        
        # Load and analyze data
        df = pd.read_csv(csv_path)
        
        # Auto-detect column types
        continuous_cols, categorical_cols = self.auto_detect_column_types(df)
        
        # Apply manual overrides
        for col in force_continuous:
            if col in categorical_cols:
                categorical_cols.remove(col)
            if col not in continuous_cols:
                continuous_cols.append(col)
        
        for col in force_categorical:
            if col in continuous_cols:
                continuous_cols.remove(col)
            if col not in categorical_cols:
                categorical_cols.append(col)
        
        if verbose:
            print(f"Continuous columns: {continuous_cols}")
            print(f"Categorical columns: {categorical_cols}")
        
        # Preprocess data
        df_processed, categorical_mappings = self.preprocess_data(df)
        
        # Get indices
        con_idx = [df_processed.columns.get_loc(col) for col in continuous_cols]
        dis_idx = [df_processed.columns.get_loc(col) for col in categorical_cols]
        
        # Split data
        data = df_processed.values.astype(np.float32)
        n_samples, n_features = data.shape
        n_test = int(n_samples * test_split)
        
        # Shuffle and split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(n_samples)
        test_data = data[indices[:n_test]]
        train_data = data[indices[n_test:]]
        
        # Create CoDi format with validation and fixes
        codi_meta = self.create_codi_format(
            dataset_name, train_data, test_data, 
            df_processed.columns.tolist(), con_idx, dis_idx, categorical_mappings
        )
        
        return {
            'dataset_name': dataset_name,
            'shape': (n_samples, n_features),
            'problem_type': codi_meta['problem_type'],
            'continuous_columns': continuous_cols,
            'categorical_columns': categorical_cols,
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'metadata': codi_meta,
            'categorical_mappings': categorical_mappings
        }


def codi(csv_path: str,
         test_split: float = 0.2,
         total_epochs_both: int = 20,
         training_batch_size: int = 1024,
         num_samples: int = 500,
         continuous_columns: Optional[List[str]] = None,
         categorical_columns: Optional[List[str]] = None,
         logdir: str = './CoDi_exp',
         verbose: bool = False,
         force_new_training: bool = True,
         cleanup_temp_files: bool = True) -> pd.DataFrame:
    """
    Plug-and-play synthetic data generation using CoDi.
    
    Args:
        csv_path: Path to input CSV file
        test_split: Fraction of data to use for testing (default: 0.2)
        total_epochs_both: Number of training epochs (default: 20)
        training_batch_size: Batch size for training (default: 1024)
        num_samples: Number of synthetic samples to generate (default: 500)
        continuous_columns: List of columns to force as continuous (optional)
        categorical_columns: List of columns to force as categorical (optional)
        logdir: Directory for CoDi experiment logs (default: './CoDi_exp')
        verbose: Whether to print detailed processing information (default: False)
        force_new_training: Whether to start fresh training (recommended for new datasets)
        cleanup_temp_files: Whether to clean up temporary files after generation (default: True)
    
    Returns:
        pd.DataFrame: Generated synthetic data with original column names and types
    
    Example:
        >>> synthetic_data = codi(
        ...     csv_path='raw_data/iris.csv',
        ...     test_split=0.2,
        ...     total_epochs_both=20,
        ...     training_batch_size=1024,
        ...     num_samples=500,
        ... )
    """
    
    # Generate unique dataset name and logdir to avoid conflicts
    timestamp = int(time.time())
    dataset_name = f"temp_dataset_{timestamp}"
    
    # Auto-generate unique logdir if not provided
    if logdir is None:
        logdir = f'./CoDi_exp_{timestamp}'
    
    try:
        if verbose:
            print(f"🔄 Processing dataset: {csv_path}")
            print(f"📁 Using logdir: {logdir}")
        
        # Step 1: Process the dataset
        processor = DatasetProcessor()
        result = processor.process_dataset(
            csv_path=csv_path,
            dataset_name=dataset_name,
            force_continuous=continuous_columns,
            force_categorical=categorical_columns,
            test_split=test_split,
            verbose=verbose
        )
        
        if verbose:
            print(f"✅ Dataset processed: {result['shape']} -> {result['problem_type']}")
        
        # Step 2: Ensure clean logdir
        os.makedirs(logdir, exist_ok=True)
        
        # Remove any existing checkpoint in this specific logdir
        # checkpoint_path = os.path.join(logdir, 'ckpt.pt')
        # if os.path.exists(checkpoint_path):
        #     if verbose:
        #         print(f"🗑️  Removing existing checkpoint: {checkpoint_path}")
        #     os.remove(checkpoint_path)
        
        # Step 3: Run CoDi training and generation
        if verbose:
            print(f"🚀 Running CoDi training...")
        
        # Prepare command
        cmd = [
            sys.executable, 'main.py',
            '--data', dataset_name,
            '--total_epochs_both', str(total_epochs_both),
            '--training_batch_size', str(training_batch_size),
            '--num_samples', str(num_samples),
            '--logdir', logdir,
            '--train'
        ]
        
        # Run CoDi
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if verbose:
                result_cmd = subprocess.run(cmd, text=True)
            else:
                result_cmd = subprocess.run(cmd, capture_output=True, text=True)
        
        if result_cmd.returncode != 0:
            raise RuntimeError(f"CoDi training failed: {result_cmd.stderr}")
        
        if verbose:
            print(f"✅ CoDi training completed")
        
        # Step 4: Load and process synthetic data
        if verbose:
            print(f"📊 Loading synthetic data...")
        
        # Load synthetic data
        synthetic_pkl_path = os.path.join(logdir, 'synthetic_data.pkl')
        if not os.path.exists(synthetic_pkl_path):
            raise FileNotFoundError(f"Synthetic data not found at {synthetic_pkl_path}")
        
        with open(synthetic_pkl_path, 'rb') as f:
            synthetic_datasets = pickle.load(f)
        
        # Load metadata
        metadata = result['metadata']
        categorical_mappings = result['categorical_mappings']
        
        # Get column names
        column_names = [col['name'] for col in metadata['columns']]
        
        # Combine all synthetic datasets
        combined_raw_data = np.vstack(synthetic_datasets)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(combined_raw_data, columns=column_names)
        
        # Map categorical values back to original strings
        for col_info in metadata['columns']:
            if col_info['type'] == 'categorical' and 'i2s' in col_info:
                col_name = col_info['name']
                i2s = col_info['i2s']
                synthetic_df[col_name] = synthetic_df[col_name].round().astype(int).apply(
                    lambda x: i2s[x] if 0 <= x < len(i2s) else f"unknown_{x}"
                )
        
        if verbose:
            print(f"✅ Synthetic data generated: {synthetic_df.shape}")
            print(f"📈 Data types preserved and mapped back to original format")
        
        return synthetic_df
        
    except Exception as e:
        print(f"❌ Error in codi(): {str(e)}")
        raise
    
    finally:
        # Cleanup temporary files if requested
        if cleanup_temp_files:
            try:
                # Remove dataset files
                if os.path.exists(f'tabular_datasets/{dataset_name}.npz'):
                    os.remove(f'tabular_datasets/{dataset_name}.npz')
                if os.path.exists(f'tabular_datasets/{dataset_name}.json'):
                    os.remove(f'tabular_datasets/{dataset_name}.json')
                
                # Remove the entire logdir if it was auto-generated
                if logdir and logdir.startswith('./CoDi_exp_') and os.path.exists(logdir):
                    shutil.rmtree(logdir)
                    if verbose:
                        print(f"🧹 Cleaned up logdir: {logdir}")
                
                if verbose:
                    print(f"🧹 Cleaned up temporary files")
            except:
                pass  # Ignore cleanup errors
