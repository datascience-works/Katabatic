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
from utils import * 

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
        csv_path: csv path
        test_split: split for testing (default: 0.2)
        total_epochs_both: training epochs (default: 20)
        training_batch_size: batch size (default: 1024)
        num_samples: samples generated (default: 500)
        continuous_columns: continuous columns (optional)
        categorical_columns: categorical columns (optional)
        logdir: experiment logs (default: './CoDi_exp')
        verbose: print detailed processing information (default: False)
        force_new_training: start fresh training (recommended for new datasets)
        cleanup_temp_files: clean up temporary files after generation (default: True)
    
    Returns:
        pd.DataFrame: Generated synthetic data 
    """
    
    timestamp = int(time.time())
    dataset_name = f"temp_dataset_{timestamp}"
    
    if logdir is None:
        logdir = f'./CoDi_exp_{timestamp}'
    
    try:
        if verbose:
            print(f"Processing dataset: {csv_path}")
            print(f"Using logdir: {logdir}")
        
        # Process the dataset
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
            print(f"Dataset processed: {result['shape']} -> {result['problem_type']}")
        
        # Ensure clean logdir
        os.makedirs(logdir, exist_ok=True)
        
        # Run CoDi training and generation
        if verbose:
            print(f"Running CoDi training...")
        
        # Prepare command
        cmd = [
            sys.executable, 'utils.py',
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
            print(f"CoDi training completed")
        
        # Load and process synthetic data
        if verbose:
            print(f"Loading synthetic data...")
        
        # Load synthetic data
        synthetic_pkl_path = os.path.join(logdir, 'synthetic_data.pkl')
        if not os.path.exists(synthetic_pkl_path):
            raise FileNotFoundError(f"Synthetic data not found at {synthetic_pkl_path}")
        
        with open(synthetic_pkl_path, 'rb') as f:
            synthetic_datasets = pickle.load(f)
        
        # Load metadata
        metadata = result['metadata']
        categorical_mappings = result['categorical_mappings']
             
        # Create DataFrame
        column_names = [col['name'] for col in metadata['columns']]
        combined_raw_data = np.vstack(synthetic_datasets)
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
            print(f"Synthetic data generated: {synthetic_df.shape}")
            print(f"Data types preserved and mapped back to original format")
        
        return synthetic_df
        
    except Exception as e:
        print(f"Error in codi(): {str(e)}")
        raise
    
    finally:
        # Cleanup temporary files
        if cleanup_temp_files:
            try:
                # Remove dataset files
                if os.path.exists(f'tabular_datasets/{dataset_name}.npz'):
                    os.remove(f'tabular_datasets/{dataset_name}.npz')
                if os.path.exists(f'tabular_datasets/{dataset_name}.json'):
                    os.remove(f'tabular_datasets/{dataset_name}.json')
                
                # Remove the entire tabular_datasets folder if it's empty or only contains temp files
                if os.path.exists('tabular_datasets'):
                    # Check if folder is empty or only contains temporary files
                    remaining_files = os.listdir('tabular_datasets')
                    temp_files = [f for f in remaining_files if f.startswith('temp_dataset_')]
                    
                    if len(remaining_files) == len(temp_files):
                        for temp_file in temp_files:
                            temp_path = os.path.join('tabular_datasets', temp_file)
                            if os.path.isfile(temp_path):
                                os.remove(temp_path)
                        
                        if not os.listdir('tabular_datasets'):
                            os.rmdir('tabular_datasets')
                            if verbose:
                                print(f"Cleaned up tabular_datasets folder")
                
                # Remove the entire logdir
                if logdir and logdir.startswith('./CoDi_exp') and os.path.exists(logdir):
                    shutil.rmtree(logdir)
                    if verbose:
                        print(f"Cleaned up logdir: {logdir}")
                
                if verbose:
                    print(f"Cleaned up temporary files")
            except Exception as cleanup_error:
                if verbose:
                    print(f"Warning: Could not clean up some files: {cleanup_error}")
                pass 
