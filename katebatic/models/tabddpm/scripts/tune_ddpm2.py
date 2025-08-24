import tomli
import tomli_w  # For writing TOML files
import shutil
import os
import argparse
from .train import train
from .sample import sample
from .eval_catboost import train_catboost
from .eval_mlp import train_mlp
from .eval_simple import train_simple
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)

def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir, 'config.toml')  # Fixed: added filename
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def create_tabddpm_dataset_from_csv(csv_path, dataset_name, target_column, test_size=0.2, random_state=42):
    """
    Convert CSV to TabDDPM format and create proper config
    
    Returns:
        tuple: (train_size, config_path, dataset_info)
    """
    print(f"🚀 Converting '{csv_path}' to TabDDPM format...")
    
    # Load and inspect CSV
    df = pd.read_csv(csv_path)
    print(f"   Dataset size: {len(df)}")
    print(f"   Features: {len(df.columns)-1}")
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Auto-detect categorical columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = [col for col in X.columns if col not in categorical_columns]
    
    print(f"   Numerical features: {len(numerical_columns)}")
    print(f"   Categorical features: {len(categorical_columns)}")
    
    # Process categorical features
    X_cat = None
    if categorical_columns:
        X_cat = X[categorical_columns].copy()
        for col in categorical_columns:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(X_cat[col].astype(str))
        X_cat = X_cat.values.astype(np.int32)
    
    # Process numerical features
    X_num = None
    if numerical_columns:
        X_num = X[numerical_columns].values.astype(np.float32)
    
    # Process target
    is_classification = y.dtype == 'object' or len(np.unique(y)) < 20
    if is_classification:
        le_target = LabelEncoder()
        y_processed = le_target.fit_transform(y)
        n_classes = len(le_target.classes_)
        task_type = 'binclass' if n_classes == 2 else 'multiclass'
    else:
        y_processed = y.values.astype(np.float32)
        task_type = 'regression'
        n_classes = None
    
    # Train-test-val split
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=y_processed if is_classification else None
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=random_state,
        stratify=y_processed[train_val_idx] if is_classification else None
    )
    
    train_size = len(train_idx)
    
    # Create data directory
    data_dir = Path(f'data/{dataset_name}')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data files
    if X_num is not None:
        np.save(data_dir / 'X_num_train.npy', X_num[train_idx])
        np.save(data_dir / 'X_num_val.npy', X_num[val_idx])
        np.save(data_dir / 'X_num_test.npy', X_num[test_idx])
    
    if X_cat is not None:
        np.save(data_dir / 'X_cat_train.npy', X_cat[train_idx])
        np.save(data_dir / 'X_cat_val.npy', X_cat[val_idx])
        np.save(data_dir / 'X_cat_test.npy', X_cat[test_idx])
    
    np.save(data_dir / 'y_train.npy', y_processed[train_idx])
    np.save(data_dir / 'y_val.npy', y_processed[val_idx])
    np.save(data_dir / 'y_test.npy', y_processed[test_idx])
    
    # Create info.json
    dataset_info = {
        'task_type': task_type,
        'n_num_features': len(numerical_columns) if numerical_columns else 0,
        'n_cat_features': len(categorical_columns) if categorical_columns else 0,
        'train_size': train_size,
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'n_classes': n_classes
    }
    
    with open(data_dir / 'info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"   ✅ Dataset created at: {data_dir}")
    return train_size, str(data_dir), dataset_info

def convert_none_to_placeholder(obj):
    """Convert None values to '__none__' placeholder for TOML serialization"""
    if isinstance(obj, dict):
        return {k: convert_none_to_placeholder(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_none_to_placeholder(item) for item in obj]
    elif obj is None:
        return '__none__'
    else:
        return obj

def convert_placeholder_to_none(obj):
    """Convert '__none__' placeholder back to None values"""
    if isinstance(obj, dict):
        return {k: convert_placeholder_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_placeholder_to_none(item) for item in obj]
    elif obj == '__none__':
        return None
    else:
        return obj

def create_simple_config(data_path, dataset_info, output_dir='./ddpm_output'):
    """
    Create a minimal working config based on the paper notebook structure
    """
    train_size = dataset_info['train_size']
    n_num_features = dataset_info['n_num_features']
    n_cat_features = dataset_info['n_cat_features']
    
    # Simple config that matches the working paper example
    config = {
        'seed': 0,
        'parent_dir': output_dir,
        'real_data_path': data_path,
        'model_type': 'mlp',
        'num_numerical_features': n_num_features,
        'device': 'cuda',
        
        'model_params': {
            'd_in': n_num_features + n_cat_features,
            'num_classes': 0,
            'is_y_cond': False,
            'rtdl_params': {
                'd_layers': [256, 256],
                'dropout': 0.0
            }
        },
        
        'diffusion_params': {
            'num_timesteps': 1000,
            'gaussian_loss_type': 'mse',
            'scheduler': 'cosine'
        },
        
        'train': {
            'main': {
                'lr': 0.002,
                'steps': 1000,  # Start small for testing
                'batch_size': min(1024, max(256, train_size // 4)),
                'weight_decay': 1e-5
            },
            'T': {
                'seed': 0,
                'normalization': 'standard',
                'num_nan_policy': '__none__',  # Use placeholder
                'cat_nan_policy': '__none__',
                'cat_min_frequency': '__none__',
                'cat_encoding': '__none__',
                'y_policy': 'default'
            }
        },
        
        'sample': {
            'num_samples': train_size,
            'batch_size': min(2000, train_size),
            'seed': 0
        },
        
        'eval': {
            'type': {
                'eval_model': 'catboost',
                'eval_type': 'synthetic'
            },
            'T': {
                'seed': 0,
                'normalization': '__none__',  # Use placeholder
                'num_nan_policy': '__none__',
                'cat_nan_policy': '__none__',
                'cat_min_frequency': '__none__',
                'cat_encoding': '__none__',
                'y_policy': 'default'
            }
        }
    }
    
    return config

def run_ddpm_pipeline(config_path, train_flag=False, sample_flag=False, eval_flag=False, change_val=False):
    """
    Simplified function to run DDPM pipeline from notebook
    """
    
    raw_config = lib.load_config(config_path)
    
    # Convert placeholders back to None
    raw_config = convert_placeholder_to_none(raw_config)
    
    # Set device
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    timer = zero.Timer()
    timer.run()
    
    # Training
    if train_flag:
        print("Starting training...")
        
        # Prepare training arguments, avoiding duplicate T_dict
        train_args = {
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            'parent_dir': raw_config['parent_dir'],
            'real_data_path': raw_config['real_data_path'],
            'model_type': raw_config['model_type'],
            'model_params': raw_config['model_params'],
            'T_dict': raw_config['train']['T'],
            'num_numerical_features': raw_config['num_numerical_features'],
            'device': device,
            'change_val': change_val
        }
        
        train(**train_args)
        print("Training completed.")
    
    # Sampling
    if sample_flag:
        print("Starting sampling...")
        
        # Prepare sampling arguments, avoiding duplicate T_dict
        sample_args = {
            'num_samples': raw_config['sample']['num_samples'],
            'batch_size': raw_config['sample']['batch_size'],
            'disbalance': raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            'parent_dir': raw_config['parent_dir'],
            'real_data_path': raw_config['real_data_path'],
            'model_path': os.path.join(raw_config['parent_dir'], 'model.pt'),
            'model_type': raw_config['model_type'],
            'model_params': raw_config['model_params'],
            'T_dict': raw_config['train']['T'],
            'num_numerical_features': raw_config['num_numerical_features'],
            'device': device,
            'seed': raw_config['sample'].get('seed', 0),
            'change_val': change_val
        }
        
        sample(**sample_args)
        
        # Save info file
        info_src = os.path.join(raw_config['real_data_path'], 'info.json')
        if os.path.exists(info_src):
            save_file(raw_config['parent_dir'], info_src)
        
        print("Sampling completed.")
    
    # Evaluation
    if eval_flag:
        print("Starting evaluation...")
        eval_model = raw_config['eval']['type']['eval_model']
        
        if eval_model == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=change_val
            )
        elif eval_model == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=change_val,
                device=device
            )
        elif eval_model == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=change_val
            )
        print("Evaluation completed.")
    
    print(f'Total elapsed time: {str(timer)}')
    return raw_config

def quick_csv_run(csv_path, target_column, dataset_name='my_dataset', output_dir='./ddpm_output', 
                  train=True, sample=True, eval=True):
    """
    Complete workflow: CSV -> TabDDPM format -> Train -> Sample -> Eval
    """
    print("=== Quick CSV to TabDDPM Pipeline ===")
    
    # Step 1: Convert CSV to TabDDPM format
    print("\n1. Converting CSV to TabDDPM format...")
    train_size, data_path, dataset_info = create_tabddpm_dataset_from_csv(
        csv_path, dataset_name, target_column
    )
    
    # Step 2: Create config
    print("\n2. Creating config...")
    config = create_simple_config(data_path, dataset_info, output_dir)
    
    # Save config
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'config.toml')
    with open(config_path, 'wb') as f:
        tomli_w.dump(config, f)
    
    print(f"   ✅ Config saved to: {config_path}")
    print(f"   📊 Dataset info:")
    print(f"      Training size: {train_size}")
    print(f"      Numerical features: {dataset_info['n_num_features']}")
    print(f"      Categorical features: {dataset_info['n_cat_features']}")
    print(f"      Task type: {dataset_info['task_type']}")
    
    # Step 3: Run pipeline
    print("\n3. Running TabDDPM pipeline...")
    result = run_ddpm_pipeline(
        config_path=config_path,
        train_flag=train,
        sample_flag=sample,
        eval_flag=eval
    )
    
    print(f"\n✅ Pipeline completed!")
    print(f"   📁 Results saved to: {output_dir}")
    
    return result

def main():
    """Original main function for command line usage"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--csv', metavar='FILE', help='Path to CSV file')
    parser.add_argument('--target', metavar='COLUMN', help='Target column name')
    parser.add_argument('--dataset', metavar='NAME', default='my_dataset', help='Dataset name')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--change_val', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.csv and args.target:
        # CSV workflow
        quick_csv_run(
            csv_path=args.csv,
            target_column=args.target,
            dataset_name=args.dataset,
            train=args.train,
            sample=args.sample,
            eval=args.eval
        )
    elif args.config:
        # Config workflow
        run_ddpm_pipeline(
            config_path=args.config,
            train_flag=args.train,
            sample_flag=args.sample,
            eval_flag=args.eval,
            change_val=args.change_val
        )
    else:
        print("Either provide --csv and --target, or --config")

if __name__ == '__main__':
    main()