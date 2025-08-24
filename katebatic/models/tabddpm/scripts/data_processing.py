import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_dataset_for_tabddpm(
    csv_path,
    dataset_name,
    target_column,
    task_type='auto',  # 'auto', 'binclass', 'multiclass', or 'regression'
    categorical_columns=None,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    data_dir='katebatic/models/tabddpm/data',
    exp_dir='katebatic/models/tabddpm/exp',
    max_cat_threshold=10,
    normalization='quantile'  # 'quantile', 'standard'
):
    """
    Prepare a CSV dataset for TabDDPM training with proper TaskType handling.
    
    Args:
        csv_path: Path to your CSV file
        dataset_name: Name for your dataset (will create directories with this name)
        target_column: Name of the target column
        task_type: 'auto', 'binclass', 'multiclass', or 'regression'
        categorical_columns: List of categorical column names (if None, will auto-detect)
        test_size: Proportion for test set
        val_size: Proportion for validation set (from training data)
        random_state: Random seed for reproducibility
        data_dir: Base directory for real data 
        exp_dir: Base directory for synthetic data 
        max_cat_threshold: Max unique values to consider a column categorical
        normalization: Normalization method for numerical features ('quantile', 'standard')
    """
    
    # Validate normalization method
    valid_normalizations = ['quantile', 'standard']
    if normalization not in valid_normalizations:
        print(f"Warning: normalization '{normalization}' not in {valid_normalizations}")
        print("Using default 'quantile' normalization")
        normalization = 'quantile'
    
    # Read the CSV
    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Normalization method: {normalization}")
    
    # Analyze target column to determine task type
    target_values = df[target_column].dropna()
    unique_targets = target_values.nunique()
    
    if task_type == 'auto':
        if target_values.dtype in ['object', 'category'] or unique_targets <= 20:
            if unique_targets == 2:
                task_type = 'binclass'
            else:
                task_type = 'multiclass'
        else:
            task_type = 'regression'
        print(f"Auto-detected task_type: '{task_type}' (based on {unique_targets} unique target values)")
    else:
        print(f"Using specified task_type: '{task_type}'")
    
    # Validate task_type
    valid_task_types = ['binclass', 'multiclass', 'regression']
    if task_type not in valid_task_types:
        print(f"Warning: task_type '{task_type}' not in {valid_task_types}")
        print("Converting to valid format...")
        if task_type in ['classification', 'binary']:
            task_type = 'binclass' if unique_targets == 2 else 'multiclass'
        elif task_type == 'regression':
            task_type = 'regression'
        else:
            # Default fallback
            task_type = 'binclass' if unique_targets == 2 else 'multiclass'
        print(f"Converted to: '{task_type}'")
    
    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = []
        numerical_columns = []
        for col in df.columns:
            if col == target_column:
                continue
            # Consider categorical if: object type, or few unique values, or explicitly boolean
            if (df[col].dtype == 'object' or 
                df[col].dtype == 'category' or
                df[col].nunique() <= max_cat_threshold or
                df[col].dtype == 'bool'):
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
    else:
        numerical_columns = [col for col in df.columns 
                            if col not in categorical_columns and col != target_column]
    
    print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")
    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    
    # Create directories
    data_path = Path(data_dir) / dataset_name
    exp_path = Path(exp_dir) / dataset_name / 'ddpm_cb_best'
    data_path.mkdir(parents=True, exist_ok=True)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Handle missing values
    print("Handling missing values...")
    if numerical_columns:
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
    if categorical_columns:
        for col in categorical_columns:
            df[col] = df[col].fillna('__missing__')
    
    # Prepare features and target
    X_num = df[numerical_columns].values.astype(np.float32) if numerical_columns else np.empty((len(df), 0), dtype=np.float32)
    X_cat = df[categorical_columns] if categorical_columns else pd.DataFrame()
    y = df[target_column].values
    
    # Encode categorical features
    cat_encoders = {}
    cat_cardinalities = []
    
    if len(categorical_columns) > 0:
        X_cat_encoded = np.zeros((len(df), len(categorical_columns)), dtype=np.int64)
        
        for i, col in enumerate(categorical_columns):
            # Convert to string type uniformly to avoid mixed type issues
            col_vals = df[col].astype(str)
            le = LabelEncoder()
            encoded = le.fit_transform(col_vals)
            X_cat_encoded[:, i] = encoded
            cat_encoders[col] = le
            cat_cardinalities.append(len(le.classes_))
            print(f"  {col}: {len(le.classes_)} categories")
    else:
        # Create empty array with correct shape for no categorical features
        X_cat_encoded = np.empty((len(df), 0), dtype=np.int64)
    
    # Encode target
    if task_type in ['binclass', 'multiclass']:
        # Convert to string uniformly to avoid mixed type issues
        y_as_str = pd.Series(y).astype(str).values
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y_as_str).astype(np.int64)
        num_classes = len(le_target.classes_)
        print(f"Target classes: {num_classes} ({le_target.classes_})")
    else:
        # For regression, ensure numeric type
        y_encoded = pd.to_numeric(pd.Series(y), errors='coerce').astype(np.float32).values
        num_classes = 0
        print(f"Regression target range: [{np.nanmin(y_encoded):.3f}, {np.nanmax(y_encoded):.3f}]")
    
    # Split the data
    indices = np.arange(len(df))
    
    # Stratify for classification tasks
    stratify_var = y_encoded if task_type in ['binclass', 'multiclass'] else None
    
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify_var
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size/(1-test_size), random_state=random_state,
        stratify=y_encoded[train_val_idx] if task_type in ['binclass', 'multiclass'] else None
    )
    
    print(f"Data splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Save all splits
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    for split_name, indices in splits.items():
        # Save numerical features (always save, even if empty)
        np.save(data_path / f'X_num_{split_name}.npy', X_num[indices])
        
        # Save categorical features only if they exist
        if len(categorical_columns) > 0:
            np.save(data_path / f'X_cat_{split_name}.npy', X_cat_encoded[indices])
        
        # Save targets
        np.save(data_path / f'y_{split_name}.npy', y_encoded[indices])
    
    # Convert categorical features to object dtype (required by TabDDPM)
    if len(categorical_columns) > 0:
        print("Converting categorical features to object dtype...")
        for split_name in ['train', 'val', 'test']:
            cat_path = data_path / f'X_cat_{split_name}.npy'
            X_cat_loaded = np.load(cat_path, allow_pickle=True)
            np.save(cat_path, X_cat_loaded.astype(object))
        print("Categorical features converted to object dtype")
    
    # Create info.json with correct task_type
    info = {
        "name": dataset_name,
        "task_type": task_type,  # This will be 'binclass', 'multiclass', or 'regression'
        "n_num_features": len(numerical_columns),
        "n_cat_features": len(categorical_columns),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "normalization": normalization,  
        "num_columns": numerical_columns, 
        "cat_columns": categorical_columns, 
        "target_column": target_column,
    }
    
    # Add class information for classification
    if task_type in ['binclass', 'multiclass']:
        info["n_classes"] = num_classes
        info["num_classes"] = num_classes  # Some parts of code expect this key
    else:
        info["n_classes"] = 0
        info["num_classes"] = 0
    
    # Save info.json
    with open(data_path / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nDataset prepared successfully!")
    print(f"Data saved to: {data_path}")
    print(f"Task type: {task_type}")
    print(f"Features: {len(numerical_columns)} numerical, {len(categorical_columns)} categorical")
    print(f"Normalization: {normalization}")
    
    # Generate config.toml with the specified normalization
    generate_config(exp_path, dataset_name, info, task_type, normalization)
    
    # Generate catboost hyperparameters
    generate_catboost_params(dataset_name)
    
    # return info
    return {
        "info": info,
        "real_data_path": data_path.as_posix(),
        "exp_parent_dir": exp_path.as_posix(),
        "config_path": (exp_path / "config.toml").as_posix(),
    }

def generate_config(exp_path, dataset_name, info, task_type, normalization='quantile'):
    """Generate config.toml file with proper TaskType and normalization"""
    
    # Determine if we need conditional generation
    is_y_cond = task_type in ['binclass', 'multiclass']
    
    config_content = f"""seed = 0
parent_dir = "katebatic/models/tabddpm/exp/{dataset_name}/ddpm_cb_best"
real_data_path = "katebatic/models/tabddpm/data/{dataset_name}/"
model_type = "mlp"
num_numerical_features = {info['n_num_features']}
device = "cuda:0"

[model_params]
num_classes = {info['num_classes']}
is_y_cond = {str(is_y_cond).lower()}
d_in = {info['n_num_features'] + info['n_cat_features']}

[model_params.rtdl_params]
d_layers = [256, 256]
dropout = 0.0

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 2000
lr = 0.002
weight_decay = 1e-05
batch_size = 1024

[train.T]
seed = 0
normalization = "{normalization}"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = {max(5000, info['train_size'])}
batch_size = 5000
seed = 0

[eval.type]
eval_model = "catboost"
eval_type = "synthetic"

[eval.T]
seed = 0
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"
"""
    
    config_path = exp_path / 'config.toml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config saved to: {config_path} with normalization={normalization}")

def generate_catboost_params(dataset_name):
    """Generate CatBoost hyperparameters file"""
    
    info_path = Path('data') / dataset_name / 'info.json'
    cat_features = []
    
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
            print(f"Dataset info loaded: n_num={info['n_num_features']}, n_cat={info['n_cat_features']}")
        
        n_num_features = info.get('n_num_features', 0)
        n_cat_features = info.get('n_cat_features', 0)
        
        # Categorical features come after numerical in the combined feature space
        if n_cat_features > 0:
            cat_features = list(range(n_num_features, n_num_features + n_cat_features))
            print(f"Detected categorical features at indices: {cat_features}")
    except Exception as e:
        print(f"Could not detect categorical features from info.json: {e}")
    
    catboost_params = {
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "leaf_estimation_iterations": 1,
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU",
        "thread_count": 4,
        "cat_features": cat_features
    }
    
    # Create tuned_models directory
    tuned_dir = Path('katebatic/models/tabddpm/tuned_models/catboost')
    tuned_dir.mkdir(parents=True, exist_ok=True)
    
    params_path = tuned_dir / f'{dataset_name}_cv.json'
    with open(params_path, 'w') as f:
        json.dump(catboost_params, f, indent=2)
    
    print(f"CatBoost params saved to: {params_path}")
    print(f"Parameters include cat_features: {cat_features}")