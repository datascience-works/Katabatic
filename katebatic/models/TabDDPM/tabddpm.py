# tabddpm_wrapper.py
import shutil
import os
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import zero
import lib

# Import from local modules
from .utils import make_dataset, get_model as get_model_arch  # main model builder
from .utils import (
    Transformations,
    prepare_fast_dataloader,
    cat_encode,
    normalize,
    round_columns
)
from .utils import dump_json

# Core training and sampling functions
from .utils import train as train_fn
from .utils import sample as sample_fn
from .utils import GaussianMultinomialDiffusion
class TabDDPM:
    """
    A sklearn-style wrapper for the Tabular DDPM model.
    
    Example:
        model = TabDDPM(steps=1000, lr=0.001, ...)
        model.fit(X_train, y_train)
        X_gen, y_gen = model.generate(1000)
    """

    def __init__(
        self,
        steps=1000,
        lr=0.001,
        weight_decay=1e-5,
        batch_size=2048,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        model_type='mlp',
        d_layers=None,
        dropout=0.0,
        num_classes=0,
        is_y_cond=False,
        normalization='quantile',  # 'standard', 'minmax', 'rank', 'quantile'
        num_nan_policy=None, #Mean
        cat_nan_policy=None, #Mode
        cat_min_frequency=None, #Numerical value as 0.01
        cat_encoding=None,  # or 'one-hot'
        y_policy='default',  # ignored for regression
        device=None,
        seed=0,
        parent_dir=None
    ):
        # Training params
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # Diffusion params
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler

        # Model params
        self.model_type = model_type
        self.d_layers = d_layers or [256, 256]
        self.dropout = dropout
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        # Preprocessing
        self.normalization = normalization
        self.num_nan_policy = num_nan_policy
        self.cat_nan_policy = cat_nan_policy
        self.cat_min_frequency = cat_min_frequency
        self.cat_encoding = cat_encoding
        self.y_policy = y_policy

        # Runtime
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.parent_dir = parent_dir or f"exp/temp_{np.random.randint(1e9)}"
        
        # Internal state
        self.is_trained = False
        self.num_features_ = None
        self.cat_features_ = None
        self.num_transform = None
        self.cat_transform = None
        self.model_params_ = None
        self.d_in_ = None
        self.K_ = None
        self.real_data_path_ = None
        self.model_path_ = None

        # Reproducibility
        zero.improve_reproducibility(seed)

    def _make_real_data_dir(self, X, y, output_dir):
        """Save X, y as .npy files in a structured directory."""
        os.makedirs(output_dir, exist_ok=True)
        info = {}

        # Handle X: split into numerical and categorical
        if isinstance(X, pd.DataFrame):
            is_cat = X.dtypes == 'object'
            X_num = X.loc[:, ~is_cat].values if (~is_cat).any() else None
            X_cat = X.loc[:, is_cat].values if is_cat.any() else None
            self.num_features_ = list(X.loc[:, ~is_cat].columns)
            self.cat_features_ = list(X.loc[:, is_cat].columns)
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            # Assume all numerical unless specified otherwise
            X_num = X
            X_cat = None
            self.num_features_ = [f"num_{i}" for i in range(X.shape[1])]
            self.cat_features_ = []
        else:
            raise ValueError("X must be a DataFrame or 2D numpy array")

        if X_num is not None:
            np.save(os.path.join(output_dir, "X_num_train.npy"), X_num)
            info["n_num_features"] = X_num.shape[1]
            self.num_numerical_features_ = X_num.shape[1]
        else:
            info["n_num_features"] = 0
            self.num_numerical_features_ = 0


        if X_cat is not None:
            np.save(os.path.join(output_dir, "X_cat_train.npy"), X_cat)
            info["n_cat_features"] = X_cat.shape[1]
        else:
            info["n_cat_features"] = 0

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            np.save(os.path.join(output_dir, "y_train.npy"), y)
            info["y_dim"] = 1
            info["is_y_cond"] = bool(self.is_y_cond)
            info["y_policy"] = self.y_policy
            info["num_classes"] = int(self.num_classes) if self.num_classes else 0
        else:
            info["y_dim"] = 0
            info["is_y_cond"] = False

        # Save info.json
        info["train_size"] = len(X)
        info["val_size"] = len(X)
        info["test_size"] = len(X) 
        info["is_registered"] = False
        # info["task_type"] = "binclass"  # or "regression", "multiclass"

        # 🔑 Required metadata
        if self.num_classes == 0:
            info["task_type"] = "regression"
            info["is_regression"] = True
            info["is_multiclass"] = False
            info["is_binclass"] = False
            info["n_classes"] = None
        elif self.num_classes == 2:
            info["task_type"] = "binclass"
            info["is_regression"] = False
            info["is_binclass"] = True
            info["is_multiclass"] = False
            info["n_classes"] = 2
        else:
            info["task_type"] = "multiclass"
            info["is_regression"] = False
            info["is_binclass"] = False
            info["is_multiclass"] = True
            info["n_classes"] = self.num_classes

        dump_json(info, os.path.join(output_dir, "info.json"))

        # 🔁 Create val and test files by copying train
        for key in ["X_num", "X_cat", "y"]:
            train_file = f"{key}_train.npy"
            val_file = f"{key}_val.npy"
            test_file = f"{key}_test.npy"
            train_path = os.path.join(output_dir, train_file)
            
            if os.path.exists(train_path):
                shutil.copyfile(train_path, os.path.join(output_dir, val_file))
                shutil.copyfile(train_path, os.path.join(output_dir, test_file))

        return output_dir

    def fit(self, X, y=None):
        """
        Fit the diffusion model on the given data.

        Args:
            X: pandas DataFrame or numpy array (n_samples, n_features)
            y: optional, target vector (n_samples,)
        """
        zero.improve_reproducibility(self.seed)

        # Create temporary real data directory
        self.real_data_path_ = os.path.join(self.parent_dir, "real_data")
        os.makedirs(self.real_data_path_, exist_ok=True)
        self._make_real_data_dir(X, y, self.real_data_path_)

        # Set up parent directory for model
        os.makedirs(self.parent_dir, exist_ok=True)

        # Infer number of numerical features
        if isinstance(X, pd.DataFrame):
            num_numerical_features = len([c for c in X.columns if X[c].dtype != 'object'])
        else:
            num_numerical_features = X.shape[1]

        self.model_params_ = {
            'is_y_cond': self.is_y_cond,
            'num_classes': self.num_classes,
            'rtdl_params': {
                'd_layers': self.d_layers or [256, 256],
                'dropout': self.dropout
            }
        }
        # Preprocessing transformation
        T_dict = {
            'seed': self.seed,
            'normalization': self.normalization,
            'num_nan_policy': None,
            'cat_nan_policy': None,
            'cat_min_frequency': None,
            'cat_encoding': None,
            'y_policy': self.y_policy
        }

        # Run training via train function
        train_fn(
            parent_dir=self.parent_dir,
            real_data_path=self.real_data_path_,
            steps=self.steps,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            model_type=self.model_type,
            model_params=deepcopy(self.model_params_),
            num_timesteps=self.num_timesteps,
            gaussian_loss_type=self.gaussian_loss_type,
            scheduler=self.scheduler,
            T_dict=T_dict,
            num_numerical_features=num_numerical_features,
            device=torch.device(self.device),
            seed=self.seed,
            change_val=False
        )

        # Mark as trained
        self.is_trained = True
        # self.model_path_ = os.path.join(self.parent_dir, "model.pt")

        self.model_path_ = os.path.join(self.parent_dir, "model.pt")
        if os.path.exists(self.model_path_):
            # Re-load dataset to get K and num_numerical_features
            T_dict = {
                'seed': self.seed,
                'normalization': self.normalization,
                'num_nan_policy': None,
                'cat_nan_policy': None,
                'cat_min_frequency': None,
                'cat_encoding': None,
                'y_policy': 'default'
            }
            T = Transformations(**T_dict)
            dataset = make_dataset(
                self.real_data_path_,
                T,
                num_classes=self.num_classes,
                is_y_cond=self.is_y_cond,
                change_val=False
            )
            self.K_ = np.array(dataset.get_category_sizes('train'))
            if len(self.K_) == 0 or T_dict['cat_encoding'] == 'one-hot':
                self.K_ = np.array([0])

            # Recompute d_in
            X_num = dataset.X_num['train'] if dataset.X_num is not None else None
            num_numerical_features = X_num.shape[1] if X_num is not None else 0
            self.d_in_ = sum(self.K_) + num_numerical_features

            # Also save num_numerical_features if needed
            # self.num_numerical_features_ = num_numerical_features
        
        return self

    def generate(self, n_samples):
        """
        Generate synthetic data.

        Returns:
            If y exists: (X_gen, y_gen)
            Else: X_gen
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before generating samples.")

        # Temporary sample config
        sample_fn(
            parent_dir=self.parent_dir,
            real_data_path=self.real_data_path_,
            num_samples=n_samples,
            batch_size=min(10000, n_samples),
            model_type=self.model_type,
            model_params=deepcopy(self.model_params_),
            model_path=self.model_path_,
            num_timesteps=self.num_timesteps,
            gaussian_loss_type=self.gaussian_loss_type,
            scheduler=self.scheduler,
            T_dict={
                'seed': self.seed,
                'normalization': self.normalization,
                'num_nan_policy': None,
                'cat_nan_policy': None,
                'cat_min_frequency': None,
                'cat_encoding': None,
                'y_policy': self.y_policy
            },
            
            num_numerical_features=self.num_numerical_features_,  
            disbalance=None,
            device=torch.device(self.device),
            seed=self.seed,
            change_val=False
        )

        # Load generated data
        X_num = np.load(os.path.join(self.parent_dir, "X_num_train.npy"), allow_pickle=True) if \
            os.path.exists(os.path.join(self.parent_dir, "X_num_train.npy")) else None
        X_cat = np.load(os.path.join(self.parent_dir, "X_cat_train.npy"), allow_pickle=True) if \
            os.path.exists(os.path.join(self.parent_dir, "X_cat_train.npy")) else None
        y_gen = np.load(os.path.join(self.parent_dir, "y_train.npy"), allow_pickle=True) if \
            os.path.exists(os.path.join(self.parent_dir, "y_train.npy")) else None

        # Combine X_num and X_cat
        if X_num is not None and X_cat is not None:
            X_gen = np.hstack([X_num, X_cat])
        elif X_num is not None:
            X_gen = X_num
        elif X_cat is not None:
            X_gen = X_cat
        else:
            raise RuntimeError("No generated data found!")

        # Return
        if y_gen is not None:
            return X_gen, y_gen
        else:
            return X_gen

    def get_original_column_names(self):
        """Return the inferred column names from training data."""
        if self.num_features_ is None and self.cat_features_ is None:
            return None
        return (self.num_features_ or []) + (self.cat_features_ or [])

    def generate_df(self, n_samples):
        """Generate as pandas DataFrame with original column names."""
        result = self.generate(n_samples)
        if isinstance(result, tuple):
            X, y = result
            df = pd.DataFrame(X, columns=self.get_original_column_names())
            df['target'] = y
            return df
        else:
            return pd.DataFrame(result, columns=self.get_original_column_names())
    

    def clear_cache(self, remove_exp_folder=False):
        """
        Remove all temporary files and directories created by this model.
        
        Args:
            remove_exp_folder (bool): If True, deletes the entire 'exp' folder.
                                      If False, only deletes this model's parent_dir.
        """
        # Remove the model's parent directory (e.g. exp/temp_12345)
        if hasattr(self, 'parent_dir') and self.parent_dir is not None:
            if os.path.exists(self.parent_dir):
                shutil.rmtree(self.parent_dir)
                print(f"Deleted model directory: {self.parent_dir}")
            else:
                print(f"Model directory not found: {self.parent_dir}")

        # Optionally remove the entire 'exp' folder
        if remove_exp_folder:
            exp_dir = "exp"
            if os.path.exists(exp_dir):
                shutil.rmtree(exp_dir)
                print(f"Deleted entire 'exp' folder: {exp_dir}")
            else:
                print(f"'exp' folder not found: {exp_dir}")

        # Reset paths
        self.parent_dir = None
        self.real_data_path_ = None
        self.model_path_ = None
        self.is_trained = False