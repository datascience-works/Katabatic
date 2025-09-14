import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
import os
import os
import warnings
from absl import app, flags
# import tensorflow as tf
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from prdc import compute_prdc
from tqdm import tqdm
from inspect import isfunction
import torch.nn as nn
import math
import os
from absl import flags
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
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



def get_act(FLAGS):
  if FLAGS.activation.lower() == 'elu':
    return nn.ELU()
  elif FLAGS.activation.lower() == 'relu':
    return nn.ReLU()
  elif FLAGS.activation.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif FLAGS.activation.lower() == 'swish':
    return nn.SiLU()
  elif FLAGS.activation.lower() == 'tanh':
    return nn.Tanh()
  elif FLAGS.activation.lower() == 'softplus':
    return nn.Softplus()
  else:
    raise NotImplementedError('activation function does not exist!')

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  
  half_dim = embedding_dim // 2
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1: 
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

get_act = get_act
default_initializer = default_init

class tabularUnet(nn.Module):
  def __init__(self, FLAGS):
    super().__init__()
    
    self.FLAGS = FLAGS

    self.embed_dim = FLAGS.nf
    tdim = self.embed_dim*4
    self.act = get_act(FLAGS)

    modules = []
    modules.append(nn.Linear(self.embed_dim, tdim))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(tdim, tdim))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    cond = FLAGS.cond_size
    cond_out = (FLAGS.input_size)//2
    if cond_out < 2:
      cond_out = FLAGS.input_size
    modules.append(nn.Linear(cond, cond_out))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    self.all_modules = nn.ModuleList(modules)

    dim_in = FLAGS.input_size + cond_out
    dim_out = list(FLAGS.encoder_dim)[0]
    self.inputs = nn.Linear(dim_in, dim_out) # input layer

    self.encoder = Encoder(list(FLAGS.encoder_dim), tdim, FLAGS) # encoder

    dim_in = list(FLAGS.encoder_dim)[-1]
    dim_out = list(FLAGS.encoder_dim)[-1]
    self.bottom_block = nn.Linear(dim_in, dim_out) #bottom_layer
    
    self.decoder = Decoder(list(reversed(FLAGS.encoder_dim)), tdim, FLAGS) #decoder

    dim_in = list(FLAGS.encoder_dim)[0]
    dim_out = FLAGS.output_size
    self.outputs = nn.Linear(dim_in, dim_out) #output layer


  def forward(self, x, time_cond, cond):

    modules = self.all_modules 
    m_idx = 0

    #time embedding
    temb = get_timestep_embedding(time_cond, self.embed_dim)
    temb = modules[m_idx](temb)
    m_idx += 1
    temb= self.act(temb)
    temb = modules[m_idx](temb)
    m_idx += 1
    
    #condition layer
    cond = modules[m_idx](cond)
    m_idx += 1

    x = torch.cat([x, cond], dim=1).float()
    inputs = self.inputs(x) #input layer
    skip_connections, encoding = self.encoder(inputs, temb)
    encoding = self.bottom_block(encoding)
    encoding = self.act(encoding)
    x = self.decoder(skip_connections, encoding, temb) 
    outputs = self.outputs(x)

    return outputs



def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init



class Encoder(nn.Module):
  def __init__(self, encoder_dim, tdim, FLAGS):
    super(Encoder, self).__init__()
    self.encoding_blocks = nn.ModuleList()
    for i in range(len(encoder_dim)):
      if (i+1)==len(encoder_dim): break
      encoding_block = EncodingBlock(encoder_dim[i], encoder_dim[i+1], tdim, FLAGS)
      self.encoding_blocks.append(encoding_block)

  def forward(self, x, t):
    skip_connections = []
    for encoding_block in self.encoding_blocks:
      x, skip_connection = encoding_block(x, t)
      skip_connections.append(skip_connection)
    return skip_connections, x

class EncodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, FLAGS):
    super(EncodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in, dim_out),
        get_act(FLAGS)
    ) 
    self.temb_proj = nn.Sequential(
        nn.Linear(tdim, dim_out),
        get_act(FLAGS)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(dim_out, dim_out),
        get_act(FLAGS)
    )
    
  def forward(self, x, t):
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)
    skip_connection = x
    return x, skip_connection

class Decoder(nn.Module):
  def __init__(self, decoder_dim, tdim, FLAGS):
    super(Decoder, self).__init__()
    self.decoding_blocks = nn.ModuleList()
    for i in range(len(decoder_dim)):
      if (i+1)==len(decoder_dim): break
      decoding_block = DecodingBlock(decoder_dim[i], decoder_dim[i+1], tdim, FLAGS)
      self.decoding_blocks.append(decoding_block)

  def forward(self, skip_connections, x, t):
    zipped = zip(reversed(skip_connections), self.decoding_blocks)
    for skip_connection, decoding_block in zipped:
      x = decoding_block(skip_connection, x, t)
    return x

class DecodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, FLAGS):
    super(DecodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in*2, dim_in),
        get_act(FLAGS)
    )
    self.temb_proj = nn.Sequential(
        nn.Linear(tdim, dim_in),
        get_act(FLAGS)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        get_act(FLAGS)
    )
    
  def forward(self, skip_connection, x, t):
    
    x = torch.cat((skip_connection, x), dim=1)
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)

    return x

def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64).double()
        alphas = 1. - betas
        self.register_buffer('betas', betas)
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

    def make_x_t(self, x_0_con, t, noise):
        x_t_con = (
            extract(self.sqrt_alphas_bar, t, x_0_con.shape) * x_0_con +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0_con.shape) * noise)
        return x_t_con
    
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.mean_type = mean_type
        self.var_type = var_type

        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64).double()

        alphas = 1. - betas
        self.register_buffer(
            'betas', betas)
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )


    def p_mean_variance(self, x_t, t, cond, trans):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t, cond)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'tabular_datasets')

eps = 1e-8

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def index_to_log_onehot(x, num_classes):
    log_x = torch.log(x.float().clamp(min=1e-30))

    return log_x


class MultinomialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, shape, denoise_fn, FLAGS, timesteps=1000,
                 loss_type='vb_stochastic', parametrization='x0'):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes 
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, dtype=torch.float64).double()
        alphas = 1. - betas
        
        alphas = np.sqrt(alphas)
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        self.num_classes_column = np.concatenate([self.num_classes[i].repeat(self.num_classes[i]) for i in range(len(self.num_classes))])
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2))
        k=0
        kl_list = []
        for i in self.num_classes:
            sub = kl[:, k:i+k].mean(dim=1)
            kl_list.append(sub)
            k+=i
        kl = torch.stack(kl_list, 1)
        return kl
    
    def log_categorical(self, log_x_start, log_prob):
        kl = (log_x_start.exp() * log_prob)
        k=0
        kl_list = []
        for i in self.num_classes:
            sub =  kl[:, k:i+k].mean(dim=1)
            kl_list.append(sub)
            k+=i
        kl = torch.stack(kl_list, 1)

        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t -torch.tensor(np.log(self.num_classes_column)).to(log_1_min_alpha_t.device)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.tensor(np.log(self.num_classes_column)).to(log_1_min_cumprod_alpha.device)
        )

        return log_probs

    def predict_start(self, log_x_t, t, cond_con):
        x_t = log_x_t
        # out = self._denoise_fn(x_t, t, cond_con)
        out = self._denoise_fn(x_t.float(), t, cond_con)

        assert out.size(0) == x_t.size(0)

        k=0
        log_pred = torch.empty_like(out)
        full_sample=[]
        for i in range(len(self.num_classes)):
            out_column = out[:, k:self.num_classes[i]+k]
            log_pred[:, k:self.num_classes[i]+k] = F.log_softmax(out_column, dim=1) 
            k+=self.num_classes[i]
        
        return log_pred


    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start.to(torch.float64), log_EV_qxtmin_x0)


        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        k=0
        unnormed_logprobs_column_list=[]
        for i in range(len(self.num_classes)):
            unnormed_logprobs_column = unnormed_logprobs[:,k:self.num_classes[i]+k]
            k+=self.num_classes[i]
            for j in range(self.num_classes[i]):
                unnormed_logprobs_column_list.append(torch.logsumexp(unnormed_logprobs_column, dim=1, keepdim=True))
        unnormed_logprobs_column_ = torch.stack(unnormed_logprobs_column_list,1).squeeze()


        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs - unnormed_logprobs_column_

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, cond_con):
        # Ensure conditioning tensor is float32
        if cond_con is not None:
            cond_con = cond_con.float()
            
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t, cond_con = cond_con)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, cond_con = cond_con)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, t, cond_con):
        model_log_prob, log_x_recon = self.p_pred(log_x=log_x, t=t, cond_con=cond_con)
        out = self.log_sample_categorical(model_log_prob).to(log_x.device)
        return out

    def log_sample_categorical(self, logits):
        full_sample = []
        k=0
        for i in range(len(self.num_classes)):
            logits_column = logits[:,k:self.num_classes[i]+k]
            k+=self.num_classes[i]
            uniform = torch.rand_like(logits_column)
            gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
            sample = (gumbel_noise + logits_column).argmax(dim=1)
            col_t =np.zeros(logits_column.shape)
            col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
            full_sample.append(col_t)
        full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample


    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0).to(log_EV_qxt_x0.device)
        return log_sample


    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(torch.tensor(self.num_classes_column, device=device) * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob).mean(dim=1)
        return kl_prior

    def compute_Lt(self, log_x_start, log_x_t, t, cond_con, detach_mean=False):
        # Ensure conditioning tensor is float32
        if cond_con is not None:
            cond_con = cond_con.float()
            
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob, log_x_recon = self.p_pred(log_x=log_x_t, t=t, cond_con=cond_con)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob).mean(dim=1)

        decoder_nll = -self.log_categorical(log_x_start, log_model_prob).mean(dim=1)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss, log_x_recon


_MODELS = {
    'binary_classification': [ # 184
         {
             'class': DecisionTreeClassifier, # 48
             'kwargs': {
                 'max_depth': [4, 8, 16, 32], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8]
             }
         },
         {
             'class': AdaBoostClassifier, # 4
             'kwargs': {
                 'n_estimators': [10, 50, 100, 200]
             }
         },
         {
            'class': LogisticRegression, # 36
            'kwargs': {
                 'solver': ['lbfgs'],
                 'n_jobs': [-1],
                 'max_iter': [10, 50, 100, 200],
                 'C': [0.01, 0.1, 1.0],
                 'tol': [1e-01, 1e-02, 1e-04]
             }
         },
        {
            'class': MLPClassifier, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': RandomForestClassifier, # 48
            'kwargs': {
                 'max_depth': [8, 16, None], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8],
                'n_jobs': [-1]

            }
        },
        {
            'class': XGBClassifier, # 36
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1],
                 'tree_method': ['gpu_hist']
            },
        }

    ],
    'multiclass_classification': [ # 132
        
        {
            'class': MLPClassifier, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
         {
             'class': DecisionTreeClassifier, # 48
             'kwargs': {
                 'max_depth': [4, 8, 16, 32], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8]
             }
         },
        {
            'class': RandomForestClassifier, # 36
            'kwargs': {
                 'max_depth': [8, 16, None], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8],
                 'n_jobs': [-1]

            }
        },
        {
            'class': XGBClassifier, # 36
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1],
                 'tree_method': ['gpu_hist']
            }
        }

    ],
    'regression': [ # 84
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor, # 12
            'kwargs': {
                'hidden_layer_sizes': [(100, ), (200, ), (100, 100)],
                'max_iter': [50, 100],
                'alpha': [0.0001, 0.001]
            }
        },
        {
            'class': XGBRegressor, # 36 
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 1.0],
                 'objective': ['reg:linear'],
                 'nthread': [-1],
                 'tree_method': ['gpu_hist']
            }
        },
        {
            'class': RandomForestRegressor, # 36
            'kwargs': {
                 'max_depth': [8, 16, None], 
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 4, 8],
                 'n_jobs': [-1]
            }
        }
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, val, test, metadata, eval): 
    fm = FeatureMaker(metadata)
    x_trains, y_trains = [], []

    for i in train:
        x_train, y_train = fm.make_features(i)
        x_trains.append(x_train)
        y_trains.append(y_train)

    x_val, y_val = fm.make_features(val)
    if eval is None:
        x_test = None
        y_test = None
    else:
        x_test, y_test = fm.make_features(test)
    model = _MODELS[metadata['problem_type']]

    return x_trains, y_trains, x_val, y_val, x_test, y_test, model


def _weighted_f1(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    classes = list(report.keys())[:-3]
    proportion = [  report[i]['support'] / len(y_test) for i in classes]
    weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
    return weighted_f1 


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_multi_classification(train, test, fake, metadata, eval):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = _prepare_ml_problem(fake, train, test, metadata, eval)
    best_f1_scores = []
    unique_labels = np.unique(y_trains[0])
    
    if eval is None:
        for model_spec in classifiers:
            model_class = model_spec['class']
            model_kwargs = model_spec.get('kwargs', dict())
            model_repr = model_class.__name__

            param_set = list(ParameterGrid(model_kwargs))

            results = []
            for param in tqdm(param_set):
                model = model_class(**param)

                try:
                    model.fit(x_trains[0], y_trains[0])
                except:
                    # pass 
                    continue
                
                if len(unique_labels) != len(np.unique(y_valid)):
                    pred = [unique_labels[0]] * len(x_valid)
                    pred_prob = np.array([1.] * len(x_valid))
                else:
                    try:
                        pred = model.predict(x_valid)
                        pred_prob = model.predict_proba(x_valid)
                    except:
                        continue

                macro_f1 = f1_score(y_valid, pred, average='macro')
                weighted_f1 = _weighted_f1(y_valid, pred)
                acc = accuracy_score(y_valid, pred)

                # 3. auroc
                size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                rest_label = set(range(size)) - set(unique_labels)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_valid.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis].reshape(x_valid.shape[0],1))
                        j += 1
                try:
                    roc_auc = roc_auc_score(np.eye(size)[y_valid], np.hstack(tmp), multi_class='ovr')
                except ValueError:
                    roc_auc = None
                results.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc
                    }
                )

            if results:
                results = pd.DataFrame(results)   
                best_f1_scores.append(results.values[results.macro_f1.idxmax()])
            else:
                best_f1_scores.append([model_repr, {}, 0.0, 0.0, None, 0.0])

    else:
        params = eval
        i=0
        for model_spec in classifiers:
            model_class = model_spec['class']
            model_kwargs = model_spec.get('kwargs', dict())
            model_repr = model_class.__name__

            def _calc(best_model):
                best_scores = []
                for x_train, y_train in zip(x_trains, y_trains):
                    try:
                        best_model.fit(x_train, y_train)
                    except:
                        # pass 
                        continue

                    unique_labels = np.unique(y_train)

                    if len(unique_labels) != len(np.unique(y_test)):
                        pred = [unique_labels[0]] * len(x_test)
                        pred_prob = np.array([1.] * len(x_test))
                    else:
                        try:
                            pred = best_model.predict(x_test)
                            pred_prob = best_model.predict_proba(x_test)
                        except:
                            continue

                    macro_f1 = f1_score(y_test, pred, average='macro')
                    weighted_f1 = _weighted_f1(y_test, pred)
                    acc = accuracy_score(y_test, pred)

                    # 3. auroc
                    size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                    rest_label = set(range(size)) - set(unique_labels)
                    tmp = []
                    j = 0
                    for i in range(size):
                        if i in rest_label:
                            tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                        else:
                            try:
                                tmp.append(pred_prob[:,[j]])
                            except:
                                tmp.append(pred_prob[:, np.newaxis].reshape(x_test.shape[0],1))
                            j += 1
                    try:
                        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp), multi_class='ovr')
                    except:
                        roc_auc = None
                        
                    best_scores.append(
                        {   
                            "name": model_repr,
                            "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1,
                            "roc_auc": roc_auc, 
                            "accuracy": acc
                        }
                    )
                # return pd.DataFrame(best_scores).mean(axis=0)
                return pd.DataFrame(best_scores).mean(axis=0) if best_scores else pd.Series({
                    "name": model_repr, "macro_f1": 0.0, "weighted_f1": 0.0, "roc_auc": None, "accuracy": 0.0
                })

            def _df(dataframe):
                return {
                    "name": model_repr,
                    "macro_f1": dataframe.macro_f1,
                    "roc_auc": dataframe.roc_auc,
                    "weighted_f1": dataframe.weighted_f1,
                    "accuracy": dataframe.accuracy,
                }

            # best_f1_scores.append(_df(_calc(model_class(**params['param'][i]))))
            # i+=1
            if i < len(params['param']):
                best_f1_scores.append(_df(_calc(model_class(**params['param'][i]))))
            else:
                best_f1_scores.append({
                    "name": model_repr, "macro_f1": 0.0, "roc_auc": None, 
                    "weighted_f1": 0.0, "accuracy": 0.0
                })
            i+=1

    if eval is None:
        return pd.DataFrame(best_f1_scores, columns=['name', 'param', 'macro_f1', 'weighted_f1', 'roc_auc', 'accuracy']), None, None
    else:
        return pd.DataFrame(best_f1_scores), None, None


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_binary_classification(train, test, fake, metadata, eval):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = _prepare_ml_problem(fake, train, test, metadata, eval)

    best_f1_scores = []
    unique_labels = np.unique(y_trains[0])
    if eval is None:
        for model_spec in classifiers:
            model_class = model_spec['class']
            model_kwargs = model_spec.get('kwargs', dict())
            model_repr = model_class.__name__


            param_set = list(ParameterGrid(model_kwargs))

            results = []
            for param in tqdm(param_set):
                model = model_class(**param)
                
                try:
                    model.fit(x_trains[0], y_trains[0])
                except ValueError:
                    # pass
                    continue

                if len(unique_labels) == 1:
                    pred = [unique_labels[0]] * len(x_valid)
                    pred_prob = np.array([1.] * len(x_valid))
                else:
                    try:
                        pred = model.predict(x_valid)
                        pred_prob = model.predict_proba(x_valid)
                    except:
                        continue

                binary_f1 = f1_score(y_valid, pred, average='binary')
                weighted_f1 = _weighted_f1(y_valid, pred)
                acc = accuracy_score(y_valid, pred)
                precision = precision_score(y_valid, pred, average='binary')
                recall = recall_score(y_valid, pred, average='binary')
                macro_f1 = f1_score(y_valid, pred, average='macro')

                # auroc
                size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                rest_label = set(range(size)) - set(unique_labels)
                tmp = []
                j = 0
                for i in range(size):
                    if i in rest_label:
                        tmp.append(np.array([0] * y_valid.shape[0])[:,np.newaxis])
                    else:
                        try:
                            tmp.append(pred_prob[:,[j]])
                        except:
                            tmp.append(pred_prob[:, np.newaxis].reshape(x_valid.shape[0],1))
                        j += 1
                try:
                    roc_auc = roc_auc_score(np.eye(size)[y_valid], np.hstack(tmp))
                except:
                    roc_auc = None

                results.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "binary_f1": binary_f1,
                        "weighted_f1": weighted_f1,
                        "roc_auc": roc_auc, 
                        "accuracy": acc, 
                        "precision": precision, 
                        "recall": recall, 
                        "macro_f1": macro_f1
                    }
                )

            if results:
                results = pd.DataFrame(results)  
                best_f1_scores.append(results.values[results.binary_f1.idxmax()])
            else:
                best_f1_scores.append([model_repr, {}, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0])
    else:
        params = eval
        i=0
        for model_spec in classifiers:
            model_class = model_spec['class']
            model_kwargs = model_spec.get('kwargs', dict())
            model_repr = model_class.__name__

            def _calc(best_model):
                best_scores = []
                for x_train, y_train in zip(x_trains, y_trains):
                    try:
                        best_model.fit(x_train, y_train)
                    except ValueError:
                        # pass
                        continue
                    unique_labels = np.unique(y_train)

                    if len(unique_labels) == 1:
                        pred = [unique_labels[0]] * len(x_test)
                        pred_prob = np.array([1.] * len(x_test))
                    else:
                        try:
                            pred = best_model.predict(x_test)
                            pred_prob = best_model.predict_proba(x_test)
                        except:
                            continue

                    binary_f1 = f1_score(y_test, pred, average='binary')
                    weighted_f1 = _weighted_f1(y_test, pred)
                    acc = accuracy_score(y_test, pred)
                    precision = precision_score(y_test, pred, average='binary')
                    recall = recall_score(y_test, pred, average='binary')
                    macro_f1 = f1_score(y_test, pred, average='macro')

                    # auroc
                    size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
                    rest_label = set(range(size)) - set(unique_labels)
                    tmp = []
                    j = 0
                    for i in range(size):
                        if i in rest_label:
                            tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                        else:
                            try:
                                tmp.append(pred_prob[:,[j]])
                            except:
                                tmp.append(pred_prob[:, np.newaxis].reshape(x_test.shape[0],1))
                            j += 1
                    try:
                        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
                    except ValueError:
                        # roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
                        roc_auc = None

                    best_scores.append(
                        {   
                            "name": model_repr,
                            # "param": param,
                            "binary_f1": binary_f1,
                            "weighted_f1": weighted_f1,
                            "roc_auc": roc_auc, 
                            "accuracy": acc, 
                            "precision": precision, 
                            "recall": recall, 
                            "macro_f1": macro_f1
                        }
                    )
                return pd.DataFrame(best_scores).mean(axis=0) if best_scores else pd.Series({
                    "name": model_repr, "binary_f1": 0.0, "weighted_f1": 0.0, "roc_auc": None, 
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0
                })

            def _df(dataframe):
                return {
                    "name": model_repr,
                    "binary_f1": dataframe.binary_f1,
                    "roc_auc": dataframe.roc_auc,
                    "weighted_f1": dataframe.weighted_f1,
                    "accuracy": dataframe.accuracy,
                }

            if i < len(params['param']):
                best_f1_scores.append(_df(_calc(model_class(**params['param'][i]))))
            else:
                best_f1_scores.append({
                    "name": model_repr, "binary_f1": 0.0, "roc_auc": None, 
                    "weighted_f1": 0.0, "accuracy": 0.0
                })
            i+=1

    if eval is None:
        return pd.DataFrame(best_f1_scores, columns=['name', 'param', 'binary_f1', 'weighted_f1', 'roc_auc', 'accuracy', 'precision', 'recall','macro_f1']), None, None
    else:
        return pd.DataFrame(best_f1_scores), None, None


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_regression(train, test, fake, metadata, eval):
    
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, regressors = _prepare_ml_problem(fake, train, test, metadata, eval)

    best_r2_scores = []


    y_trains = [np.log(np.clip(i, 1, None)) for i in y_trains]
    y_valid = np.log(np.clip(y_valid, 1, None))

    if eval is None:
        for model_spec in regressors:
            model_class = model_spec['class']
            model_kwargs = model_spec.get('kwargs', dict())
            model_repr = model_class.__name__

            param_set = list(ParameterGrid(model_kwargs))

            results = []
            for param in tqdm(param_set):
                model = model_class(**param)
                try:
                    model.fit(x_trains[0], y_trains[0])
                    pred = model.predict(x_valid)
                except:
                    continue

                r2 = r2_score(y_valid, pred)
                explained_variance = explained_variance_score(y_valid, pred)
                mean_squared = mean_squared_error(y_valid, pred)
                root_mean_squared = mean_squared_error(y_valid, pred, squared=False)
                mean_absolute = mean_absolute_error(y_valid, pred)

                results.append(
                    {   
                        "name": model_repr,
                        "param": param,
                        "r2": r2,
                        "explained_variance": explained_variance,
                        "mean_squared": mean_squared, 
                        "mean_absolute": mean_absolute, 
                        "rmse": root_mean_squared
                    }
                )

            if results:
                results = pd.DataFrame(results)
                best_r2_scores.append(results.values[results.r2.idxmax()])
            else:
                best_r2_scores.append([model_repr, {}, 0.0, 0.0, 0.0, 0.0, 0.0])

    else:
        y_test = np.log(np.clip(y_test, 1, None))
        params = eval
        i=0
        for model_spec in regressors:
            model_class = model_spec['class']
            model_kwargs = model_spec.get('kwargs', dict())
            model_repr = model_class.__name__

            def _calc(best_model):
                best_scores = []
                for x_train, y_train in zip(x_trains, y_trains):
                    try:
                        best_model.fit(x_train, y_train)
                        pred = best_model.predict(x_test)
                    except:
                        continue

                    r2 = r2_score(y_test, pred)
                    explained_variance = explained_variance_score(y_test, pred)
                    mean_squared = mean_squared_error(y_test, pred)
                    root_mean_squared = mean_squared_error(y_test, pred, squared=False)
                    mean_absolute = mean_absolute_error(y_test, pred)

                    best_scores.append(
                        {   
                            "name": model_repr,
                            "r2": r2,
                            "explained_variance": explained_variance,
                            "mean_squared": mean_squared, 
                            "mean_absolute": mean_absolute, 
                            "rmse": root_mean_squared
                        }
                    )

                return pd.DataFrame(best_scores).mean(axis=0) if best_scores else pd.Series({
                    "name": model_repr, "r2": 0.0, "explained_variance": 0.0, 
                    "mean_squared": 0.0, "mean_absolute": 0.0, "rmse": 0.0
                })

            def _df(dataframe):
                return {
                    "name": model_repr,
                    "r2": dataframe.r2,
                    "explained_variance": dataframe.explained_variance,
                    "MAE": dataframe.mean_absolute,
                    "RMSE": dataframe.rmse,
                }

            if i < len(params['param']):
                best_r2_scores.append(_df(_calc(model_class(**params['param'][i]))))
            else:
                best_r2_scores.append({
                    "name": model_repr, "r2": 0.0, "explained_variance": 0.0, 
                    "MAE": 0.0, "RMSE": 0.0
                })
            i+=1

    if eval is None:
        return pd.DataFrame(best_r2_scores, columns=['name', 'param', 'r2', 'explained_variance', 'mean_squared', 'mean_absolute', 'rmse']), None, None
    else:     
        return pd.DataFrame(best_r2_scores), None, None 

@ignore_warnings(category=ConvergenceWarning)
def compute_diversity(train, fake):
    nearest_k = 5
    if train.shape[0] >= 50000:
        num = np.random.randint(0, train.shape[0], 50000)
        real_features = train[num]
        fake_features_lst = [i[num] for i in fake]
    else:
        num = train.shape[0]
        real_features = train[:num]
        fake_features_lst = [i[:num] for i in fake]
    scores = []
    for i, data in enumerate(fake_features_lst):
        fake_features = data
        metrics = compute_prdc(real_features=real_features,
                        fake_features=fake_features,
                        nearest_k=nearest_k)
        metrics['i'] = i
        scores.append(metrics)
    return pd.DataFrame(scores).mean(axis=0), pd.DataFrame(scores).std(axis=0)

_EVALUATORS = {
    'binary_classification': _evaluate_binary_classification,
    'multiclass_classification': _evaluate_multi_classification,
    'regression': _evaluate_regression
}

def compute_scores(train, test, synthesized_data, metadata, eval):
    a, b, c = _EVALUATORS[metadata['problem_type']](train=train, test=test, fake=synthesized_data, metadata=metadata, eval=eval)
    if eval is None:
        return a.mean(axis=0), a.std(axis=0), a[['name','param']]
    else:
        return a.mean(axis=0), a.std(axis=0)



def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    
    if loader == np.load:
        return loader(local_path, allow_pickle=True)
    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()

    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)

    return categorical_columns


def load_data(name, benchmark=False):
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns = _get_columns(meta)
    train = data['train']
    test = data['test']


    return train, test, (categorical_columns, meta)

def get_dataset(FLAGS, evaluation=False):

  batch_size = FLAGS.training_batch_size if not evaluation else FLAGS.eval_batch_size

  if batch_size % torch.cuda.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({torch.cuda.device_count()})')


  # Create dataset builders for tabular data.
  train, test, cols = load_data(FLAGS.data)
  cols_idx = list(np.arange(train.shape[1]))
  dis_idx = cols[0]
  con_idx = [x for x in cols_idx if x not in dis_idx]
  
  #split continuous and categorical
  train_con = train[:,con_idx]
  train_dis = train[:,dis_idx]
  
  #new index
  cat_idx_ = list(np.arange(train_dis.shape[1]))[:len(cols[0])]

  transformer_con = GeneralTransformer()
  transformer_dis = GeneralTransformer()

  transformer_con.fit(train_con, [])
  transformer_dis.fit(train_dis, cat_idx_)

  train_con_data = transformer_con.transform(train_con)
  train_dis_data = transformer_dis.transform(train_dis)


  return train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, cols[1]), con_idx, dis_idx

class Transformer:

    @staticmethod
    def get_metadata(data, categorical_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else: 
                meta.append({
                    "name": index,
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    def fit(self, data, categorical_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class GeneralTransformer(Transformer):

    def __init__(self, act='tanh'):
        self.act = act
        self.meta = None
        self.output_dim = None

    def fit(self, data, categorical_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns)
        self.output_dim = 0
        for info in self.meta:
            if info['type'] in [CONTINUOUS]:
                self.output_dim += 1
            else:
                self.output_dim += info['size']

    def transform(self, data):
        data_t = []
        self.output_info = []
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info['type'] == CONTINUOUS:
                col = (col - (info['min'])) / (info['max'] - info['min'])
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))

            else:
                col_t = np.zeros([len(data), info['size']])
                idx = list(map(info['i2s'].index, col))
                col_t[np.arange(len(data)), idx] = 1
                data_t.append(col_t)
                self.output_info.append((info['size'], 'softmax'))

        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])

        data = data.copy()
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == 'tanh':
                    current = (current + 1) / 2

                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info['max'] - info['min']) + info['min']

            else:
                current = data[:, :info['size']]
                data = data[:, info['size']:]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))

        return data_t


def warmup_lr(step):
    return min(step, 5000) / 5000

def infiniteloop(dataloader):
    while True:
        for _, y in enumerate(dataloader):
            yield y

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'sigmoid':
            ed = st + item[0]
            data_t.append(data[:,st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.softmax(data[:, st:ed], dim=1))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)

def log_sample_categorical(logits, num_classes):
    full_sample = []
    k=0
    for i in range(len(num_classes)):
        logits_column = logits[:,k:num_classes[i]+k]
        k+=num_classes[i]
        uniform = torch.rand_like(logits_column)
        gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
        sample = (gumbel_noise + logits_column).argmax(dim=1)
        col_t =np.zeros(logits_column.shape)
        col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
        full_sample.append(col_t)
    full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
    log_sample = torch.log(full_sample.float().clamp(min=1e-30))
    return log_sample


def sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, trans, FLAGS):
    x_t_con = x_T_con
    x_t_dis = log_x_T_dis

    for time_step in reversed(range(FLAGS.T)):
        t = x_t_con.new_ones([x_t_con.shape[0], ], dtype=torch.long) * time_step
        mean, log_var = net_sampler.p_mean_variance(x_t=x_t_con, t=t, cond = x_t_dis.to(x_t_con.device), trans=trans)
        if time_step > 0:
            noise = torch.randn_like(x_t_con)
        elif time_step == 0:
            noise = 0
        x_t_minus_1_con = mean + torch.exp(0.5 * log_var) * noise
        x_t_minus_1_con = torch.clip(x_t_minus_1_con, -1., 1.)
        x_t_minus_1_dis = trainer_dis.p_sample(x_t_dis, t, x_t_con)
        x_t_con = x_t_minus_1_con
        x_t_dis = x_t_minus_1_dis

    return x_t_con, x_t_dis

def training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, trans, FLAGS):
    
    # Ensure all tensors are float32
    x_0_con = x_0_con.float()
    x_0_dis = x_0_dis.float()
    ns_con = ns_con.float()
    ns_dis = ns_dis.float()
    
    t = torch.randint(FLAGS.T, size=(x_0_con.shape[0], ), device=x_0_con.device)
    pt = torch.ones_like(t).float() / FLAGS.T

    #co-evolving training and predict positive samples
    noise = torch.randn_like(x_0_con)
    x_t_con = trainer.make_x_t(x_0_con, t, noise)
    log_x_start = torch.log(x_0_dis.float().clamp(min=1e-30))
    x_t_dis = trainer_dis.q_sample(log_x_start=log_x_start, t=t)
    eps = trainer.model(x_t_con, t, x_t_dis.to(x_t_con.device))
    ps_0_con = trainer.predict_xstart_from_eps(x_t_con, t, eps=eps)
    con_loss = F.mse_loss(eps, noise, reduction='none')
    con_loss = con_loss.mean()
    kl, ps_0_dis = trainer_dis.compute_Lt(log_x_start, x_t_dis, t, x_t_con)
    ps_0_dis = torch.exp(ps_0_dis)
    kl_prior = trainer_dis.kl_prior(log_x_start)
    dis_loss = (kl / pt + kl_prior).mean()

    # negative condition -> predict negative samples
    noise_ns = torch.randn_like(ns_con)
    ns_t_con = trainer.make_x_t(ns_con, t, noise_ns)
    log_ns_start = torch.log(ns_dis.float().clamp(min=1e-30))
    ns_t_dis = trainer_dis.q_sample(log_x_start=log_ns_start, t=t)
    eps_ns = trainer.model(x_t_con, t, ns_t_dis.to(ns_t_dis.device))
    ns_0_con = trainer.predict_xstart_from_eps(x_t_con, t, eps=eps_ns)
    _, ns_0_dis = trainer_dis.compute_Lt(log_x_start, x_t_dis, t, ns_t_con)
    ns_0_dis = torch.exp(ns_0_dis)
    
    # contrastive learning loss
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    triplet_con = triplet_loss(x_0_con, ps_0_con, ns_0_con)
    st=0
    triplet_dis = []
    for item in trans.output_info:
        ed = st + item[0]
        ps_dis = F.cross_entropy(ps_0_dis[:, st:ed], torch.argmax(x_0_dis[:, st:ed], dim=-1).long(), reduction='none')
        ns_dis = F.cross_entropy(ns_0_dis[:, st:ed], torch.argmax(x_0_dis[:, st:ed], dim=-1).long(), reduction='none')

        triplet_dis.append(max((ps_dis-ns_dis).mean()+1,0))
        st = ed
    triplet_dis = sum(triplet_dis)/len(triplet_dis)
    return con_loss, triplet_con, dis_loss, triplet_dis

def make_negative_condition(x_0_con, x_0_dis):

    device = x_0_con.device
    x_0_con = x_0_con.detach().cpu().numpy()
    x_0_dis = x_0_dis.detach().cpu().numpy()

    nsc_raw = pd.DataFrame(x_0_con)
    nsd_raw = pd.DataFrame(x_0_dis)
    nsc = np.array(nsc_raw.sample(frac=1, replace = False).reset_index(drop=True))
    nsd = np.array(nsd_raw.sample(frac=1, replace = False).reset_index(drop=True))
    ns_con = nsc
    ns_dis = nsd

    # return torch.tensor(ns_con).to(device), torch.tensor(ns_dis).to(device)
    return torch.tensor(ns_con, dtype=torch.float32).to(device), torch.tensor(ns_dis, dtype=torch.float32).to(device)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore", category=DeprecationWarning)

randomSeed = 2022
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)

FLAGS = flags.FLAGS
flags.DEFINE_string('data', 'heart', help='dataset')
flags.DEFINE_string('logdir', './codi_exp', help='log directory')
flags.DEFINE_bool('train', True, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate')

# Network Architecture
flags.DEFINE_multi_integer('encoder_dim', None, help='encoder_dim')
flags.DEFINE_string('encoder_dim_con', "64,128,256", help='encoder_dim_con')
flags.DEFINE_string('encoder_dim_dis', "64,128,256", help='encoder_dim_dis')
flags.DEFINE_integer('nf', None, help='nf')
flags.DEFINE_integer('nf_con', 16, help='nf_con')
flags.DEFINE_integer('nf_dis', 64, help='nf_dis')
flags.DEFINE_integer('input_size', None, help='input_size')
flags.DEFINE_integer('cond_size', None, help='cond_size')
flags.DEFINE_integer('output_size', None, help='output_size')
flags.DEFINE_string('activation', 'relu', help='activation')

# Training
flags.DEFINE_integer('num_samples', 100, help='number of synthetic samples to generate')  
flags.DEFINE_integer('training_batch_size', 2100, help='batch size')
flags.DEFINE_integer('eval_batch_size', 2100, help='batch size')
flags.DEFINE_integer('T', 50, help='total diffusion steps')
flags.DEFINE_float('beta_1', 0.00001, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_float('lr_con', 2e-03, help='target learning rate')
flags.DEFINE_float('lr_dis', 2e-03, help='target learning rate')
# flags.DEFINE_integer('total_epochs_both', 20000, help='total training steps')
flags.DEFINE_integer('total_epochs_both', 30, help='total training steps')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_bool('parallel', False, help='multi gpu training')

# Sampling
flags.DEFINE_integer('sample_step', 2000, help='frequency of sampling')

# Continuous diffusion model
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedsmall', ['fixedlarge', 'fixedsmall'], help='variance type')

# Contrastive Learning
flags.DEFINE_integer('ns_method', 0, help='negative condition method')
flags.DEFINE_float('lambda_con', 0.2, help='lambda_con')
flags.DEFINE_float('lambda_dis', 0.2, help='lambda_dis')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def co_evolving_condition(FLAGS):

    FLAGS = flags.FLAGS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load Datasets
    train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, meta), con_idx, dis_idx = get_dataset(FLAGS) 
    
    # Fix: Ensure data is float32
    train_con_data = train_con_data.astype(np.float32)
    train_dis_data = train_dis_data.astype(np.float32)
    
    train_iter_con = DataLoader(train_con_data, batch_size=FLAGS.training_batch_size)
    train_iter_dis = DataLoader(train_dis_data, batch_size=FLAGS.training_batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    num_class=[]
    for i in transformer_dis.output_info:
        num_class.append(i[0])
    num_class = np.array(num_class)
    
    if meta['problem_type'] == 'binary_classification': 
        metric = 'binary_f1'
    elif meta['problem_type'] == 'regression': metric = "r2"
    else: metric = 'macro_f1'
    
    # Condtinuous Diffusion Model Setup
    FLAGS.input_size = train_con_data.shape[1] 
    FLAGS.cond_size = train_dis_data.shape[1]
    FLAGS.output_size = train_con_data.shape[1]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_con.split(',')))
    FLAGS.nf =  FLAGS.nf_con
    model_con = tabularUnet(FLAGS)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=FLAGS.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.mean_type, FLAGS.var_type).to(device)

    FLAGS.input_size = train_dis_data.shape[1] 
    FLAGS.cond_size = train_con_data.shape[1]
    FLAGS.output_size = train_dis_data.shape[1]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_dis.split(',')))
    FLAGS.nf =  FLAGS.nf_dis
    model_dis = tabularUnet(FLAGS)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=FLAGS.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, FLAGS, timesteps=FLAGS.T,loss_type='vb_stochastic').to(device)

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)



    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    logging.info('Continuous model params: %d' % (num_params_con))
    logging.info('Discrete model params: %d' % (num_params_dis))

    scores_max_eval = -10

    total_steps_both = FLAGS.total_epochs_both * int(train.shape[0]/FLAGS.training_batch_size+1)
    sample_step = FLAGS.sample_step * int(train.shape[0]/FLAGS.training_batch_size+1)
    logging.info("Total steps: %d" %total_steps_both)
    logging.info("Sample steps: %d" %sample_step)
    logging.info("Continuous: %d, %d" %(train_con_data.shape[0], train_con_data.shape[1]))
    logging.info("Discrete: %d, %d"%(train_dis_data.shape[0], train_dis_data.shape[1]))

    # Start Training
    if FLAGS.eval==False:
        epoch = 0
        train_iter_con = DataLoader(train_con_data, batch_size=FLAGS.training_batch_size)
        train_iter_dis = DataLoader(train_dis_data, batch_size=FLAGS.training_batch_size)
        datalooper_train_con = infiniteloop(train_iter_con)
        datalooper_train_dis = infiniteloop(train_iter_dis)
        writer = SummaryWriter(FLAGS.logdir)
        writer.flush()
        for step in range(total_steps_both):
            model_con.train()
            model_dis.train()

            x_0_con = next(datalooper_train_con).to(device).float()
            # x_0_dis = next(datalooper_train_dis).to(device)
            x_0_dis = next(datalooper_train_dis).to(device).float()

            ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
            con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, transformer_dis, FLAGS)

            loss_con = con_loss + FLAGS.lambda_con * con_loss_ns
            loss_dis = dis_loss + FLAGS.lambda_dis * dis_loss_ns

            optim_con.zero_grad()
            loss_con.backward()
            torch.nn.utils.clip_grad_norm_(model_con.parameters(), FLAGS.grad_clip)
            optim_con.step()
            sched_con.step()

            optim_dis.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), FLAGS.grad_clip)#, self.args.clip_value)
            torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), FLAGS.grad_clip)#, self.args.clip_norm)
            optim_dis.step()
            sched_dis.step()

            # log
            writer.add_scalar('loss_continuous', con_loss, step)
            writer.add_scalar('loss_discrete', dis_loss, step)
            writer.add_scalar('loss_continuous_ns', con_loss_ns, step)
            writer.add_scalar('loss_discrete_ns', dis_loss_ns, step)
            writer.add_scalar('total_continuous', loss_con, step)
            writer.add_scalar('total_discrete', loss_dis, step)

            if (step+1) % int(train.shape[0]/FLAGS.training_batch_size+1) == 0:

                logging.info(f"Epoch :{epoch}, diffusion continuous loss: {con_loss:.3f}, discrete loss: {dis_loss:.3f}")
                logging.info(f"Epoch :{epoch}, CL continuous loss: {con_loss_ns:.3f}, discrete loss: {dis_loss_ns:.3f}")
                logging.info(f"Epoch :{epoch}, Total continuous loss: {loss_con:.3f}, discrete loss: {loss_dis:.3f}")
                epoch +=1

            if step > 0 and sample_step > 0 and step % sample_step == 0 or step==(total_steps_both-1):
                model_con.eval()
                model_dis.eval()
                with torch.no_grad():
                    x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
                    log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
                    x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
                x_dis = apply_activate(x_dis, transformer_dis.output_info)
                sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
                sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
                sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])
                for i in range(len(con_idx)):
                    sample[:,con_idx[i]]=sample_con[:,i]
                for i in range(len(dis_idx)):
                    sample[:,dis_idx[i]]=sample_dis[:,i]
                sample = np.array(pd.DataFrame(sample).dropna())
                scores, std, param = compute_scores(train=train, test = None, synthesized_data=[sample], metadata=meta, eval=None)
                div_mean, div_std = compute_diversity(train=train, fake=[sample])
                scores['coverage'] = div_mean['coverage']
                std['coverage'] = div_std['coverage']
                scores['density'] = div_mean['density']
                std['density'] = div_std['density']
                f1 = scores[metric]
                # logging.info(f"---------Epoch {epoch} Evaluation----------")
                # logging.info(scores)
                # logging.info(std)

                if scores_max_eval < torch.tensor(f1):
                    scores_max_eval = torch.tensor(f1)
                    logging.info(f"Save model!")
                    ckpt = {
                        'model_con': model_con.state_dict(),
                        'model_dis': model_dis.state_dict(),
                        'sched_con': sched_con.state_dict(),
                        'sched_dis': sched_dis.state_dict(),
                        'optim_con': optim_con.state_dict(),
                        'optim_dis': optim_dis.state_dict(),
                        'step': step,
                        'sample': sample, 
                        'ml_param': param
                    }
                    torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
        # logging.info(f"Evaluation best : {scores_max_eval}")

        # #final test
        # ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        # if os.path.exists(ckpt):
        #     model_con.load_state_dict(ckpt['model_con'])
        #     model_dis.load_state_dict(ckpt['model_dis'])
        #     model_con.eval()
        #     model_dis.eval()
        # else:
        #     print("No checkpoint found, using trained model")
        # fake_sample=[]
        # for i in range(10):
        #     logging.info(f"sampling {i}")
        #     with torch.no_grad():
        #         x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
        #         log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
        #         x_con, x_dis= sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
        #     x_dis = apply_activate(x_dis, transformer_dis.output_info)
        #     sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
        #     sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
        #     sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])
        #     for i in range(len(con_idx)):
        #         sample[:,con_idx[i]]=sample_con[:,i]
        #     for i in range(len(dis_idx)):
        #         sample[:,dis_idx[i]]=sample_dis[:,i]
        #     fake_sample.append(sample)
        
        # In co_evolving_condition.py, modify the sampling section:
        fake_sample=[]
        # Generate one dataset with the specified number of samples
        logging.info(f"sampling {FLAGS.num_samples} samples")
        with torch.no_grad():
            x_T_con = torch.randn(FLAGS.num_samples, train_con_data.shape[1]).to(device)  # Use FLAGS.num_samples instead of train_con_data.shape[0]
            log_x_T_dis = log_sample_categorical(torch.zeros((FLAGS.num_samples, train_dis_data.shape[1]), device=device), num_class).to(device)
            x_con, x_dis= sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
        x_dis = apply_activate(x_dis, transformer_dis.output_info)
        sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
        sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
        sample = np.zeros([FLAGS.num_samples, len(con_idx+dis_idx)])  # Use FLAGS.num_samples
        for i in range(len(con_idx)):
            sample[:,con_idx[i]]=sample_con[:,i]
        for i in range(len(dis_idx)):
            sample[:,dis_idx[i]]=sample_dis[:,i]
        fake_sample.append(sample)
            
        import pickle
        # Save all synthetic samples as pickle
        with open(os.path.join(FLAGS.logdir, 'synthetic_data.pkl'), 'wb') as f:
            pickle.dump(fake_sample, f)    
        
        # # Save individual samples as numpy arrays
        # for i, sample in enumerate(fake_sample):
        #     np.save(os.path.join(FLAGS.logdir, f'synthetic_sample_{i}.npy'), sample)

        # logging.info(f"Saved {len(fake_sample)} synthetic datasets to {FLAGS.logdir}")
        # logging.info(f"Files: synthetic_data.pkl and synthetic_sample_0.npy to synthetic_sample_{len(fake_sample)-1}.npy")
        
        # scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        # div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        # scores['coverage'] = div_mean['coverage']
        # std['coverage'] = div_std['coverage']
        # scores['density'] = div_mean['density']
        # std['density'] = div_std['density']
        # logging.info(f"---------Test----------")
        # logging.info(scores)
        # logging.info(std)

    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        model_con.load_state_dict(ckpt['model_con'])
        model_dis.load_state_dict(ckpt['model_dis'])
        model_con.eval()
        model_dis.eval()
        fake_sample = []
        for i in range(5):
            logging.info(f"sampling {i}")
            with torch.no_grad():
                x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
                log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
                x_con, x_dis= sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, FLAGS)
            x_dis = apply_activate(x_dis, transformer_dis.output_info)
            sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
            sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
            sample = np.zeros([train_con_data.shape[0], len(con_idx+dis_idx)])
            for i in range(len(con_idx)):
                sample[:,con_idx[i]]=sample_con[:,i]
            for i in range(len(dis_idx)):
                sample[:,dis_idx[i]]=sample_dis[:,i]
            fake_sample.append(sample)
        scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        scores['coverage'] = div_mean['coverage']
        std['coverage'] = div_std['coverage']
        scores['density'] = div_mean['density']
        std['density'] = div_std['density']
        logging.info(f"---------Test----------")
        logging.info(scores)
        logging.info(std)

def main(argv):

    if FLAGS.eval == True:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(FLAGS.logdir,exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.logdir, 'eval.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    else:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(FLAGS.logdir,exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.logdir, 'train.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    
    logging.info("Co-evolving Conditional Diffusion models")
    # co_evolving_condition.train(FLAGS)
    co_evolving_condition(FLAGS)
    
if __name__ == '__main__':
    app.run(main)