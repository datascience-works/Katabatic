import numpy as np
import os
from torch.profiler import record_function
from inspect import isfunction

import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace, asdict, fields, is_dataclass
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List, Callable, get_args, get_origin, Type, TypeVar

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import statistics
import rtdl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zero
from torch import Tensor
import enum
from functools import partial
import scipy.special
import sklearn.metrics as skm
import atexit
import json
import pickle
import shutil
import sys
import time
import uuid
from pathlib import Path
from pprint import pprint
import __main__
import tomli
import tomli_w
import math

### Comment out this if you aren't running the model on colab
# Add the parent directory (project root) to the system path
"""
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
"""
###



ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']



class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)
        splits = [k for k in ['train', 'val', 'test'] if dir_.joinpath(f'y_{k}.npy').exists()]

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle=True))  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / 'info.json').exists():
            info = util.load_json(dir_ / 'info.json')
        else:
            info = None
        return Dataset(
            load('X_num') if dir_.joinpath('X_num_train.npy').exists() else None,
            load('X_cat') if dir_.joinpath('X_cat_train.npy').exists() else None,
            load('y'),
            {},
            TaskType(info['task_type']),
            info.get('n_classes'),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
        return metrics
    

@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
        num_transform = num_transform
    
    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
            return_encoder=True
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
        # return dataset, num_transform, cat_transform
    return dataset


def build_dataset(
    path: Union[str, Path],
    transformations: Transformations,
    cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y


def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    data_path: str,
    T: Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = load_json(os.path.join(data_path, 'info.json'))

    D = Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = change_val(D)
    
    return transform_dataset(D, T, None)


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


def change_val(dataset: Dataset, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)

    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)

    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]

    return dataset

def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer : bool = False
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample= int(1e9),
            random_state=seed,
        )
        # noise = 1e-3
        # if noise > 0:
        #     assert seed is not None
        #     stds = np.std(X_train, axis=0, keepdims=True)
        #     noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        #     X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
        #         X_train.shape
        #     )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        util.raise_unknown('policy', policy)
    return y, info


###############
## DataLoader##
###############

class TabDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset : Dataset, split : Literal['train', 'val', 'test']
    ):
        super().__init__()
        
        self.X_num = torch.from_numpy(dataset.X_num[split]) if dataset.X_num is not None else None
        self.X_cat = torch.from_numpy(dataset.X_cat[split]) if dataset.X_cat is not None else None
        self.y = torch.from_numpy(dataset.y[split])

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        out_dict = {
            'y': self.y[idx].long() if self.y is not None else None,
        }

        x = np.empty((0,))
        if self.X_num is not None:
            x = self.X_num[idx]
        if self.X_cat is not None:
            x = torch.cat([x, self.X_cat[idx]], dim=0)
        return x.float(), out_dict

def prepare_dataloader(
    dataset : Dataset,
    split : str,
    batch_size: int,
):

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=1,
    )
    while True:
        yield from loader

def prepare_torch_dataloader(
    dataset : Dataset,
    split : str,
    shuffle : bool,
    batch_size: int,
) -> torch.utils.data.DataLoader:

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader

def dataset_from_csv(paths : Dict[str, str], cat_features, target, T):
    assert 'train' in paths
    y = {}
    X_num = {}
    X_cat = {} if len(cat_features) else None
    for split in paths.keys():
        df = pd.read_csv(paths[split])
        y[split] = df[target].to_numpy().astype(float)
        if X_cat is not None:
            X_cat[split] = df[cat_features].to_numpy().astype(str)
        X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)

    dataset = Dataset(X_num, X_cat, y, {}, None, len(np.unique(y['train'])))
    return transform_dataset(dataset, T, None)

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def prepare_fast_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    while True:
        yield from dataloader

def prepare_fast_torch_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X

def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat([
            pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    if X_cat is not None:
        return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(X_cat, columns=list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)

def read_pure_data(path, split='train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)

    return X_num, X_cat, y

def read_changed_val(path, val_size=0.2):
    path = Path(path)
    X_num_train, X_cat_train, y_train = read_pure_data(path, 'train')
    X_num_val, X_cat_val, y_val = read_pure_data(path, 'val')
    is_regression = load_json(path / 'info.json')['task_type'] == 'regression'

    y = np.concatenate([y_train, y_val], axis=0)

    ixs = np.arange(y.shape[0])
    if is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)
    y_train = y[train_ixs]
    y_val = y[val_ixs]

    if X_num_train is not None:
        X_num = np.concatenate([X_num_train, X_num_val], axis=0)
        X_num_train = X_num[train_ixs]
        X_num_val = X_num[val_ixs]

    if X_cat_train is not None:
        X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
        X_cat_train = X_cat[train_ixs]
        X_cat_val = X_cat[val_ixs]
    
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val

#############

def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = Path("data/" + dataset_dir_name)
    info = util.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info

def cos_sin(x: Tensor) -> Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


@dataclass
class PeriodicOptions:
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']


class Periodic(nn.Module):
    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        if options.initialization == 'log-linear':
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == 'normal'
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer('coefficients', coefficients)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_loss_fn(task_type: TaskType) -> Callable[..., Tensor]:
    return (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )


def default_zero_weight_decay_condition(module_name, module, parameter_name, parameter):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            rtdl.CLSToken,
            rtdl.NumericalFeatureTokenizer,
            rtdl.CategoricalFeatureTokenizer,
            Periodic,
        ),
    )


def split_parameters_by_weight_decay(
    model: nn.Module, zero_weight_decay_condition=default_zero_weight_decay_condition
) -> list[dict[str, Any]]:
    parameters_info = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters():
            full_parameter_name = (
                f'{module_name}.{parameter_name}' if module_name else parameter_name
            )
            parameters_info.setdefault(full_parameter_name, ([], parameter))[0].append(
                zero_weight_decay_condition(
                    module_name, module, parameter_name, parameter
                )
            )
    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    for full_parameter_name, (results, parameter) in parameters_info.items():
        (params_without_wd if any(results) else params_with_wd)['params'].append(
            parameter
        )
    return [params_with_wd, params_without_wd]


def make_optimizer(
    config: dict[str, Any],
    parameter_groups,
) -> optim.Optimizer:
    if config['optimizer'] == 'FT-Transformer-default':
        return optim.AdamW(parameter_groups, lr=1e-4, weight_decay=1e-5)
    return getattr(optim, config['optimizer'])(
        parameter_groups,
        **{x: config[x] for x in ['lr', 'weight_decay', 'momentum'] if x in config},
    )


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def is_oom_exception(err: RuntimeError) -> bool:
    return any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def train_with_auto_virtual_batch(
    optimizer,
    loss_fn,
    step,
    batch,
    chunk_size: int,
) -> tuple[Tensor, int]:
    batch_size = len(batch)
    random_state = zero.random.get_state()
    loss = None
    while chunk_size != 0:
        try:
            zero.random.set_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = loss_fn(*step(batch))
                loss.backward()
            else:
                loss = None
                for chunk in zero.iter_batches(batch, chunk_size):
                    chunk_loss = loss_fn(*step(chunk))
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
    return cast(Tensor, loss), chunk_size


def process_epoch_losses(losses: list[Tensor]) -> tuple[list[float], float]:
    losses_ = torch.stack(losses).tolist()
    return losses_, statistics.mean(losses_)




class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'

class MetricsReport:
    def __init__(self, report: dict, task_type: TaskType):
        self._res = {k: {} for k in report.keys()}
        if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
            self._metrics_names = ["acc", "f1"]
            for k in report.keys():
                self._res[k]["acc"] = report[k]["accuracy"]
                self._res[k]["f1"] = report[k]["macro avg"]["f1-score"]
                if task_type == TaskType.BINCLASS:
                    self._res[k]["roc_auc"] = report[k]["roc_auc"]
                    self._metrics_names.append("roc_auc")

        elif task_type == TaskType.REGRESSION:
            self._metrics_names = ["r2", "rmse"]
            for k in report.keys():
                self._res[k]["r2"] = report[k]["r2"]
                self._res[k]["rmse"] = report[k]["rmse"]
        else:
            raise "Unknown TaskType!"

    def get_splits_names(self) -> list[str]:
        return self._res.keys()

    def get_metrics_names(self) -> list[str]:
        return self._metrics_names

    def get_metric(self, split: str, metric: str) -> float:
        return self._res[split][metric]

    def get_val_score(self) -> float:
        return self._res["val"]["r2"] if "r2" in self._res["val"] else self._res["val"]["f1"]
    
    def get_test_score(self) -> float:
        return self._res["test"]["r2"] if "r2" in self._res["test"] else self._res["test"]["f1"]
    
    def print_metrics(self) -> None:
        res = {
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]}
        }
    
        print("*"*100)
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])

        return res

class SeedsMetricsReport:
    def __init__(self):
        self._reports = []

    def add_report(self, report: MetricsReport) -> None:
        self._reports.append(report)
    
    def get_mean_std(self) -> dict:
        res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def print_result(self) -> dict:
        res = {split: {k: float(np.around(self._agg_res[split][k], 4)) for k in self._agg_res[split]} for split in ["val", "test"]}
        print("="*100)
        print("EVAL RESULTS:")
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])
        print("="*100)
        return res

def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        util.raise_unknown('prediction_type', prediction_type)

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert 'std' in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info['std'])
        r2 = skm.r2_score(y_true, y_pred)
        result = {'rmse': rmse, 'r2': r2}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == TaskType.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result


RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __str__(self) -> str:
        return self.value



class Timer(zero.Timer):
    @classmethod
    def launch(cls) -> 'Timer':
        timer = cls()
        timer.run()
        return timer


def update_training_log(training_log, data, metrics):
    def _update(log_part, data_part):
        for k, v in data_part.items():
            if isinstance(v, dict):
                _update(log_part.setdefault(k, {}), v)
            elif isinstance(v, list):
                log_part.setdefault(k, []).extend(v)
            else:
                log_part.setdefault(k, []).append(v)

    _update(training_log, data)
    transposed_metrics = {}
    for part, part_metrics in metrics.items():
        for metric_name, value in part_metrics.items():
            transposed_metrics.setdefault(metric_name, {})[part] = value
    _update(training_log, transposed_metrics)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)


def _get_output_item_path(
    path: Union[str, Path], filename: str, must_exist: bool
) -> Path:
    path = env.get_path(path)
    if path.suffix == '.toml':
        path = path.with_suffix('')
    if path.is_dir():
        path = path / filename
    else:
        assert path.name == filename
    assert path.parent.exists()
    if must_exist:
        assert path.exists()
    return path


def load_report(path: Path) -> Report:
    return load_json(_get_output_item_path(path, 'report.json', True))


def dump_report(report: dict, path: Path) -> None:
    dump_json(report, _get_output_item_path(path, 'report.json', False))


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(_get_output_item_path(path, 'predictions.npz', True)) as predictions:
        return {x: predictions[x] for x in predictions}


def dump_predictions(predictions: Dict[str, np.ndarray], path: Path) -> None:
    np.savez(_get_output_item_path(path, 'predictions.npz', False), **predictions)


def dump_metrics(metrics: Dict[str, Any], path: Path) -> None:
    dump_json(metrics, _get_output_item_path(path, 'metrics.json', False))


def load_checkpoint(path: Path, *args, **kwargs) -> Dict[str, np.ndarray]:
    return torch.load(
        _get_output_item_path(path, 'checkpoint.pt', True), *args, **kwargs
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def _print_sep(c, size=100):
    print(c * size)


def start(
    config_cls: Type[T] = RawConfig,
    argv: Optional[List[str]] = None,
    patch_raw_config: Optional[Callable[[RawConfig], None]] = None,
) -> Tuple[T, Path, Report]:  # config  # output dir  # report
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--continue', action='store_true', dest='continue_')
    if argv is None:
        program = __main__.__file__
        args = parser.parse_args()
    else:
        program = argv[0]
        try:
            args = parser.parse_args(argv[1:])
        except Exception:
            print(
                'Failed to parse `argv`.'
                ' Remember that the first item of `argv` must be the path (relative to'
                ' the project root) to the script/notebook.'
            )
            raise
    args = parser.parse_args(argv)

    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert args.continue_

    config_path = env.get_path(args.config)
    output_dir = config_path.with_suffix('')
    _print_sep('=')
    print(f'[output] {output_dir}')
    _print_sep('=')

    assert config_path.exists()
    raw_config = load_config(config_path)
    if patch_raw_config is not None:
        patch_raw_config(raw_config)
    if is_dataclass(config_cls):
        config = from_dict(config_cls, raw_config)
        full_raw_config = asdict(config)
    else:
        assert config_cls is dict
        full_raw_config = config = raw_config
    full_raw_config = asdict(config)

    if output_dir.exists():
        if args.force:
            print('Removing the existing output and creating a new one...')
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        elif not args.continue_:
            backup_output(output_dir)
            print('The output directory already exists. Done!\n')
            sys.exit()
        elif output_dir.joinpath('DONE').exists():
            backup_output(output_dir)
            print('The "DONE" file already exists. Done!')
            sys.exit()
        else:
            print('Continuing with the existing output...')
    else:
        print('Creating the output...')
        output_dir.mkdir()

    report = {
        'program': str(env.get_relative_path(program)),
        'environment': {},
        'config': full_raw_config,
    }
    if torch.cuda.is_available():  # type: ignore[code]
        report['environment'].update(
            {
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpus': zero.hardware.get_gpus_info(),
                'torch.version.cuda': torch.version.cuda,
                'torch.backends.cudnn.version()': torch.backends.cudnn.version(),  # type: ignore[code]
                'torch.cuda.nccl.version()': torch.cuda.nccl.version(),  # type: ignore[code]
            }
        )
    dump_report(report, output_dir)
    dump_json(raw_config, output_dir / 'raw_config.json')
    _print_sep('-')
    pprint(full_raw_config, width=100)
    _print_sep('-')
    return cast(config_cls, config), output_dir, report


_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(env.PROJ)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        # the case for evaluate.py which automatically creates configs
        if output_dir.with_suffix('.toml').exists():
            shutil.copyfile(
                output_dir.with_suffix('.toml'), new_output_dir.with_suffix('.toml')
            )
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def _get_scores(metrics: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, float]]:
    return (
        {k: v['score'] for k, v in metrics.items()}
        if 'score' in next(iter(metrics.values()))
        else None
    )


def format_scores(metrics: Dict[str, Dict[str, Any]]) -> str:
    return ' '.join(
        f"[{x}] {metrics[x]['score']:.3f}"
        for x in ['test', 'val', 'train']
        if x in metrics
    )


def finish(output_dir: Path, report: dict) -> None:
    print()
    _print_sep('=')

    metrics = report.get('metrics')
    if metrics is not None:
        scores = _get_scores(metrics)
        if scores is not None:
            dump_json(scores, output_dir / 'scores.json')
            print(format_scores(metrics))
            _print_sep('-')

    dump_report(report, output_dir)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if json_output_path:
        try:
            key = str(output_dir.relative_to(env.PROJ))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_json(output_dir / 'report.json')
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
        )

    output_dir.joinpath('DONE').touch()
    backup_output(output_dir)
    print(f'Done! | {report.get("time")} | {output_dir}')
    _print_sep('=')
    print()


def from_dict(datacls: Type[T], data: dict) -> T:
    assert is_dataclass(datacls)
    data = deepcopy(data)
    for field in fields(datacls):
        if field.name not in data:
            continue
        if is_dataclass(field.type):
            data[field.name] = from_dict(field.type, data[field.name])
        elif (
            get_origin(field.type) is Union
            and len(get_args(field.type)) == 2
            and get_args(field.type)[1] is type(None)
            and is_dataclass(get_args(field.type)[0])
        ):
            if data[field.name] is not None:
                data[field.name] = from_dict(get_args(field.type)[0], data[field.name])
    return datacls(**data)


def replace_factor_with_value(
    config: RawConfig,
    key: str,
    reference_value: int,
    bounds: Tuple[float, float],
) -> None:
    factor_key = key + '_factor'
    if factor_key not in config:
        assert key in config
    else:
        assert key not in config
        factor = config.pop(factor_key)
        assert bounds[0] <= factor <= bounds[1]
        config[key] = int(factor * reference_value)


def get_temporary_copy(path: Union[str, Path]) -> Path:
    path = env.get_path(path)
    assert not path.is_dir() and not path.is_symlink()
    tmp_path = path.with_name(
        path.stem + '___' + str(uuid.uuid4()).replace('-', '') + path.suffix
    )
    shutil.copyfile(path, tmp_path)
    atexit.register(lambda: tmp_path.unlink())
    return tmp_path


def get_python():
    python = Path('python3.9')
    return str(python) if python.exists() else 'python'

def get_catboost_config(real_data_path, is_cv=False):
    ds_name = Path(real_data_path).name
    C = load_json(f'tuned_models/catboost/{ds_name}_cv.json')
    return C



def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ohe_to_categories(ohe, K):
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i+1]].argmax(dim=1))
    return torch.stack(res, dim=1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
 
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot

def log_sum_exp_by_classes(x, slices):
    device = x.device
    res = torch.zeros_like(x)
    for ixs in slices:
        res[:, ixs] = torch.logsumexp(x[:, ixs], dim=1, keepdim=True)

    assert x.size() == res.size()

    return res

@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m

@torch.jit.script
def sliced_logsumexp(x, slices):
    slices = slices.long()  # ensure slices are integer indices

    lse = torch.logcumsumexp(
        torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')),
        dim=-1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(
        slice_lse,
        slice_ends - slice_starts, 
        dim=-1
    )
    return slice_lse_repeated

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

class FoundNANsError(BaseException):
    """Found NANs during sampling"""
    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)



ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    assert all(x is None for x in values) or all(x is not None for x in values)

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls: Type['ResNet'],
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> 'ResNet':
        """Create a "baseline" `ResNet`.
        This variation of ResNet was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`
        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
#### For diffusion 

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, num_classes, is_y_cond, rtdl_params, dim_t = 128):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        # d0 = rtdl_params['d_layers'][0]

        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)

        if self.num_classes > 0 and is_y_cond:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond:
            self.label_emb = nn.Linear(1, dim_t)
        
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            if self.num_classes > 0:
                y = y.squeeze()
            else:
                y = y.resize(y.size(0), 1).float()
            emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)

class ResNetDiffusion(nn.Module):
    def __init__(self, d_in, num_classes, rtdl_params, dim_t = 256):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes

        rtdl_params['d_in'] = d_in
        rtdl_params['d_out'] = d_in
        rtdl_params['emb_d'] = dim_t
        self.resnet = ResNet.make_baseline(**rtdl_params)

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if y is not None and self.num_classes > 0:
            emb += self.label_emb(y.squeeze())
        return self.resnet(x, emb)




class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)



    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    zero.improve_reproducibility(seed)

    T = Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            gaussian_parametrization='eps',
            multinomial_loss_type='vb_stochastic',
            parametrization='x0',
            scheduler='cosine',
            device=torch.device('cpu')
        ):

        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float().to(device)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))

        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))
    
    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]], dim=0)
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == 'mse':
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]


        return terms['loss']
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):

        # model_out = self._denoise_fn(x_t, t.to(x_t.device), **out_dict)

        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - sliced_logsumexp(unnormed_logprobs, self.offsets)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), out_dict)
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                out_dict=out_dict)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):

        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t, out_dict
            )
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return vb_loss

        elif self.multinomial_loss_type == 'vb_all':
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x, out_dict)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss
    
    def mixed_loss(self, x, out_dict):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        model_out = self._denoise_fn(
            x_in,
            t,
            **out_dict
        )

        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(self.num_classes)
        
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)

        # loss_multi = torch.where(out_dict['y'] == 1, loss_multi, 2 * loss_multi)
        # loss_gauss = torch.where(out_dict['y'] == 1, loss_gauss, 2 * loss_gauss)

        return loss_multi.mean(), loss_gauss.mean()
    
    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1),
                t_array,
                **out_dict
            )
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                    out_dict=out_dict
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)



        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        eta=0.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
    
    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise,
        T,
        out_dict,
        eta=0.0
    ):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(
                out_num,
                x,
                t_array
            )
        print()
        return x


    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
        out_dict,
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f'Reverse timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(
                out_num,
                x,
                t_array,
                eta=0.0
            )
        print()

        return x


    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat,
        log_x_t,
        t,
        out_dict,
        eta=0.0
    ):
        # not ddim, essentially
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict)

        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2
        

        log_ps = torch.stack([
            torch.log(coef1) + log_x_t,
            torch.log(coef2) + log_x0,
            torch.log(coef3) - torch.log(self.num_classes_expanded)
        ], dim=2)

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    

    @torch.no_grad()
    def sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    
    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        if ddim:
            print('Sample using DDIM.')
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        
        b = batch_size

        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict['y'] = out_dict['y'][~mask_nan]

            all_samples.append(sample)
            all_y.append(out_dict['y'].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]

        return x_gen, y_gen