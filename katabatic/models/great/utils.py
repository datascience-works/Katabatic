import random
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

from datasets import Dataset


# -------------------------
# Basic helpers
# -------------------------
def _array_to_dataframe(
    data: pd.DataFrame | np.ndarray,
    columns=None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data

    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

    if columns is None:
        raise ValueError("Column names are required when input is numpy ndarray.")

    if len(columns) != data.shape[1]:
        raise ValueError(
            f"{len(columns)} column names given, but array has {data.shape[1]} columns."
        )

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> list | dict:
    if pd.api.types.is_float_dtype(df[col]):
        return df[col].dropna().to_list()

    return df[col].value_counts(normalize=True).to_dict()


def _convert_tokens_to_text(
    tokens: list[torch.Tensor],
    tokenizer: AutoTokenizer,
) -> list[str]:
    text_data = [tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
    text_data = [d.replace("\n", " ").replace("\r", "").strip() for d in text_data]
    return text_data


def _convert_text_to_tabular_data(
    text: list[str],
    columns: list[str],
) -> pd.DataFrame:
    generated = []

    for t in text:
        parts = t.replace(";", ",").split(",")
        row = dict.fromkeys(columns, "placeholder")

        for part in parts:
            values = part.strip().split(" is ", 1)

            if len(values) != 2:
                continue

            col_name = values[0].strip()
            value = values[1].strip()

            if col_name in columns and row[col_name] == "placeholder":
                row[col_name] = value

        generated.append(row)

    df_gen = pd.DataFrame(generated)
    df_gen.replace("None", None, inplace=True)

    return df_gen


def _encode_row_partial(row, shuffle=True, float_precision=None):
    if not shuffle:
        idx_list = np.arange(len(row.index))
    else:
        idx_list = np.random.permutation(len(row.index))

    parts = []

    for i in idx_list:
        value = row[row.index[i]]

        if pd.isna(value):
            continue

        col_name = row.index[i]

        if isinstance(value, (float, np.floating)) and float_precision is not None:
            formatted = f"{value:.{float_precision}f}".rstrip("0").rstrip(".")
            value = formatted

        parts.append(f"{col_name} is {value}")

    return ", ".join(parts)


def _get_random_missing(row):
    missing_cols = list(row[pd.isna(row)].index)
    return np.random.choice(missing_cols) if len(missing_cols) > 0 else None


def _partial_df_to_promts(partial_df: pd.DataFrame, float_precision=None):
    def encoder(x):
        return _encode_row_partial(x, True, float_precision)

    encoded_rows = list(partial_df.apply(encoder, axis=1))
    first_missing = list(partial_df.apply(_get_random_missing, axis=1))

    prompts = [
        ((enc + ", ") if len(enc) > 0 else "")
        + (fst + " is" if fst is not None else "")
        for enc, fst in zip(encoded_rows, first_missing)
    ]

    return prompts


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


# -------------------------
# Dataset + collator
# -------------------------
class GReaTDataset(Dataset):
    """
    HuggingFace Dataset wrapper for GReaT.

    Each row is converted into text:
    column is value, column is value, ...
    """

    def set_tokenizer(self, tokenizer, float_precision=None):
        self.tokenizer = tokenizer
        self.float_precision = float_precision

    def _format_value(self, value):
        if isinstance(value, (float, np.floating)) and self.float_precision is not None:
            formatted = f"{value:.{self.float_precision}f}"
            return formatted.rstrip("0").rstrip(".")

        return str(value).strip()

    def _getitem(
        self,
        key: int | slice | str,
        decoded: bool = True,
        **kwargs,
    ) -> dict | list:
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        row_text = ", ".join(
            [
                f"{row.column_names[i]} is {self._format_value(row.columns[i].to_pylist()[0])}"
                for i in shuffle_idx
            ]
        )

        tokenized = self.tokenizer(row_text, padding=True)
        return tokenized

    def __getitems__(self, keys: int | slice | str | list):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]

        return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """
    Data collator that also creates labels for causal language modeling.
    """

    def __call__(self, features: list[dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["labels"] = batch["input_ids"].clone()
        return batch


# -------------------------
# Trainer
# -------------------------
def _seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


class GReaTTrainer(Trainer):
    """
    Custom HuggingFace Trainer for GReaT.

    Keeps the full dataset format and uses a seeded DataLoader.
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )


# -------------------------
# Start-token helpers
# -------------------------
def _pad(x, length: int, pad_value=50256):
    return [pad_value] * (length - len(x)) + x


def _pad_tokens(tokens):
    max_length = len(max(tokens, key=len))
    return [_pad(t, max_length) for t in tokens]


class GReaTStart:
    """
    Base class for creating generation start tokens.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_start_tokens(self, n_samples: int) -> list[list[int]]:
        raise NotImplementedError("Subclasses must implement get_start_tokens().")


class CategoricalStart(GReaTStart):
    """
    Start generation from a categorical column distribution.
    """

    def __init__(self, tokenizer, start_col: str, start_col_dist: dict):
        super().__init__(tokenizer)

        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())

    def get_start_tokens(self, n_samples):
        start_values = random.choices(self.population, self.weights, k=n_samples)
        start_text = [f"{self.start_col} is {value}," for value in start_values]
        return _pad_tokens(self.tokenizer(start_text)["input_ids"])


class ContinuousStart(GReaTStart):
    """
    Start generation from a continuous column distribution.
    """

    def __init__(
        self,
        tokenizer,
        start_col: str,
        start_col_dist: list[float],
        noise: float = 0.01,
        decimal_places: int = 5,
    ):
        super().__init__(tokenizer)

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places

    def get_start_tokens(self, n_samples):
        start_values = random.choices(self.start_col_dist, k=n_samples)

        start_text = [
            f"{self.start_col} is {format(value, f'.{self.decimal_places}f')},"
            for value in start_values
        ]

        return _pad_tokens(self.tokenizer(start_text)["input_ids"])


class RandomStart(GReaTStart):
    """
    Start generation from random column names.
    """

    def __init__(self, tokenizer, all_columns: list[str]):
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_tokens(self, n_samples):
        start_columns = random.choices(self.all_columns, k=n_samples)
        start_text = [f"{col} is" for col in start_columns]
        return _pad_tokens(self.tokenizer(start_text)["input_ids"])
