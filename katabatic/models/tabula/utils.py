import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    Trainer,
)

from datasets import Dataset

# ============================================================
# TABULA DATAFRAME / TEXT UTILITIES
# ============================================================


def _array_to_dataframe(
    data,
    columns=None,
):
    if isinstance(
        data,
        pd.DataFrame,
    ):
        return data

    assert isinstance(
        data,
        np.ndarray,
    )

    assert columns

    assert len(columns) == len(data[0])

    return pd.DataFrame(
        data=data,
        columns=columns,
    )


def _get_column_distribution(
    df,
    col,
):
    if pd.api.types.is_float_dtype(df[col]):
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()

    return col_dist


def _convert_tokens_to_text(
    tokens,
    tokenizer,
):
    text_data = [tokenizer.decode(t) for t in tokens]

    text_data = [
        d.replace(
            "<|endoftext|>",
            "",
        )
        for d in text_data
    ]

    text_data = [
        d.replace(
            "\n",
            " ",
        )
        for d in text_data
    ]

    text_data = [
        d.replace(
            "\r",
            "",
        )
        for d in text_data
    ]

    return text_data


def _convert_text_to_tabular_data(
    text,
    df_gen,
):
    columns = df_gen.columns.to_list()

    result_list = []

    for t in text:
        features = t.split(",")

        td = dict.fromkeys(columns)

        for f in features:
            values = f.strip().split(" ")

            if len(values) == 0:
                continue

            if values[0] in columns and not td[values[0]]:
                try:
                    td[values[0]] = [values[1]]

                except IndexError:
                    pass

        result_list.append(pd.DataFrame(td))

    if not result_list:
        return df_gen

    generated_df = pd.concat(
        result_list,
        ignore_index=True,
        axis=0,
    )

    df_gen = pd.concat(
        [
            df_gen,
            generated_df,
        ],
        ignore_index=True,
        axis=0,
    )

    return df_gen


# ============================================================
# TABULA START SAMPLERS
# ============================================================


def _pad_left(
    x: list[int],
    length: int,
    pad_value: int,
) -> list[int]:

    if len(x) >= length:
        return x

    return [pad_value] * (length - len(x)) + x


def _pad_tokens(
    tokens: list[list[int]],
    pad_value: int,
) -> list[list[int]]:

    max_length = len(
        max(
            tokens,
            key=len,
        )
    )

    return [
        _pad_left(
            t,
            max_length,
            pad_value,
        )
        for t in tokens
    ]


class TabulaStart:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def get_start_tokens(
        self,
        n_samples,
    ):
        raise NotImplementedError


class CategoricalStart(TabulaStart):
    def __init__(
        self,
        tokenizer,
        start_col,
        start_col_dist,
    ):
        super().__init__(tokenizer)

        self.start_col = start_col

        self.population = list(start_col_dist.keys())

        self.weights = list(start_col_dist.values())

    def get_start_tokens(
        self,
        n_samples,
    ):

        start_words = random.choices(
            self.population,
            self.weights,
            k=n_samples,
        )

        start_text = [f"{self.start_col} {s}," for s in start_words]

        ids = self.tokenizer(start_text)["input_ids"]

        return _pad_tokens(
            ids,
            int(self.tokenizer.eos_token_id),
        )


class ContinuousStart(TabulaStart):
    def __init__(
        self,
        tokenizer,
        start_col,
        start_col_dist,
        decimal_places=2,
    ):
        super().__init__(tokenizer)

        self.start_col = start_col

        self.start_col_dist = start_col_dist

        self.decimal_places = decimal_places

    def get_start_tokens(
        self,
        n_samples,
    ):

        start_words = random.choices(
            self.start_col_dist,
            k=n_samples,
        )

        start_text = [
            (f"{self.start_col} {format(float(s), f'.{self.decimal_places}f')},")
            for s in start_words
        ]

        ids = self.tokenizer(start_text)["input_ids"]

        return _pad_tokens(
            ids,
            int(self.tokenizer.eos_token_id),
        )


class RandomStart(TabulaStart):
    def __init__(
        self,
        tokenizer,
        all_columns,
    ):
        super().__init__(tokenizer)

        self.all_columns = all_columns

    def get_start_tokens(
        self,
        n_samples,
    ):

        start_words = random.choices(
            self.all_columns,
            k=n_samples,
        )

        start_text = [f"{s} " for s in start_words]

        ids = self.tokenizer(start_text)["input_ids"]

        return _pad_tokens(
            ids,
            int(self.tokenizer.eos_token_id),
        )


# ============================================================
# TABULA DATASET
# ============================================================


class TabulaDataset(Dataset):
    def set_tokenizer(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def _getitem(
        self,
        key,
        decoded=True,
        **kwargs,
    ):

        row = self._data.fast_slice(
            key,
            1,
        )

        shuffle_idx = list(range(row.num_columns))

        random.shuffle(shuffle_idx)

        shuffled_text = ", ".join(
            [
                f"{row.column_names[i]} {str(row.columns[i]).strip()}"
                for i in shuffle_idx
            ]
        )

        tokenized_text = self.tokenizer(shuffled_text)

        return tokenized_text

    def __getitems__(
        self,
        keys,
    ):

        if isinstance(
            keys,
            list,
        ):
            return [self._getitem(key) for key in keys]

        return self._getitem(keys)


@dataclass
class TabulaDataCollator(DataCollatorWithPadding):
    def __call__(
        self,
        features,
    ):

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=(self.pad_to_multiple_of),
            return_tensors=(self.return_tensors),
        )

        batch["labels"] = batch["input_ids"].clone()

        return batch


# ============================================================
# TABULA TRAINER
# ============================================================


def _seed_worker(_):

    worker_seed = torch.initial_seed() % 2**32

    random.seed(worker_seed)

    np.random.seed(worker_seed)

    torch.manual_seed(worker_seed)

    torch.cuda.manual_seed_all(worker_seed)


class TabulaTrainer(Trainer):
    def get_train_dataloader(
        self,
    ):

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator

        train_dataset = self.train_dataset

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=(self._train_batch_size),
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=(self.args.dataloader_drop_last),
            num_workers=(self.args.dataloader_num_workers),
            pin_memory=(self.args.dataloader_pin_memory),
            worker_init_fn=_seed_worker,
        )
