"""
Utility classes for CTAB-GAN+ preprocessing and data transformation.
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection, preprocessing
from sklearn.mixture import BayesianGaussianMixture


class DataPrep:
    """Prepare raw tabular data before CTAB-GAN+ transformation."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        categorical: list,
        log: list,
        mixed: dict,
        general: list,
        non_categorical: list,
        integer: list,
        type: dict,
        test_ratio: Optional[float],
    ):
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.general_columns = general
        self.non_categorical_columns = non_categorical
        self.integer_columns = integer

        self.column_types = {
            "categorical": [],
            "mixed": {},
            "general": [],
            "non_categorical": [],
        }

        self.lower_bounds = {}
        self.label_encoder_list = []

        problem = list(type.keys())[0] if type else None
        target_col = list(type.values())[0] if type else None

        if problem and test_ratio is not None:
            y_real = raw_df[target_col]
            x_real = raw_df.drop(columns=[target_col])

            if problem == "Classification":
                x_train, _, y_train, _ = model_selection.train_test_split(
                    x_real,
                    y_real,
                    test_size=test_ratio,
                    stratify=y_real,
                    random_state=42,
                )
            else:
                x_train, _, y_train, _ = model_selection.train_test_split(
                    x_real,
                    y_real,
                    test_size=test_ratio,
                    random_state=42,
                )

            x_train[target_col] = y_train
            self.df = x_train
        else:
            self.df = raw_df.copy()

        self.df = self.df.replace(r" ", np.nan)
        self.df = self.df.fillna("empty")

        all_columns = set(self.df.columns)
        categorical_names = {
            self.df.columns[i]
            for i in self.categorical_columns
            if isinstance(i, int) and i < len(self.df.columns)
        }

        relevant_missing_columns = list(all_columns - categorical_names)

        for col in relevant_missing_columns:
            if "empty" not in self.df[col].values:
                continue

            self.df[col] = self.df[col].apply(
                lambda x: -9999999 if x == "empty" else x
            )

            if col in self.log_columns:
                self.mixed_columns[col] = [-9999999]
            elif col in self.mixed_columns:
                self.mixed_columns[col].append(-9999999)
            else:
                self.mixed_columns[col] = [-9999999]

        if self.log_columns:
            for log_column in self.log_columns:
                valid_indices = [
                    idx
                    for idx, value in enumerate(self.df[log_column].values)
                    if value != -9999999
                ]

                eps = 1
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower

                if lower > 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x) if x != -9999999 else -9999999
                    )
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x + eps) if x != -9999999 else -9999999
                    )
                else:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x - lower + eps)
                        if x != -9999999
                        else -9999999
                    )

        for column_index, column in enumerate(self.df.columns):
            if column_index in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)

                label_encoder.fit(self.df[column])
                self.df[column] = label_encoder.transform(self.df[column])

                self.label_encoder_list.append({
                    "column": column,
                    "label_encoder": label_encoder,
                })

                self.column_types["categorical"].append(column_index)

                if column_index in self.general_columns:
                    self.column_types["general"].append(column_index)

                if column_index in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)

            elif column_index in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[
                    column_index
                ]

            elif column_index in self.general_columns:
                self.column_types["general"].append(column_index)

    def inverse_prep(self, data, eps=1):
        """Reverse preprocessing and return generated data as a DataFrame."""

        df_sample = pd.DataFrame(data, columns=self.df.columns)

        for item in self.label_encoder_list:
            label_encoder = item["label_encoder"]
            column = item["column"]

            df_sample[column] = df_sample[column].astype(int)
            df_sample[column] = label_encoder.inverse_transform(df_sample[column])

        if self.log_columns:
            for column in self.log_columns:
                lower = self.lower_bounds[column]

                if lower > 0:
                    df_sample[column] = df_sample[column].apply(lambda x: np.exp(x))
                elif lower == 0:
                    df_sample[column] = df_sample[column].apply(
                        lambda x: np.ceil(np.exp(x) - eps)
                        if (np.exp(x) - eps) < 0
                        else (np.exp(x) - eps)
                    )
                else:
                    df_sample[column] = df_sample[column].apply(
                        lambda x: np.exp(x) - eps + lower
                    )

        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column] = np.round(df_sample[column].values).astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace("empty", np.nan, inplace=True)

        return df_sample


class DataTransformer:
    """Transform tabular records into CTAB-GAN+ numerical representation."""

    def __init__(
        self,
        train_data=None,
        categorical_list=None,
        mixed_dict=None,
        general_list=None,
        non_categorical_list=None,
        n_clusters=10,
        eps=0.005,
    ):
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps

        self.train_data = train_data
        self.categorical_columns = categorical_list or []
        self.mixed_columns = mixed_dict or {}
        self.general_columns = general_list or []
        self.non_categorical_columns = non_categorical_list or []

    def get_metadata(self):
        """Create column metadata used during transformation."""

        meta = []

        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:, index]

            if index in self.categorical_columns:
                if index in self.non_categorical_columns:
                    meta.append({
                        "name": index,
                        "type": "continuous",
                        "min": column.min(),
                        "max": column.max(),
                    })
                else:
                    mapper = column.value_counts().index.tolist()

                    if len(mapper) == 0:
                        mapper = [0]

                    meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper,
                    })

            elif index in self.mixed_columns:
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index],
                })

            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    def fit(self):
        """Fit Gaussian mixture models and define output dimensions."""

        data = self.train_data.values
        self.meta = self.get_metadata()

        self.model = []
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []

        for column_id, info in enumerate(self.meta):
            if info["type"] == "continuous":
                if column_id not in self.general_columns:
                    gm = BayesianGaussianMixture(
                        n_components=self.n_clusters,
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=0.001,
                        max_iter=100,
                        n_init=1,
                        random_state=42,
                    )

                    gm.fit(data[:, column_id].reshape(-1, 1))

                    mode_freq = (
                        pd.Series(gm.predict(data[:, column_id].reshape(-1, 1)))
                        .value_counts()
                        .keys()
                    )

                    old_components = gm.weights_ > self.eps
                    components = [
                        True if (i in mode_freq) and old_components[i] else False
                        for i in range(self.n_clusters)
                    ]

                    if not any(components):
                        components[np.argmax(gm.weights_)] = True

                    self.model.append(gm)
                    self.components.append(components)
                    self.output_info += [
                        (1, "tanh", "no_g"),
                        (np.sum(components), "softmax"),
                    ]
                    self.output_dim += 1 + np.sum(components)

                else:
                    self.model.append(None)
                    self.components.append(None)
                    self.output_info += [(1, "tanh", "yes_g")]
                    self.output_dim += 1

            elif info["type"] == "mixed":
                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )

                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )

                gm1.fit(data[:, column_id].reshape(-1, 1))

                filter_arr = [value not in info["modal"] for value in data[:, column_id]]
                self.filter_arr.append(filter_arr)

                gm2.fit(data[:, column_id][filter_arr].reshape(-1, 1))

                mode_freq = (
                    pd.Series(
                        gm2.predict(data[:, column_id][filter_arr].reshape(-1, 1))
                    )
                    .value_counts()
                    .keys()
                )

                old_components = gm2.weights_ > self.eps
                components = [
                    True if (i in mode_freq) and old_components[i] else False
                    for i in range(self.n_clusters)
                ]

                if not any(components):
                    components[np.argmax(gm2.weights_)] = True

                self.model.append((gm1, gm2))
                self.components.append(components)

                self.output_info += [
                    (1, "tanh", "no_g"),
                    (np.sum(components) + len(info["modal"]), "softmax"),
                ]
                self.output_dim += 1 + np.sum(components) + len(info["modal"])

            else:
                self.model.append(None)
                self.components.append(None)
                self.output_info += [(max(info["size"], 1), "softmax")]
                self.output_dim += max(info["size"], 1)

    def transform(self, data):
        """Transform raw records into the encoded CTAB-GAN+ feature space."""

        values = []
        mixed_counter = 0

        for column_id, info in enumerate(self.meta):
            current = data[:, column_id]

            if info["type"] == "continuous":
                if column_id not in self.general_columns:
                    current = current.reshape(-1, 1)

                    means = self.model[column_id].means_.reshape(1, -1)
                    stds = np.sqrt(self.model[column_id].covariances_).reshape(1, -1)

                    features = (current - means) / (4 * stds)
                    probs = self.model[column_id].predict_proba(current)

                    features = features[:, self.components[column_id]]
                    probs = probs[:, self.components[column_id]]

                    selected = np.array([
                        np.random.choice(len(prob), p=(prob + 1e-6) / np.sum(prob + 1e-6))
                        for prob in probs
                    ])

                    features = features[
                        np.arange(len(features)),
                        selected
                    ].reshape(-1, 1)

                    features = np.clip(features, -0.99, 0.99)

                    onehot = np.zeros_like(probs)
                    onehot[np.arange(len(probs)), selected] = 1

                    column_sum = onehot.sum(axis=0)
                    order = np.argsort(-column_sum)

                    self.ordering.append(order)
                    values += [features, onehot[:, order]]

                else:
                    self.ordering.append(None)

                    current = (current - info["min"]) / (info["max"] - info["min"])
                    current = current * 2 - 1

                    values.append(current.reshape(-1, 1))

            elif info["type"] == "mixed":
                current = current.reshape(-1, 1)
                filter_arr = self.filter_arr[mixed_counter]
                current_filtered = current[filter_arr]

                means = self.model[column_id][1].means_.reshape(1, -1)
                stds = np.sqrt(self.model[column_id][1].covariances_).reshape(1, -1)

                features = (current_filtered - means) / (4 * stds)
                probs = self.model[column_id][1].predict_proba(current_filtered)

                features = features[:, self.components[column_id]]
                probs = probs[:, self.components[column_id]]

                selected = np.array([
                    np.random.choice(len(prob), p=(prob + 1e-6) / np.sum(prob + 1e-6))
                    for prob in probs
                ])

                features = features[
                    np.arange(len(features)),
                    selected
                ].reshape(-1, 1)

                features = np.clip(features, -0.99, 0.99)

                onehot = np.zeros_like(probs)
                onehot[np.arange(len(probs)), selected] = 1

                n_modal = len(info["modal"])
                final = np.zeros((len(data), 1 + onehot.shape[1] + n_modal))
                filtered_index = 0

                for row_index in range(len(data)):
                    if not filter_arr[row_index]:
                        # Set the indicator for which modal value this row holds
                        modal_val = current[row_index, 0]
                        try:
                            modal_idx = info["modal"].index(modal_val)
                        except ValueError:
                            modal_idx = 0
                        final[row_index, 1 + onehot.shape[1] + modal_idx] = 1
                    else:
                        final[row_index, 0] = features[filtered_index]
                        final[row_index, 1:1 + onehot.shape[1]] = onehot[filtered_index]
                        filtered_index += 1

                column_sum = final[:, 1:].sum(axis=0)
                order = np.argsort(-column_sum)

                self.ordering.append(order)
                final[:, 1:] = final[:, 1:][:, order]

                values.append(final)
                mixed_counter += 1

            else:
                self.ordering.append(None)

                if info["size"] <= 1:
                    column = np.ones((len(data), 1))
                else:
                    column = np.zeros((len(data), info["size"]))
                    index = list(map(info["i2s"].index, current))
                    column[np.arange(len(data)), index] = 1

                values.append(column)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        """Convert transformed CTAB-GAN+ output back to tabular values."""

        data_t = np.zeros((len(data), len(self.meta)))
        start = 0

        for column_id, info in enumerate(self.meta):
            if info["type"] == "continuous":
                if column_id not in self.general_columns:
                    value = data[:, start]
                    mode = data[
                        :,
                        start + 1:start + 1 + np.sum(self.components[column_id])
                    ]

                    order = self.ordering[column_id]
                    mode = mode[:, np.argsort(order)]

                    means = self.model[column_id].means_.reshape(-1)
                    stds = np.sqrt(self.model[column_id].covariances_).reshape(-1)

                    selected_mode = np.argmax(mode, axis=1)
                    data_t[:, column_id] = (
                        value * 4 * stds[selected_mode] + means[selected_mode]
                    )

                    start += 1 + np.sum(self.components[column_id])

                else:
                    value = (data[:, start] + 1) / 2
                    data_t[:, column_id] = (
                        value * (info["max"] - info["min"]) + info["min"]
                    )

                    start += 1

            elif info["type"] == "mixed":
                n_components = np.sum(self.components[column_id])
                n_modal = len(info["modal"])

                value = data[:, start]
                mode_part = data[:, start + 1:start + 1 + n_components]
                modal_part = data[:, start + 1 + n_components:start + 1 + n_components + n_modal]

                order = self.ordering[column_id]
                mode_part = mode_part[:, np.argsort(order)]

                means = self.model[column_id][1].means_.reshape(-1)
                stds = np.sqrt(self.model[column_id][1].covariances_).reshape(-1)

                # Pick the dominant mode across continuous components and modal indicators
                all_modes = np.concatenate([mode_part, modal_part], axis=1)
                selected = np.argmax(all_modes, axis=1)

                for row_idx in range(len(data)):
                    mode_idx = selected[row_idx]
                    if mode_idx >= n_components:
                        data_t[row_idx, column_id] = info["modal"][mode_idx - n_components]
                    else:
                        data_t[row_idx, column_id] = (
                            value[row_idx] * 4 * stds[mode_idx] + means[mode_idx]
                        )

                start += 1 + n_components + n_modal

            elif info["type"] == "categorical":
                column = data[:, start:start + info["size"]]
                index = np.argmax(column, axis=1)

                data_t[:, column_id] = [info["i2s"][i] for i in index]

                start += info["size"]

        return data_t, 0


class ImageTransformer:
    """Reshape tabular vectors into square image tensors for CNN training."""

    def __init__(self, side):
        self.height = side

    def transform(self, data):
        """Convert flat vectors into square image tensors."""

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        batch_size = data.size(0)
        target_dim = self.height * self.height
        current_dim = data.size(1)

        if current_dim < target_dim:
            padding = torch.zeros(
                batch_size,
                target_dim - current_dim,
                device=data.device,
            )
            data = torch.cat([data, padding], dim=1)

        elif current_dim > target_dim:
            data = data[:, :target_dim]

        return data.view(batch_size, 1, self.height, self.height)

    def inverse_transform(self, data):
        """Convert square image tensors back into flat vectors."""

        if data.dim() == 4:
            data = data.view(data.size(0), -1)

        batch_size = data.size(0)
        target_dim = self.height * self.height
        current_dim = data.size(1)

        if current_dim < target_dim:
            padding = torch.zeros(
                batch_size,
                target_dim - current_dim,
                device=data.device,
            )
            data = torch.cat([data, padding], dim=1)

        elif current_dim > target_dim:
            data = data[:, :target_dim]

        return data