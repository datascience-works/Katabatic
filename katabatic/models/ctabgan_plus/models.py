"""
CTAB-GAN+ implementation for the Katabatic framework.

Based on "CTAB-GAN+: Enhancing Tabular Data Synthesis"
by Zhao et al. (2022) - https://arxiv.org/abs/2204.00401
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import (
    Module, Sequential, Linear, Conv2d, ConvTranspose2d,
    LeakyReLU, ReLU, Dropout, Sigmoid
)
from torch.nn import functional as F
from torch.optim import Adam

from katabatic.models.base_model import Model
from katabatic.models.ctabgan_plus.utils import (
    DataPrep,
    DataTransformer,
    ImageTransformer,
)

logger = logging.getLogger(__name__)


class Classifier(Module):
    """Auxiliary classifier used to support downstream task consistency."""

    def __init__(self, input_dim, dis_dims, st_ed):
        super().__init__()

        dim = input_dim - (st_ed[1] - st_ed[0])
        self.str_end = st_ed
        layers = []

        for hidden_dim in list(dis_dims):
            layers += [Linear(dim, hidden_dim), LeakyReLU(0.2), Dropout(0.5)]
            dim = hidden_dim

        target_dim = st_ed[1] - st_ed[0]

        if target_dim == 1:
            layers += [Linear(dim, 1)]
        elif target_dim == 2:
            layers += [Linear(dim, 1), Sigmoid()]
        else:
            layers += [Linear(dim, target_dim)]

        self.seq = Sequential(*layers)

    def forward(self, input):
        target_dim = self.str_end[1] - self.str_end[0]

        if target_dim == 1:
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            label = torch.argmax(
                input[:, self.str_end[0]:self.str_end[1]],
                axis=-1
            )

        features = torch.cat(
            (input[:, :self.str_end[0]], input[:, self.str_end[1]:]),
            dim=1
        )

        if target_dim in [1, 2]:
            return self.seq(features).view(-1), label

        return self.seq(features), label


def apply_activate(data, output_info):
    """Apply column-wise activation functions to generated data."""

    activated_columns = []
    start = 0

    for dim, activation, *_ in output_info:
        end = start + dim

        if activation == "tanh":
            activated_columns.append(torch.tanh(data[:, start:end]))
        elif activation == "softmax":
            activated_columns.append(F.gumbel_softmax(data[:, start:end], tau=0.2))

        start = end

    return torch.cat(activated_columns, dim=1)


def get_st_ed(target_col_index, output_info):
    """Return the transformed start and end indices of the target column."""

    start = 0
    column_index = 0
    transformed_index = 0

    for item in output_info:
        if column_index == target_col_index:
            break

        start += item[0]

        if item[1] == "softmax" or (item[1] == "tanh" and item[2] == "yes_g"):
            column_index += 1

        transformed_index += 1

    return start, start + output_info[transformed_index][0]


class Discriminator(Module):
    """CNN discriminator for distinguishing real and synthetic records."""

    def __init__(self, layers):
        super().__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    """CNN generator for producing transformed synthetic tabular records."""

    def __init__(self, layers):
        super().__init__()
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)


class CTABGANSynthesizer:
    """Core CTAB-GAN+ synthesizer."""

    def __init__(
        self,
        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
    ):
        self.class_dim = class_dim
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, categorical, mixed, general, non_categorical, type):
        """Fit CTAB-GAN+ on prepared tabular data."""

        problem_type = None
        target_index = None

        if type:
            problem_type = list(type.keys())[0]
            target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(
            train_data=train_data,
            categorical_list=categorical,
            mixed_dict=mixed,
            general_list=general,
            non_categorical_list=non_categorical,
        )
        self.transformer.fit()

        data = self.transformer.transform(train_data.values)
        data = torch.tensor(data, dtype=torch.float32, device=self.device)

        data_dim = self.transformer.output_dim
        side = self._find_side(data_dim)

        self.Gtransformer = ImageTransformer(side)
        self.Dtransformer = ImageTransformer(side)

        self.generator = Generator(self._gen_layers()).to(self.device)
        self.discriminator = Discriminator(self._dis_layers()).to(self.device)

        self.classifier = None
        if target_index is not None:
            self.classifier = Classifier(
                data_dim,
                self.class_dim,
                get_st_ed(target_index, self.transformer.output_info)
            ).to(self.device)

        self.opt_g = Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9),
            weight_decay=self.l2scale,
        )

        discriminator_params = list(self.discriminator.parameters())

        if self.classifier is not None:
            discriminator_params += list(self.classifier.parameters())

        self.opt_d = Adam(
            discriminator_params,
            lr=2e-4,
            betas=(0.5, 0.9),
            weight_decay=self.l2scale,
        )

        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            permutation = torch.randperm(len(data), device=self.device)

            for start in range(0, len(data), self.batch_size):
                real = data[permutation[start:start + self.batch_size]]

                if len(real) == 0:
                    continue

                noise = torch.randn(
                    len(real),
                    self.random_dim,
                    1,
                    1,
                    device=self.device,
                )

                fake = self.generator(noise)
                fake = self.Gtransformer.inverse_transform(fake)
                fake = apply_activate(fake, self.transformer.output_info)

                real_img = self.Dtransformer.transform(real)
                fake_img = self.Dtransformer.transform(fake.detach())

                d_real = self.discriminator(real_img)
                d_fake = self.discriminator(fake_img)

                loss_d = (
                    torch.mean(F.relu(1.0 - d_real)) +
                    torch.mean(F.relu(1.0 + d_fake))
                )

                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

                fake_img = self.Dtransformer.transform(fake)
                g_fake = self.discriminator(fake_img)

                loss_g = -torch.mean(g_fake)

                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()

            logger.info(
                "CTAB-GAN+ epoch %s/%s completed",
                epoch + 1,
                self.epochs
            )

    def sample(self, n):
        """Generate synthetic records from the trained generator."""

        self.generator.eval()
        steps = n // self.batch_size + 1
        generated_batches = []

        with torch.no_grad():
            for _ in range(steps):
                noise = torch.randn(
                    self.batch_size,
                    self.random_dim,
                    1,
                    1,
                    device=self.device,
                )

                fake = self.generator(noise)
                fake = self.Gtransformer.inverse_transform(fake)
                fake = apply_activate(fake, self.transformer.output_info)

                generated_batches.append(fake.detach().cpu().numpy())

        generated_data = np.concatenate(generated_batches, axis=0)
        result, _ = self.transformer.inverse_transform(generated_data)

        return result[:n]

    @staticmethod
    def _find_side(dim):
        """Find the smallest supported square image side."""

        for side in [4, 8, 16, 32, 64]:
            if side * side >= dim:
                return side

        return 64

    def _gen_layers(self):
        """Build generator layers for the selected image side."""

        layers = [
            ConvTranspose2d(self.random_dim, 256, 4, 1, 0),
            ReLU(True),
        ]

        current_size = 4
        current_channels = 256

        while current_size < self.Gtransformer.height:
            next_channels = max(current_channels // 2, 64)

            layers += [
                ConvTranspose2d(current_channels, next_channels, 4, 2, 1),
                ReLU(True),
            ]

            current_channels = next_channels
            current_size *= 2

        layers += [Conv2d(current_channels, 1, 3, 1, 1)]

        return layers

    def _dis_layers(self):
        """Build discriminator layers for the selected image side."""

        reduced_side = self.Dtransformer.height // 4

        return [
            Conv2d(1, 64, 4, 2, 1),
            LeakyReLU(0.2),
            Conv2d(64, 128, 4, 2, 1),
            LeakyReLU(0.2),
            torch.nn.Flatten(),
            Linear(128 * reduced_side * reduced_side, 1),
        ]


class CTABGANPlus(Model):
    """Katabatic-compatible CTAB-GAN+ model wrapper."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        self.config = config or {}

        self.synthesizer = CTABGANSynthesizer(
            class_dim=self.config.get("class_dim", (256, 256, 256, 256)),
            random_dim=self.config.get("random_dim", 100),
            num_channels=self.config.get("num_channels", 64),
            l2scale=self.config.get("l2scale", 1e-5),
            batch_size=self.config.get("batch_size", 500),
            epochs=self.config.get("epochs", 300),
        )

        self.data_prep = None

    def train(
        self,
        dataset_dir: str,
        synthetic_dir: str,
        categorical: Optional[List[int]] = None,
        **kwargs,
    ):
        """Train CTAB-GAN+ and save synthetic data in Katabatic format."""

        logger.info("Training CTAB-GAN+")

        x_path = os.path.join(dataset_dir, "x_train.csv")
        y_path = os.path.join(dataset_dir, "y_train.csv")

        x_train = pd.read_csv(x_path)

        if os.path.exists(y_path):
            y_train = pd.read_csv(y_path)
            target_col = y_train.columns[0]
            train_df = pd.concat([x_train, y_train], axis=1)
            problem_type = {"Classification": target_col}
        else:
            target_col = None
            train_df = x_train.copy()
            problem_type = {}

        if categorical is None:
            raise ValueError(
                "categorical must be provided as a list of column indices "
                "aligned with x_train and y_train."
            )

        if not all(isinstance(index, int) for index in categorical):
            raise TypeError("categorical must contain integer column indices only.")

        max_index = train_df.shape[1] - 1

        if any(index < 0 or index > max_index for index in categorical):
            raise ValueError(f"categorical indices must be between 0 and {max_index}.")

        self.data_prep = DataPrep(
            raw_df=train_df,
            categorical=categorical,
            log=self.config.get("log_columns", []),
            mixed=self.config.get("mixed_columns", {}),
            general=self.config.get("general_columns", []),
            non_categorical=self.config.get("non_categorical_columns", []),
            integer=self.config.get("integer_columns", []),
            type=problem_type,
            test_ratio=None,
        )

        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=problem_type,
        )

        n_samples = len(self.data_prep.df)
        synthetic_data = self.synthesizer.sample(n_samples)
        synthetic_df = self.data_prep.inverse_prep(synthetic_data)

        os.makedirs(synthetic_dir, exist_ok=True)

        if target_col is not None:
            x_synth = synthetic_df.drop(columns=[target_col])
            y_synth = synthetic_df[[target_col]]

            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)
        else:
            synthetic_df.to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"),
                index=False
            )

        logger.info("Synthetic data saved to %s", synthetic_dir)

        return self

    def sample(self, n: int):
        """Generate synthetic samples after training."""

        if self.data_prep is None:
            raise RuntimeError("The model must be trained before calling sample().")

        synthetic_data = self.synthesizer.sample(n)
        return self.data_prep.inverse_prep(synthetic_data)

    def evaluate(self):
        """Evaluation is handled by the Katabatic benchmarking pipeline."""

        raise NotImplementedError(
            "CTAB-GAN+ evaluation is handled by the Katabatic TSTR pipeline."
        )