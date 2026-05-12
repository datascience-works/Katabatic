"""Self-contained TVAE-GAN utilities for the Katabatic 3-file model format.

This file intentionally collates the original helper modules:
- data_handler.py
- data_loader.py
- mlp_network.py
- encoder_critic.py
- decoder_generator.py
- tvaegan_network.py
- tvaegan_synthesizer.py
"""

from __future__ import annotations

import decimal
import random
import sys
import uuid
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

class EnergyDistanceLoss(torch.nn.Module):
    def forward(self, x, y):
        xx = torch.cdist(x, x, p=2)
        yy = torch.cdist(y, y, p=2)
        xy = torch.cdist(x, y, p=2)

        return 2 * xy.mean() - xx.mean() - yy.mean()


class DataHandler:
    def __init__(self, df: pd.DataFrame, forced_cat_columns=None):
        self.org_df: pd.DataFrame = df
        self.forced_cat_columns = set(forced_cat_columns or [])

        self._cat_scaler = None
        self._cat_columns = None
        self._cat_scaled_np = None

        self._num_scaler = None
        self._num_columns = None
        self._num_scaled_np = None

        self.none_guid = str(uuid.uuid4())

    def transform(self) -> np.ndarray:
        self._cat_columns, self._num_columns = self._get_cat_num(
            self.org_df,
            forced_cat_columns=self.forced_cat_columns
        )

        self._cat_scaler, self._cat_scaled_np = self._scale_categoricals(
            self.org_df[self._cat_columns]
        )
        self._num_scaler, self._num_scaled_np = self._scale_numericals(
            self.org_df[self._num_columns]
        )

        if self._cat_scaled_np is not None and self._num_scaled_np is not None:
            scaled_np = np.concatenate([self._cat_scaled_np, self._num_scaled_np], axis=1)
        else:
            scaled_np = self._cat_scaled_np if self._cat_scaled_np is not None else self._num_scaled_np

        if scaled_np is None:
            raise ValueError("TVAEGAN received an empty dataframe or no usable columns.")

        return scaled_np.astype(np.float32)

    def _scale_categoricals(self, cat_df: pd.DataFrame):
        cat_scaled_np = None
        cat_scaler = None

        if len(cat_df.columns) > 0:
            if cat_df.isnull().any(axis=1).any():
                cat_df = cat_df.fillna(self.none_guid)

            cat_scaler = {"onehot": {}, "minmax": MinMaxScaler((-1, 1), clip=True)}

            columns = []
            for column in cat_df.columns:
                cat_scaler["onehot"][column] = LabelBinarizer()
                cat_scaler["onehot"][column].fit(cat_df[column])
                transformed = cat_scaler["onehot"][column].transform(cat_df[column])
                if transformed.ndim == 1:
                    transformed = transformed.reshape(-1, 1)
                columns.append(transformed)

            cat_scaled_np = np.concatenate(columns, axis=1)
            cat_scaler["minmax"].fit(cat_scaled_np)
            cat_scaled_np = cat_scaler["minmax"].transform(cat_scaled_np)

        return cat_scaler, cat_scaled_np

    def _scale_numericals(self, num_df: pd.DataFrame):
        num_scaler = None
        num_scaled_np = None

        if len(num_df.columns) > 0:
            num_scaler = MinMaxScaler((-1, 1), clip=True)
            num_scaler.fit(num_df)
            num_scaled_np = num_scaler.transform(num_df)

        return num_scaler, num_scaled_np

    def get_cat_sizes(self) -> list[int]:
        cat_sizes = []
        if self._cat_columns is not None and len(self._cat_columns) > 0:
            for column in self._cat_scaler["onehot"]:
                size = len(self._cat_scaler["onehot"][column].classes_)
                cat_sizes.append(size if size > 2 else 1)
        return cat_sizes

    def get_num_sizes(self) -> list[int]:
        num_sizes = []
        if self._num_columns is not None and len(self._num_columns) > 0:
            for num_column in self._num_columns:
                num_sizes.append(len(self.org_df[num_column].unique()))
        return num_sizes

    def reverse(self, scaled_np: np.ndarray) -> pd.DataFrame:
        cat_width = sum(self.get_cat_sizes())
        num_width = len(self.get_num_sizes())
        cat_scaled_np, num_scaled_np, _ = np.split(scaled_np, [cat_width, cat_width + num_width], axis=1)

        cat_rev_df = None
        if cat_width > 0:
            cat_rev_np = self._cat_scaler["minmax"].inverse_transform(cat_scaled_np)

            cat_sizes = self.get_cat_sizes()
            split_indices = np.cumsum(cat_sizes)[:-1]
            cat_columns_np = np.split(cat_rev_np, split_indices, axis=1)

            decoded_columns = []
            for column_name, cat_column_np in zip(self._cat_scaler["onehot"], cat_columns_np):
                decoded_columns.append(self._cat_scaler["onehot"][column_name].inverse_transform(cat_column_np))
            cat_columns_rev_np = np.stack(decoded_columns, axis=1)
            cat_rev_df = pd.DataFrame(cat_columns_rev_np, columns=self._cat_columns)
            cat_rev_df = cat_rev_df.replace(self.none_guid, np.nan)

        num_rev_df = None
        if num_width > 0:
            num_rev_df = pd.DataFrame(
                self._num_scaler.inverse_transform(num_scaled_np),
                columns=self._num_columns,
            )

        frames = [frame for frame in [cat_rev_df, num_rev_df] if frame is not None]
        rev_df = pd.concat(frames, axis=1)
        rev_df = rev_df[self.org_df.columns]

        for column, org_dtype in zip(self.org_df.columns, self.org_df.dtypes):
            if org_dtype == np.dtype("float") or np.issubdtype(org_dtype, np.floating):
                decimal_digit = self._get_max_decimal_digit(list(self.org_df[column].dropna()))
                rev_df[column] = rev_df[column].apply(lambda x: round(float(x), decimal_digit) if pd.notna(x) else x)
            if org_dtype == np.dtype("int64") or np.issubdtype(org_dtype, np.integer):
                rev_df[column] = rev_df[column].apply(lambda x: round(float(x)) if pd.notna(x) else x)
            rev_df[column] = rev_df[column].astype(org_dtype)

        return rev_df

    def _get_max_decimal_digit(self, numbers: List[str]) -> int:
        if len(numbers) == 0:
            return 0

        def _get_decimal_digit(number: str):
            d = decimal.Decimal(number)
            return max(d.as_tuple().exponent * (-1), 0)

        return max(_get_decimal_digit(str(number)) for number in numbers)

    @staticmethod
    def _get_cat_num(df: pd.DataFrame, discrete_threshold: int = 20, forced_cat_columns=None):
        forced_cat_columns = set(forced_cat_columns or [])
    
        cat_columns = []
        num_columns = []
    
        for column, dtype in zip(df.columns, df.dtypes):
            unique_values = df[column].nunique(dropna=True)
    
            if column in forced_cat_columns:
                cat_columns.append(column)
    
            elif dtype in [np.dtype("object"), np.dtype("bool")]:
                cat_columns.append(column)
    
            elif unique_values <= discrete_threshold:
                cat_columns.append(column)
    
            else:
                num_columns.append(column)
    
        return cat_columns, num_columns


class DataLoader:
    def __init__(self, *tensors, batch_size: int = 32, shuffle: bool = False):
        self.tensors = tensors
        self.data_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.randperm(self.data_len) if self.shuffle else torch.arange(self.data_len)
        for start_idx in range(0, self.data_len, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.data_len)
            yield tuple(t[indices[start_idx:end_idx]] for t in self.tensors)

    def __len__(self):
        return (self.data_len + self.batch_size - 1) // self.batch_size


class MLPNetwork(torch.nn.Module):
    def __init__(self, in_size: int, hidden_sizes: List[int], out_size: int, dropout: float, device=None):
        super().__init__()

        out_features = hidden_sizes[0] if len(hidden_sizes) > 0 else out_size
        self.input = torch.nn.Sequential(
            torch.nn.Linear(in_size, out_features),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
        ).to(device=device)

        self.hidden = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                    torch.nn.Dropout(dropout),
                    torch.nn.ReLU(),
                )
                for i in range(1, len(hidden_sizes))
            ]
        ).to(device=device)

        in_features = hidden_sizes[-1] if len(hidden_sizes) > 0 else out_size
        self.output = torch.nn.Linear(in_features=in_features, out_features=out_size).to(device=device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input(x)
        for layer in self.hidden:
            x = layer(x)
        return self.output(x)


class EncoderCritic(torch.nn.Module):
    def __init__(self, in_size: int, hidden_dims: List[int], out_size: int, dropout: float, device=None, is_vae=False):
        super().__init__()
        self.device = device
        self.is_vae = is_vae
        self.network = MLPNetwork(
            in_size=in_size,
            hidden_sizes=hidden_dims,
            out_size=out_size,
            device=device,
            dropout=dropout,
        ).to(device=self.device)

        self.mu_fc = torch.nn.Linear(out_size, out_size).to(device=device)
        self.sigma_fc = torch.nn.Linear(out_size, out_size).to(device=device)
        torch.nn.init.xavier_uniform_(self.mu_fc.weight)
        torch.nn.init.xavier_uniform_(self.sigma_fc.weight)
        if self.mu_fc.bias is not None:
            torch.nn.init.zeros_(self.mu_fc.bias)
        if self.sigma_fc.bias is not None:
            torch.nn.init.zeros_(self.sigma_fc.bias)

    def forward(self, x):
        out = self.network(x)
        if self.is_vae:
            mu = self.mu_fc(out)
            sigma = torch.exp(self.sigma_fc(out))
            out = mu + sigma * torch.randn_like(out)
        return out


class DecoderGenerator(torch.nn.Module):
    def __init__(self, in_size: int, hidden_dims: List[int], out_size: int, dropout: float, device=None):
        super().__init__()
        self.device = device
        self.mlp_network = MLPNetwork(
            in_size=in_size,
            hidden_sizes=hidden_dims,
            out_size=out_size,
            dropout=dropout,
            device=device,
        ).to(device=device)
        self.activation = torch.nn.Hardtanh().to(device=device)

    def forward(self, x):
        return self.activation(self.mlp_network(x))


class TVAEGANNetwork:
    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        latent_dim: int,
        w_regularize: float,
        w_reconstruct: float,
        s_generat: float,
        s_encoder: float,
        clip: float,
        dropout: float,
        random_state: int = 42,
    ):
        self._set_seed(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = dropout
        self.latent_dim = latent_dim
        self.w_regularize = w_regularize
        self.w_reconstruct = w_reconstruct
        self.s_generat = s_generat
        self.s_encoder = s_encoder
        self.clip = clip

        self.encoder = EncoderCritic(
            in_size=in_size,
            hidden_dims=hidden_sizes,
            out_size=latent_dim,
            dropout=self.dropout,
            device=self.device,
            is_vae=True,
        ).to(self.device)
        self.generat = DecoderGenerator(
            in_size=latent_dim,
            hidden_dims=hidden_sizes[::-1],
            out_size=in_size,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)
        self.critic = EncoderCritic(
            in_size=in_size,
            hidden_dims=hidden_sizes,
            out_size=1,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)

        self.encoder_optim: Optional[torch.optim.RMSprop] = None
        self.decoder_optim: Optional[torch.optim.RMSprop] = None
        self.generat_optim: Optional[torch.optim.RMSprop] = None
        self.critic_optim: Optional[torch.optim.RMSprop] = None

        self.loss_regularize = EnergyDistanceLoss()
        self.loss_restore = torch.nn.MSELoss()

    def _set_seed(self, seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def train(self, data_loader: DataLoader, epochs: int, lr_encoder: float, lr_critic: float, lr_generat: float):
        self.encoder_optim = torch.optim.RMSprop(self.encoder.parameters(), lr=lr_encoder)
        self.decoder_optim = torch.optim.RMSprop(self.generat.parameters(), lr=lr_encoder)
        self.generat_optim = torch.optim.RMSprop(self.generat.parameters(), lr=lr_generat)
        self.critic_optim = torch.optim.RMSprop(self.critic.parameters(), lr=lr_critic)

        self.encoder.train(True)
        self.generat.train(True)
        self.critic.train(True)
        self._train(epochs, data_loader)
        self.encoder.train(False)
        self.generat.train(False)
        self.critic.train(False)

    def _train(self, num_epochs: int, data_loader: DataLoader):
        for epoch in range(num_epochs):
            encoder_losses = []
            critic_losses = []
            generat_losses = []

            for step, rows in enumerate(data_loader):
                batch_size = rows[0].shape[0]
                x = rows[0].type(torch.FloatTensor).to(self.device)

                if self.s_generat != sys.maxsize:
                    critic_losses.append(self._critic_step(batch_size, x))
                if step % self.s_generat == 0 and self.s_generat != sys.maxsize:
                    generat_losses.append(self._generat_step(batch_size, x))
                if step % self.s_encoder == 0 and self.s_encoder != sys.maxsize:
                    encoder_losses.append(self._encoder_step(x))

            encoder_loss_avg = sum(encoder_losses) / len(encoder_losses) if encoder_losses else 0
            critic_loss_avg = sum(critic_losses) / len(critic_losses) if critic_losses else 0
            generat_loss_avg = sum(generat_losses) / len(generat_losses) if generat_losses else 0
            print({
                "epoch": epoch + 1,
                "encoder_loss_avg": encoder_loss_avg,
                "critic_loss_avg": critic_loss_avg,
                "generat_loss_avg": generat_loss_avg,
            })

    def _encoder_step(self, x):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.generat_optim.zero_grad()
        self.critic_optim.zero_grad()

        x_encoded = self.encoder(x)
        y = self.generat(x_encoded)
        z = torch.randn(x_encoded.shape).to(self.device)
        regularize_loss = self.loss_regularize(x_encoded, z)
        reconstruct_loss = self.loss_restore(y, x)
        loss = (regularize_loss * self.w_regularize) + (reconstruct_loss * self.w_reconstruct)
        loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()
        return loss.item()

    def _generat_step(self, batch_size, x):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.generat_optim.zero_grad()
        self.critic_optim.zero_grad()

        z = torch.randn((batch_size, self.latent_dim)).to(self.device)
        z_out = self.generat(z)
        critic_z_out = self.critic(z_out)
        loss = -torch.mean(critic_z_out)
        loss.backward()
        self.generat_optim.step()
        return loss.item()

    def _critic_step(self, batch_size, x):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.generat_optim.zero_grad()
        self.critic_optim.zero_grad()

        critic_x_out = self.critic(x)
        with torch.no_grad():
            z = torch.randn((batch_size, self.latent_dim)).to(self.device)
            z_out = self.generat(z)
            x_encoded = self.encoder(x)
            y_out = self.generat(x_encoded)

        critic_z_out = self.critic(z_out)
        critic_y_out = self.critic(y_out)
        loss = -2 * torch.mean(critic_x_out) + torch.mean(critic_y_out) + torch.mean(critic_z_out)
        loss.backward()
        self.critic_optim.step()
        for p in self.critic.parameters():
            p.data.clamp_(-self.clip, self.clip)
        return loss.item()

    def generate(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z_tensor = torch.from_numpy(z).type(torch.FloatTensor).to(self.device)
            z_out = self.generat(z_tensor)
            return z_out.detach().cpu().numpy()

class TVAEGANSynthesizer(BaseEstimator):
    def __init__(
        self,
        epochs: int = 700,
        batch_size: int = 500,
        cat_emb_size: int = 25,
        num_emb_size: int = 25,
        w_regularize: float = 1,
        w_reconstruct: float = 10,
        s_generat: int = 5,
        s_encoder: int = 5,
        lr_generat: float = 0.00005,
        lr_critic: float = 0.00005,
        lr_encoder: float = 0.00005,
        clip: float = 0.01,
        dropout: float = 0.1,
        hidden_layers_multipliers: Optional[List[float]] = None,
        shuffle: bool = True,
        random_state: int = 42,
        forced_cat_columns=None,
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_critic = lr_critic
        self.lr_encoder = lr_encoder
        self.lr_generat = lr_generat
        self.cat_emb_size = cat_emb_size
        self.num_emb_size = num_emb_size
        self.w_regularize = w_regularize
        self.w_reconstruct = w_reconstruct
        self.s_encoder = s_encoder
        self.s_generat = s_generat
        self.clip = clip
        self.dropout = dropout
        self.shuffle = shuffle
        self.hidden_layers_multipliers = hidden_layers_multipliers or [1.0, 1.0]
        self.random_state = random_state
        self.forced_cat_columns = forced_cat_columns or []

        self.latent_dim: Optional[int] = None
        self.data_handler: Optional[DataHandler] = None
        self.model: Optional[TVAEGANNetwork] = None

    def fit(self, df: pd.DataFrame):
        self.data_handler = DataHandler(
            df,
            forced_cat_columns=self.forced_cat_columns
        )
        scaled_np = self.data_handler.transform()

        # Lightweight sanity check: confirms transform/reverse round-trip can run.
        _ = self.data_handler.reverse(deepcopy(scaled_np[: min(len(scaled_np), 5)]))

        cat_sizes = self.data_handler.get_cat_sizes()
        num_sizes = self.data_handler.get_num_sizes()
        cat_size_sum = sum(self.cat_emb_size for _ in cat_sizes)
        num_size_sum = sum(min(self.num_emb_size, num_size) for num_size in num_sizes)
        vec_total_length = max(cat_size_sum + num_size_sum, len(df.columns), 1)

        in_size = sum(cat_sizes) + len(num_sizes)
        hidden_layers_sizes = [
            max(1, int(vec_total_length * hidden_layer_multiplier))
            for hidden_layer_multiplier in self.hidden_layers_multipliers
        ]

        self.latent_dim = len(df.columns)
        self.model = TVAEGANNetwork(
            in_size=in_size,
            hidden_sizes=hidden_layers_sizes,
            latent_dim=self.latent_dim,
            w_regularize=self.w_regularize,
            w_reconstruct=self.w_reconstruct,
            s_generat=self.s_generat,
            s_encoder=self.s_encoder,
            clip=self.clip,
            dropout=self.dropout,
            random_state=self.random_state,
        )

        torch_ds = torch.from_numpy(scaled_np.astype(np.float32))
        data_loader = DataLoader(torch_ds, batch_size=self.batch_size, shuffle=self.shuffle)
        self.model.train(
            data_loader=data_loader,
            epochs=self.epochs,
            lr_encoder=self.lr_encoder,
            lr_critic=self.lr_critic,
            lr_generat=self.lr_generat,
        )
        return self

    def predict(self, samples: int, seed: int = None) -> pd.DataFrame:
        if self.model is None or self.data_handler is None or self.latent_dim is None:
            raise RuntimeError("TVAEGANSynthesizer must be fit before predict().")
        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        X = rng.standard_normal((samples, self.latent_dim)).astype(np.float32)
        Z = self.model.generate(X)
        return self.data_handler.reverse(Z)
