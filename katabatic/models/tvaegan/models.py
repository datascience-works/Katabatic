"""Katabatic wrapper for TVAE-GAN, rebuilt to match Larsen et al. (2016)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

from katabatic.models.base_model import Model

from .utils import (
    DecoderGenerator,
    Discriminator,
    Encoder,
    TabularPreprocessor,
    TVAEGANConfig,
    build_networks,
    train_vaegan,
)

if TYPE_CHECKING:
    from katabatic.artifacts.base import ArtifactStore
    from katabatic.artifacts.refs import ModelRef

logger = logging.getLogger(__name__)


class TVAEGANModel(Model):
    """
    Katabatic wrapper for TVAE-GAN.

    """

    ARTIFACT_STATE_FILES = ("tvaegan_state.pt",)

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        discriminator_hidden_dims: list[int] | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 1.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = TVAEGANConfig(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims or [128, 64],
            discriminator_hidden_dims=discriminator_hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            seed=seed,
        )

        self.preprocessor: TabularPreprocessor | None = None
        self.encoder: Encoder | None = None
        self.decoder_generator: DecoderGenerator | None = None
        self.discriminator: Discriminator | None = None
        self.y_col_name_: str | None = None
        self.data_dim_: int | None = None
        self.is_fitted = False

    def train(
        self, dataset_dir: str | Path, synthetic_dir: str | Path, **kwargs
    ) -> TVAEGANModel:
        dataset_dir = Path(dataset_dir)
        synthetic_dir = Path(synthetic_dir)
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        x_train_path = dataset_dir / "x_train.csv"
        y_train_path = dataset_dir / "y_train.csv"
        if not x_train_path.exists():
            raise FileNotFoundError(f"x_train.csv not found at {x_train_path}")
        if not y_train_path.exists():
            raise FileNotFoundError(f"y_train.csv not found at {y_train_path}")

        x_train = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)
        self.y_col_name_ = y_train_df.columns[0]

        df_train = x_train.copy()
        df_train[self.y_col_name_] = y_train_df[self.y_col_name_].values

        logger.info("Training TVAEGANModel on shape %s", df_train.shape)

        self.preprocessor = TabularPreprocessor.infer_and_fit(
            df_train, force_categorical=[self.y_col_name_]
        )
        data = self.preprocessor.encode(df_train)
        self.data_dim_ = data.shape[1]

        self.encoder, self.decoder_generator, self.discriminator = train_vaegan(
            data, self.config
        )
        self.is_fitted = True

        df_synth = self.sample(len(df_train))
        x_synth = df_synth.drop(columns=[self.y_col_name_])
        y_synth = df_synth[[self.y_col_name_]]

        x_synth.to_csv(synthetic_dir / "x_synth.csv", index=False)
        y_synth.to_csv(synthetic_dir / "y_synth.csv", index=False)
        df_synth.to_csv(synthetic_dir / "synthetic.csv", index=False)

        self._maybe_save_artifact_state(kwargs.get("artifact_state_dir"))
        return self

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        """
        Persist fitted state so load_from_ref() can rebuild the model.

        Networks are stored as state_dicts and rebuilt through the same
        build_networks() helper train_vaegan() uses, so the reloaded
        architecture always matches the trained one.
        """
        state_dir = Path(artifact_state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": asdict(self.config),
            "data_dim": self.data_dim_,
            "y_col_name": self.y_col_name_,
            "preprocessor": self.preprocessor,
            "encoder": self.encoder.state_dict(),
            "decoder_generator": self.decoder_generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }
        torch.save(payload, state_dir / self.ARTIFACT_STATE_FILES[0])

    @classmethod
    def load_from_ref(cls, store: ArtifactStore, ref: ModelRef) -> TVAEGANModel:
        """Rehydrate a trained TVAE-GAN from a versioned artifact."""
        state_path = cls._require_state_file(store, ref)

        # weights_only defaults to True in torch >= 2.6, but the payload also
        # carries the fitted preprocessor (sklearn encoder and scaler).
        payload = torch.load(state_path, map_location="cpu", weights_only=False)  # nosec B614: our own artifact

        instance = cls(**payload["config"])
        instance.data_dim_ = payload["data_dim"]
        instance.y_col_name_ = payload["y_col_name"]
        instance.preprocessor = payload["preprocessor"]

        encoder, decoder_generator, discriminator = build_networks(
            instance.data_dim_, instance.config
        )
        encoder.load_state_dict(payload["encoder"])
        decoder_generator.load_state_dict(payload["decoder_generator"])
        discriminator.load_state_dict(payload["discriminator"])

        instance.encoder = encoder.eval()
        instance.decoder_generator = decoder_generator.eval()
        instance.discriminator = discriminator.eval()
        instance.is_fitted = True
        return instance

    def sample(self, n: int, seed: int | None = None, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("TVAEGANModel must be trained before calling sample().")

        device = next(self.decoder_generator.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            z = torch.randn(n, self.config.latent_dim, device=device)
            x_gen = self.decoder_generator(z).cpu().numpy()

        return self.preprocessor.decode(x_gen)

    def evaluate(self, *, data_dir: str, split: str = "test", **kwargs) -> float:
        """
        Returns a reconstruction loss (lower is better), consistent with
        other Katabatic models. Uses the discriminator-feature reconstruction
        loss from training (paper Eq. 6-7), applied to the given split.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TVAEGANModel must be trained before calling evaluate()."
            )

        from pathlib import Path

        import pandas as pd

        from .utils import discriminator_feature_loss, reparameterize

        data_dir = Path(data_dir)
        x_path = data_dir / f"x_{split}.csv"
        y_path = data_dir / f"y_{split}.csv"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Expected x_{split}.csv and y_{split}.csv in {data_dir}"
            )

        x_df = pd.read_csv(x_path)
        y_df = pd.read_csv(y_path)
        df = x_df.copy()
        df[self.y_col_name_] = y_df[self.y_col_name_].values

        arr = self.preprocessor.encode(df)
        device = next(self.encoder.parameters()).device
        x = torch.from_numpy(arr).float().to(device)

        with torch.no_grad():
            mu, logsigma = self.encoder(x)
            z = reparameterize(mu, logsigma)
            x_recon = self.decoder_generator(z)
            loss = discriminator_feature_loss(self.discriminator, x, x_recon)

        return loss.item()
