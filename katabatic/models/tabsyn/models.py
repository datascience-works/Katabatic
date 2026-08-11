from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
import os

# Base Katabatic model interface
from katabatic.models.base_model import Model as BaseModel

# TabSyn utilities and training pipeline
from .utils import (
    TabSynConfig,
    TabSynState,
    train_tabsyn,
    evaluate_tabsyn,
    sample_tabsyn,
)


class TabSyn(BaseModel):
    """
    TabSyn synthetic tabular data generator.

    This implementation learns a latent representation of tabular data
    using an encoder-decoder architecture and applies a diffusion-based
    denoising process within the latent space to generate synthetic samples.

    The model is designed to integrate with the Katabatic benchmarking
    framework and follows the standard train / sample / evaluate workflow.
    """

    def __init__(
        self,
        *,
        # Latent representation settings
        d_token: int = 16,

        # Decoder training configuration
        decoder_epochs: int = 50,
        decoder_batch_size: int = 2048,

        # Diffusion model training configuration
        diffusion_epochs: int = 500,
        diffusion_batch_size: int = 4096,

        # Diffusion sampling configuration
        diffusion_steps: int = 50,

        # Optimisation settings
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 20,

        # Reproducibility and device configuration
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise the TabSyn model configuration.
        """

        super().__init__()

        self.config = TabSynConfig(
            d_token=d_token,
            decoder_epochs=decoder_epochs,
            decoder_batch_size=decoder_batch_size,
            diffusion_epochs=diffusion_epochs,
            diffusion_batch_size=diffusion_batch_size,
            diffusion_steps=diffusion_steps,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed,
            device=device,
        )

        # Model state is populated after successful training
        self.state: Optional[TabSynState] = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """
        Return external runtime dependencies required for this model.
        """

        return ["torch", "tqdm"]

    def train(
        self,
        data_dir: str,
        save_dir: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> "TabSyn":
        """
        Train the TabSyn model and generate synthetic outputs
        compatible with the Katabatic TSTR pipeline.
        """

        # Verify required dependencies
        self.check_dependencies()

        # Train latent encoder-decoder and diffusion model
        self.state = train_tabsyn(
            data_dir=data_dir,
            cfg=self.config,
            save_dir=save_dir,
            extra_info=extra_info or {},
        )

        self.is_fitted = True

        # Determine synthetic output directory
        synth_dir = kwargs.get("synthetic_dir")

        if not synth_dir or not isinstance(synth_dir, str):
            dataset_name = os.path.basename(
                os.path.normpath(data_dir)
            ) or "dataset"

            synth_dir = os.path.join(
                "synthetic",
                dataset_name,
                "tabsyn"
            )

        os.makedirs(synth_dir, exist_ok=True)

        # Generate synthetic samples
        df_s = self.sample(
            n_samples=None,
            return_df=True
        )

        state = self.state

        n_num = state.n_num
        n_cat = len(state.cat_sizes)

        # TSTR requires at least one categorical target column
        if n_cat == 0:
            raise ValueError(
                "TSTR evaluation requires a categorical target column, "
                "but no categorical columns were detected."
            )

        # Temporary synthetic column naming
        num_cols = [f"num_{i}" for i in range(n_num)]
        cat_cols = [f"cat_{i}" for i in range(n_cat)]

        # First categorical column is treated as target label
        y_col = cat_cols[0]

        # Remaining columns are feature inputs
        X_cols = [col for col in df_s.columns if col != y_col]

        x_synth = df_s[X_cols]

        # Preserve original label distribution
        real_y_path = os.path.join(data_dir, "y_train.csv")

        real_y = (
            pd.read_csv(real_y_path)
            .iloc[:, 0]
            .astype(str)
            .to_numpy()
        )

        classes, counts = np.unique(real_y, return_counts=True)

        probs = counts / counts.sum()

        y_synth = np.random.choice(
            classes,
            size=len(x_synth),
            p=probs
        )

        y_synth = pd.Series(
            y_synth,
            name=y_col
        )

        # Align synthetic feature names with real dataset
        real_x_train_path = os.path.join(
            data_dir,
            "x_train.csv"
        )

        try:
            real_cols = (
                pd.read_csv(real_x_train_path, nrows=0)
                .columns
                .tolist()
            )

            if len(real_cols) == x_synth.shape[1]:

                # Rename synthetic columns
                x_synth.columns = real_cols

                # Reorder columns for consistency
                x_synth = x_synth.reindex(columns=real_cols)

            else:
                print(
                    "[TabSyn] Warning: feature count mismatch "
                    f"(synthetic={x_synth.shape[1]}, "
                    f"real={len(real_cols)}). "
                    "Synthetic column names were left unchanged."
                )

        except Exception as e:
            print(
                "[TabSyn] Warning: unable to align synthetic "
                f"feature names using {real_x_train_path}: {e}"
            )

        # Save synthetic datasets
        x_path = os.path.join(synth_dir, "x_synth.csv")
        y_path = os.path.join(synth_dir, "y_synth.csv")

        x_synth.to_csv(x_path, index=False)
        y_synth.to_csv(y_path, index=False, header=True)

        print(
            "[TabSyn] Synthetic data generated successfully:\n"
            f"  Features -> {x_path}\n"
            f"  Labels   -> {y_path}"
        )

        return self

    def evaluate(
        self,
        *,
        data_dir: str,
        split: str = "test",
    ) -> float:
        """
        Evaluate the trained TabSyn model on a dataset split.
        """

        if not self.is_fitted or self.state is None:
            raise RuntimeError(
                "The model must be trained before evaluation."
            )

        return evaluate_tabsyn(
            self.state,
            data_dir=data_dir,
            split=split
        )

    def sample(
        self,
        n_samples: Optional[int] = None,
        return_df: bool = True,
        save_path: Optional[str] = None,
        *args,
        **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Generate synthetic tabular samples.
        """

        if not self.is_fitted or self.state is None:
            raise RuntimeError(
                "The model must be trained before sampling."
            )

        out = sample_tabsyn(
            self.state,
            n_samples=n_samples,
            return_df=return_df,
        )

        # Save generated samples if requested
        if save_path is not None:

            if isinstance(out, pd.DataFrame):
                out.to_csv(save_path, index=False)

            else:
                np.save(save_path, out)


        return out