import pandas as pd
import numpy as np

from katabatic.models.base_model import Model
from katabatic.models.CopulaGAN.models import CopulaGANModel


class CopulaGANAdapter(Model):
    """
    Katabatic Model-style adapter for CopulaGAN.

    - Automatically resolves label column per dataset
    - No dataset-specific logic required in pipeline scripts
    """

   
    DATASET_LABEL_MAP = {
        "adult": "income",
        "magic": "class",
        "nursery": "8",
        "shuttle": "class",
        "car": "6",
    }

    def __init__(self, target_col: str = "target", **copula_kwargs):
        super().__init__()
        self.target_col = target_col
        self.copula_kwargs = copula_kwargs
        self.model = None

    def train(self, output_dir: str, label_col=None):
       
        # Auto-detect dataset name
      
        dataset_name = output_dir.rstrip("/").split("/")[-1]

       
        # Resolve label column
        
        if label_col is None:
            if dataset_name not in self.DATASET_LABEL_MAP:
                raise ValueError(
                    f"Unknown dataset '{dataset_name}'. "
                    f"Expected one of {list(self.DATASET_LABEL_MAP.keys())}"
                )
            label_col = self.DATASET_LABEL_MAP[dataset_name]

   
        # Load training split
        
        x_train_df = pd.read_csv(f"{output_dir}/x_train.csv")
        y_df = pd.read_csv(f"{output_dir}/y_train.csv")

        if label_col is None:
            y_train = y_df.iloc[:, 0]
        else:
            col = str(label_col)
            if col not in y_df.columns:
                y_df.columns = [str(c) for c in y_df.columns]
            y_train = y_df[str(col)]

        
        # Fit CopulaGAN
        
        self.model = CopulaGANModel(
            target_col=self.target_col,
            **self.copula_kwargs
        )
        self.model.fit(x_train_df, y_train)

        self.is_fitted = True
        return self

    def sample(self, n_samples: int):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be trained before sampling.")

        x_synth_df, y_synth = self.model.sample(n_samples)

        return x_synth_df.to_numpy(), np.asarray(y_synth)

    def evaluate(self, *args, **kwargs):
        return None