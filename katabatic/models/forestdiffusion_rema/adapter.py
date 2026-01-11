import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from katabatic.models.base_model import Model
from katabatic.models.forestdiffusion.models import ForestDiffusionModel


class ForestDiffusionAdapter(Model):
    def __init__(self, **fd_kwargs):
        super().__init__()
        self.fd_kwargs = fd_kwargs
        self.model = None
        self.encoder = None

    def train(self, output_dir: str, label_col=None):
        x_train_df = pd.read_csv(f"{output_dir}/x_train.csv")
        y_df = pd.read_csv(f"{output_dir}/y_train.csv")

        if label_col is None:
            y_train = y_df.iloc[:, 0].values
        else:
            col = str(label_col)
            if col not in y_df.columns:
                y_df.columns = [str(c) for c in y_df.columns]
            y_train = y_df[str(label_col)].values

        if any(x_train_df.dtypes == "object"):
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            x_train = self.encoder.fit_transform(x_train_df.astype(str))
        else:
            x_train = x_train_df.to_numpy()

        x_train = x_train.astype(np.float32)

        self.model = ForestDiffusionModel(
            X=x_train,
            label_y=y_train,
            **self.fd_kwargs
        )

        self.is_fitted = True
        return self

    def sample(self, n_samples: int):
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be trained before sampling.")

        synth = self.model.generate(batch_size=n_samples)
        x_synth = synth[:, :-1]
        y_synth = synth[:, -1]
        return x_synth, y_synth

    def evaluate(self, *args, **kwargs):
        return None