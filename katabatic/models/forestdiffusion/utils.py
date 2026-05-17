import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def encode_categorical_features(x_train_df: pd.DataFrame):
    if any(x_train_df.dtypes == "object"):
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        x_train = encoder.fit_transform(x_train_df.astype(str))
        return x_train.astype(np.float32), encoder

    return x_train_df.to_numpy().astype(np.float32), None