import os
import sys
import warnings

import pandas as pd
import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tabula.models import TABULA  # noqa: E402

warnings.filterwarnings("ignore")

# Device detection: CUDA > Apple Silicon MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

config = RunConfig(
    dataset_name="adult",
    model_name="tabula",
    categorical_cols=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    continuous_cols=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
    target_col_raw="class",
    constraints={
        "age": (17, 90),
        "fnlwgt": (12285, 1490400),
        "education-num": (1, 16),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Subsample to 50 rows per class for CPU/MPS validation.
# Remove this block and use a CUDA GPU to reproduce full paper results (Zhao et al., 2023).
_groups = []
for cls_val in train_df[target_col].unique():
    _g = train_df[train_df[target_col] == cls_val]
    _groups.append(_g.sample(n=min(len(_g), 50), random_state=42))
_sub = pd.concat(_groups).reset_index(drop=True)
_sub.drop(columns=[target_col]).to_csv(os.path.join(paths["split_dir"], "x_train.csv"), index=False)
_sub[[target_col]].to_csv(os.path.join(paths["split_dir"], "y_train.csv"), index=False)

model = TABULA(
    categorical_columns=config.categorical_cols,
    epochs=50,  # paper: 50 epochs for large datasets (Zhao et al., 2023)
)

model.train(
    dataset_dir=paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    device=device,
    n_samples=1000,
    k=16,
    max_length=256,
    max_rounds=100,
)

x_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "x_synth.csv"))
y_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "y_synth.csv"))
synthetic_df = pd.concat([x_synth, y_synth], axis=1)
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)