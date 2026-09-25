import logging
import os
import platform
import sys
import warnings
from time import perf_counter

import pandas as pd
import psutil
import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabula.models import TABULA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("pgmpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
start_time = perf_counter()

# Device detection — supports CUDA, Apple Silicon (MPS), and CPU fallback
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Paper: Zhao et al. (2023) "Tabula: Harnessing Language Models for Tabular Data Synthesis"
# arXiv:2310.12746. Bank Marketing dataset, large dataset setting.
# Paper hyperparameters: epochs=50 for large datasets, distilgpt2 with random init, k=16.
config = RunConfig(
    dataset_name="bank_marketing",
    model_name="tabula",
    categorical_cols=["job", "marital", "education", "default", "housing", "loan",
                      "contact", "month", "poutcome"],
    continuous_cols=["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
    target_col_raw="y",
    constraints={
        "age": (18, 95),
        "balance": (-8019, 102127),
        "day": (1, 31),
        "duration": (0, 4918),
        "campaign": (1, 63),
        "pdays": (-1, 871),
        "previous": (0, 275),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Subsample for validation on CPU/MPS — cap at 50 rows per class (100 total, binary target).
# Remove this block and run the full dataset on GPU to reproduce paper results.
_groups = []
for cls_val in train_df[target_col].unique():
    _g = train_df[train_df[target_col] == cls_val]
    _groups.append(_g.sample(n=min(len(_g), 50), random_state=42))
_sub = pd.concat(_groups).reset_index(drop=True)
_sub.drop(columns=[target_col]).to_csv(
    os.path.join(paths["split_dir"], "x_train.csv"), index=False
)
_sub[[target_col]].to_csv(
    os.path.join(paths["split_dir"], "y_train.csv"), index=False
)

print("\n" + "=" * 60)
print("STEP 3 — Train TabuLa")
print("=" * 60)
model = TABULA(
    categorical_columns=config.categorical_cols,
    epochs=50,  # paper: 50 epochs for large datasets (Zhao et al., 2023, Table 2)
)

model.train(
    dataset_dir=paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    device=device,
    n_samples=1000,
    k=16,           # paper: k=16 generation batch size per round (Zhao et al., 2023, Section 4.2)
    max_length=256,
    max_rounds=100,
)
print("\nTabuLa training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
x_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "x_synth.csv"))
y_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "y_synth.csv"))
synthetic_df = pd.concat([x_synth, y_synth], axis=1)
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)

end_time = perf_counter()
time_diff = end_time - start_time
print("\n" + "=" * 70)
print(f"tabula has taken {time_diff:.2f} seconds to run the bank_marketing dataset.")
print("=" * 70)