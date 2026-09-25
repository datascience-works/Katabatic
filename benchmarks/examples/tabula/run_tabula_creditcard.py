import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import torch
from benchmarks.runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabula.models import TABULA

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = RunConfig(
    dataset_name="creditcard",
    model_name="tabula",
    categorical_cols=[],
    continuous_cols=[
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
        "V28", "Amount",
    ],
    target_col_raw="Class",
    constraints={
        "Amount": (0, 25691),
        "Time": (0, 172792),
    },
    max_train_rows=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Subsample to avoid memory issues (credit card has 284k rows)
_groups = []
for cls_val in train_df[target_col].unique():
    _g = train_df[train_df[target_col] == cls_val]
    _groups.append(_g.sample(n=min(len(_g), 500), random_state=42))
_sub = pd.concat(_groups).reset_index(drop=True)
_sub.drop(columns=[target_col]).to_csv(
    os.path.join(paths["split_dir"], "x_train.csv"), index=False
)
_sub[[target_col]].to_csv(
    os.path.join(paths["split_dir"], "y_train.csv"), index=False
)

model = TABULA(
    categorical_columns=[],
    epochs=50,  # paper uses 50 epochs for large datasets (Zhao et al., 2023)
)

model.train(
    dataset_dir=paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    device=device,
    n_samples=1000,
    k=16,
    max_length=512,
    max_rounds=100,
)

x_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "x_synth.csv"))
y_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "y_synth.csv"))
synthetic_df = pd.concat([x_synth, y_synth], axis=1)
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=[])

eval_train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
evaluate(model, config, eval_train_df, synthetic_df, target_col, paths, test_df)
