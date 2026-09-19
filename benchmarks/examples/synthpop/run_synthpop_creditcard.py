import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.synthpop import SynthPop

config = RunConfig(
    dataset_name="creditcard",
    model_name="synthpop",
    categorical_cols=[],
    continuous_cols=[
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount",
    ],
    target_col_raw="Class",
    max_train_rows=None,
    constraints={
        "Time":   (0.0,     172792.0),
        "V1":     (-56.41,  2.45),
        "V2":     (-72.72,  22.06),
        "V3":     (-48.33,  9.38),
        "V4":     (-5.68,   16.88),
        "V5":     (-113.74, 34.80),
        "V6":     (-26.16,  73.30),
        "V7":     (-43.56,  120.59),
        "V8":     (-73.22,  20.01),
        "V9":     (-13.43,  15.60),
        "V10":    (-24.59,  23.75),
        "V11":    (-4.80,   12.02),
        "V12":    (-18.68,  7.85),
        "V13":    (-5.79,   7.13),
        "V14":    (-19.21,  10.53),
        "V15":    (-4.50,   8.88),
        "V16":    (-14.13,  17.32),
        "V17":    (-25.16,  9.25),
        "V18":    (-9.50,   5.04),
        "V19":    (-7.21,   5.59),
        "V20":    (-54.50,  39.42),
        "V21":    (-34.83,  27.20),
        "V22":    (-10.93,  10.50),
        "V23":    (-44.81,  22.53),
        "V24":    (-2.84,   4.58),
        "V25":    (-10.30,  7.52),
        "V26":    (-2.60,   3.52),
        "V27":    (-22.57,  31.61),
        "V28":    (-15.43,  33.85),
        "Amount": (0.0,     25691.16),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Per-class subsampling — creditcard is heavily imbalanced (99.8% vs 0.2%)
# Subsample to keep runs manageable while preserving class ratio
MAX_PER_CLASS = 500
subsampled = []
for cls_val, group in train_df.groupby(target_col):
    subsampled.append(group.sample(n=min(len(group), MAX_PER_CLASS), random_state=42))
train_df = pd.concat(subsampled).reset_index(drop=True)
print(f"Subsampled train_df: {len(train_df)} rows | class counts:\n{train_df[target_col].value_counts()}")

# SynthPop reads a CSV directly (R cannot take a DataFrame)
os.makedirs(paths["synthetic_dir"], exist_ok=True)
train_csv_path = os.path.join(paths["split_dir"], "train_synthpop.csv")
train_df.to_csv(train_csv_path, index=False)

synthetic_csv_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")

# SynthPop — parameters from Nowok et al. (2016), Journal of Statistical Software 74(11)
# method = "cart"      : CART sequential conditional synthesis
# cart.minbucket = 5   : Minimum observations in leaf nodes (paper p.6)
# m = 1                : One synthetic dataset (simple synthesis)
# k = nrow(data)       : Match original dataset size
model = SynthPop(seed=42)

model.train(
    dataset_path=train_csv_path,
    synthetic_path=synthetic_csv_path,
)

synthetic_df = model.sample()
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

# Subsample eval set for manageable evaluation
eval_train_df = train_df.sample(n=min(len(train_df), 5000), random_state=42)
evaluate(model, config, eval_train_df, synthetic_df, target_col, paths, test_df)

print("SynthPop Credit Card benchmarking completed.")