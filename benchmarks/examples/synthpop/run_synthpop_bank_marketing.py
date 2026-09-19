import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.synthpop import SynthPop

config = RunConfig(
    dataset_name="bank_marketing",
    model_name="synthpop",
    categorical_cols=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
    continuous_cols=["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
    target_col_raw="y",
    max_train_rows=None,
    constraints={
        "age":      (18, 95),
        "balance":  (-8019, 102127),
        "day":      (1, 31),
        "duration": (0, 4918),
        "campaign": (1, 63),
        "pdays":    (-1, 871),
        "previous": (0, 275),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

MAX_PER_CLASS = 500
subsampled = []
for cls_val, group in train_df.groupby(target_col):
    subsampled.append(group.sample(n=min(len(group), MAX_PER_CLASS), random_state=42))
train_df = pd.concat(subsampled).reset_index(drop=True)
print(f"Subsampled train_df: {len(train_df)} rows | class counts:\n{train_df[target_col].value_counts()}")

os.makedirs(paths["synthetic_dir"], exist_ok=True)

train_csv_path = os.path.join(paths["split_dir"], "train_synthpop.csv")
train_df.to_csv(train_csv_path, index=False)

synthetic_csv_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")

model = SynthPop(seed=42)
model.train(dataset_path=train_csv_path, synthetic_path=synthetic_csv_path)
synthetic_df = model.sample()
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
print("SynthPop Bank Marketing benchmarking completed.")
