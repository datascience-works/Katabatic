import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tabkde.models import TabKDEModel

config = RunConfig(
    dataset_name="adult",
    model_name="tabkde",
    categorical_cols=[
        "workclass",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ],
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
    target_col_raw="income",
    constraints={
        "age": (17, 90),
        "fnlwgt": (12285, 1490400),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train TabKDE")
print("=" * 60)
model = TabKDEModel()
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nTabKDE training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
