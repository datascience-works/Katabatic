import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.pategan.models import PATEGAN

config = RunConfig(
    dataset_name="adult",
    model_name="pategan",
    categorical_cols=[
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ],
    continuous_cols=[
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],
    target_col_raw="class",
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
print("STEP 3 — Train PATEGAN")
print("=" * 60)

model = PATEGAN(
    epsilon=5.0,
    delta=1e-5,
    num_teachers=10,
    niter=10000,
    batch_size=128,
    z_dim=16,
    learning_rate=5e-5,
    random_state=42,
)

model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

print("\nPATEGAN training complete.")


print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)

synthetic_df = model.sample(len(train_df))

synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)


evaluate(
    model,
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)
