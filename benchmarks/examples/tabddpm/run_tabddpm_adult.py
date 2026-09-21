import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tabddpm.models import Tabddpm


# Adult dataset configuration
config = RunConfig(
    dataset_name="adult",
    model_name="tabddpm",
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
    continuous_cols=[
        "age",
        "fnlwgt",
        "education-num",
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


# STEP 1 and STEP 2: Preprocess the dataset and create train/test splits
train_df, test_df, target_col, paths = preprocess_and_split(config)


# STEP 3: Train TabDDPM
print("\n" + "=" * 60)
print("STEP 3 - Train TabDDPM")
print("=" * 60)

model = Tabddpm()

model.train(
    paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    config=dict(
        steps=2000,
        num_timesteps=1000,
        batch_size=256,
        d_layers=(256, 256, 256),
        use_ema=True,
        eval_batches=10,
    ),
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

print("\nTabDDPM training complete.")


# STEP 4: Generate synthetic data
print("\n" + "=" * 60)
print("STEP 4 - Generate synthetic data")
print("=" * 60)

synthetic_df = model.sample(
    len(train_df),
    as_dataframe=True,
)

# Tabddpm.sample() uses "label" for the generated target.
# Rename it to match the target column in the Adult dataset.
if "label" in synthetic_df.columns and target_col not in synthetic_df.columns:
    synthetic_df = synthetic_df.rename(columns={"label": target_col})

# Do not silently drop columns when saving synthetic data.
missing_cols = [
    col for col in train_df.columns
    if col not in synthetic_df.columns
]

if missing_cols:
    raise ValueError(
        f"TabDDPM output is missing columns: {missing_cols}"
    )

# Match the training dataset's column order.
synthetic_df = synthetic_df[train_df.columns]

synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)


# STEP 5: Evaluate the model
print("\n" + "=" * 60)
print("STEP 5 - Evaluate TabDDPM")
print("=" * 60)

evaluate(
    model,
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)