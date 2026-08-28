import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from katabatic.models.meg import MEGModel
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

config = RunConfig(
    dataset_name="adult",
    model_name="meg",
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
print("STEP 3 — Train MEG")
print("=" * 60)
model = MEGModel(
    dataset_name="adult",  # sets epochs=50 automatically; override with epochs=
    epochs=100,
    batch_size=256,
    ensemble_size=5,
    hidden=512,  # hidden layer width of each MaskedNet
    lr=2e-3,
    weight_decay=1e-4,
    n_impute_steps=20,  # iterative masked-imputation steps during generation
    noise_std=0.03,  # Gaussian noise added to masked inputs during training
    mask_span_prob=0.35,  # probability of masking each feature span per sample
    balance_classes=False,  # True = equal samples per class regardless of prior
    harden_cats=True,  # enforce hard one-hot on categoricals during generation
    device="auto",  # "auto", "cpu", or "cuda"
)
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nMEG training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
