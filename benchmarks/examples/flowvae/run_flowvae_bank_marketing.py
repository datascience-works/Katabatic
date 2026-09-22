import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.flowvae.models import FlowVAEModel

config = RunConfig(
    dataset_name="bank_marketing",
    model_name="flowvae",
    categorical_cols=[
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ],
    continuous_cols=[
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ],
    target_col_raw="y",
    constraints={
        "age": (18, 100),
        "balance": (-8019, 102127),
        "day": (1, 31),
        "duration": (0, 5000),
        "campaign": (1, 63),
        "pdays": (-1, 999),
        "previous": (0, 275),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train FlowVAE")
print("=" * 60)
model = FlowVAEModel(
    hidden_dim=128,
    latent_dim=16,
    layers=2,
    flow_type="planar",
    flow_length=4,
    batch_size=256,
    epochs=100,
    learning_rate=1e-3,
    random_state=42,
)
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nFlowVAE training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
