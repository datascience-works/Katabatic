import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.tvaegan.models import TVAEGANModel

config = RunConfig(
    dataset_name     = "bank_marketing",
    model_name       = "tvaegan",
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
        "age": (18, 95),  # legal working/banking age range
        "balance": (-8020, None),  # min observed in dataset, no upper bound
        "day": (1, 31),  # day of month
        "duration": (0, None),  # call duration in seconds, cannot be negative
        "campaign": (1, None),  # at least 1 contact was made
        "pdays": (-1, None),  # -1 = not previously contacted, otherwise >= 0
        "previous": (0, None),  # number of previous contacts, cannot be negative
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train TVAEGAN")
print("=" * 60)
model = TVAEGANModel(
    epochs                  = 10,
    batch_size              = 500,
    cat_emb_size            = 25,
    num_emb_size            = 25,
    w_regularize            = 1.0,
    w_reconstruct           = 10.0,
    s_generat               = 5,
    s_encoder               = 5,
    lr_generat              = 5e-5,
    lr_critic               = 5e-5,
    lr_encoder              = 5e-5,
    clip                    = 0.01,
    dropout                 = 0.1,
    hidden_layers_multipliers = [1.0, 1.0],
    shuffle                 = True,
    random_state            = 42,
)
model.train(paths["split_dir"], paths["synthetic_dir"])
print("\nTVAEGAN training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
