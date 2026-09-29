import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.meg.models import MEGModel

config = RunConfig(
    dataset_name="car",
    model_name="meg",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train MEGModel")
print("=" * 60)
model = MEGModel(
    dataset_name="car",  # sets epochs=50 automatically; override with epochs=
    epochs=200,
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
print("\MEGModel training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
