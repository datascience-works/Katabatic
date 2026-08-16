import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.meg.models import MEGModel

config = RunConfig(
    dataset_name     = "magic",
    model_name       = "meg",
    categorical_cols = [],
    continuous_cols  = ["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist"],
    target_col_raw   = "class",
    constraints      = None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("STEP 3 - Train MEGModel")
model = MEGModel(
    dataset_name   = "magic",
    epochs         = 100,
    batch_size     = 256,
    ensemble_size  = 5,
    hidden         = 512,
    lr             = 2e-3,
    weight_decay   = 1e-4,
    n_impute_steps = 20,
    noise_std      = 0.03,
    mask_span_prob = 0.35,
    balance_classes = False,
    harden_cats    = True,
    device         = "auto",
)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("MEGModel training complete.")

print("STEP 4 - Generate synthetic data")
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)
evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
