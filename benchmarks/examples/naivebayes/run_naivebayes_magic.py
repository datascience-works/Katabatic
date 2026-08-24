import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.naivebayes.models import NaiveBayesModel

config = RunConfig(
    dataset_name="magic",
    model_name="naivebayes",
    categorical_cols=[],
    continuous_cols=[
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
    ],
    target_col_raw="class",
    constraints={
        "fLength": (4.28, 334.18),
        "fWidth": (0.0, 256.39),
        "fSize": (0.0, 2.96),
        "fConc": (0.0, 0.98),
        "fConc1": (0.0, 0.98),
        "fAsym": (-457.88, 575.81),
        "fM3Long": (-21.87, 74.63),
        "fM3Trans": (-8.67, 73.0),
        "fAlpha": (-2.06, 1.4),
        "fDist": (0.0, 283.21),
    },
)


train_df, test_df, target_col, paths = preprocess_and_split(config)


print("\n" + "=" * 60)
print("STEP 3 — Train Naive Bayes")
print("=" * 60)

model = NaiveBayesModel(seed=42)

model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

print("\nNaive Bayes training complete.")


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
