import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.naivebayes.models import NaiveBayesModel

config = RunConfig(
    dataset_name="shuttle",
    model_name="naivebayes",
    categorical_cols=[],
    continuous_cols=[
        "time",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
    ],
    target_col_raw="class",
    constraints=None,
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
