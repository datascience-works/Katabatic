import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.fairtabdiffusion.models import FairTabDiffusion

# Adult dataset column layout — matches this team's raw_data/adult.csv exactly:
# ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
#  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
#  'hours-per-week', 'native-country', 'class']
config = RunConfig(
    dataset_name="adult",
    model_name="fairtabdiffusion",
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
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train FairTabDiffusion")
print("=" * 60)
# Adult has a natural sensitive attribute ("sex"), so this is the first
# real test of FairTabDiffusion's fairness-conditioning mechanism —
# Car had no sensitive attribute and ran with sensitive_col=None.
model = FairTabDiffusion(
    sensitive_col="sex",
    epochs=200,
    timesteps=100,
    batch_size=256,
    seed=42,
)

model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)

print("\nFairTabDiffusion training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
