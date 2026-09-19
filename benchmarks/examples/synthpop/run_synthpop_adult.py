import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.synthpop import SynthPop

config = RunConfig(
    dataset_name="adult",
    model_name="synthpop",
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
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
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

# SynthPop reads a CSV directly (R cannot take a DataFrame)
# Save the (possibly subsampled) train_df to a CSV for the R script
os.makedirs(paths["synthetic_dir"], exist_ok=True)
train_csv_path = os.path.join(paths["split_dir"], "train_synthpop.csv")
train_df.to_csv(train_csv_path, index=False)

synthetic_csv_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")

# SynthPop — parameters from Nowok et al. (2016), Journal of Statistical Software 74(11)
# method = "cart"      : CART sequential conditional synthesis
# cart.minbucket = 5   : Minimum observations in leaf nodes (paper p.6)
# m = 1                : One synthetic dataset (simple synthesis)
# k = nrow(data)       : Match original dataset size
model = SynthPop(seed=42)

model.train(
    dataset_path=train_csv_path,
    synthetic_path=synthetic_csv_path,
)

synthetic_df = model.sample()
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)

print("SynthPop Adult benchmarking completed.")