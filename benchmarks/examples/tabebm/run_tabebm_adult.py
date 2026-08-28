import os
import sys
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig

config = RunConfig(
    dataset_name="adult",
    model_name="tabebm",
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

tabebm_config = TabEBMConfig(
    max_data_size=10000,
    starting_point_noise_std=0.01,
    sgld_step_size=0.01,
    sgld_noise_std=0.01,
    sgld_steps=200,
    distance_negative_class=5.0,
    seed=42,
)

model = TabEBMModel(
    target_col=target_col,
    config=tabebm_config,
)

model.train(
    output_dir=paths["data_dir"],
    synthetic_dir=paths["synthetic_dir"],
)

x_synth, y_synth = model.sample(len(train_df))

synthetic_df = save_synthetic(
    x_synth, train_df, paths, categorical_cols=config.categorical_cols, y_synth=y_synth
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
