import os
import sys
import warnings

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import (  # noqa: E402
    RunConfig,
    evaluate,
    preprocess_and_split,
    save_synthetic,
)

from katabatic.experimental.models.tabebm.models import (  # noqa: E402
    TabEBMConfig,
    TabEBMModel,
)

warnings.filterwarnings("ignore")
config = RunConfig(
    dataset_name="nursery",
    model_name="tabebm",
    categorical_cols=["0", "1", "2", "3", "4", "5", "6", "7"],
    continuous_cols=[],
    target_col_raw="8",
    constraints=None,
)


train_df, test_df, target_col, paths = preprocess_and_split(config)

tabebm_config = TabEBMConfig(
    max_data_size=1000,  # testing only
    starting_point_noise_std=0.01,
    sgld_step_size=0.01,
    sgld_noise_std=0.01,
    sgld_steps=200,
    distance_negative_class=5.0,
    seed=42,
)

model = TabEBMModel(target_col=target_col, config=tabebm_config)
model.train(paths["split_dir"])

synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
