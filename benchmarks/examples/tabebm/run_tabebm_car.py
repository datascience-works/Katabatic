import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig

config = RunConfig(
dataset_name     = "car",
model_name       = "tabebm",
categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
continuous_cols  = [],
target_col_raw   = "class",
constraints      = None,
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
model.train(output_dir=paths["split_dir"], synthetic_dir=paths["synthetic_dir"])

x_synth, y_synth = model.sample(1000)  # testing only
x_synth[target_col] = y_synth.values
synthetic_df = save_synthetic(x_synth, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)