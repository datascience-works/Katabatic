import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig

config = RunConfig(
    dataset_name     = "bank_marketing",
    model_name       = "tabebm",
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'poutcome'],
    continuous_cols  = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
    target_col_raw   = "y",
    constraints      = {
        'age':      (18,    95),
        'balance':  (-8019, 102127),
        'day':      (1,     31),
        'duration': (0,     4918),
        'campaign': (1,     63),
        'pdays':    (-1,    871),
        'previous': (0,     275),
    },
    max_train_rows   = None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

tabebm_config = TabEBMConfig(
    max_data_size            = 1000,
    starting_point_noise_std = 0.01,
    sgld_step_size           = 0.01,
    sgld_noise_std           = 0.01,
    sgld_steps               = 200,
    distance_negative_class  = 5.0,
    seed                     = 42,
)

model = TabEBMModel(target_col=target_col, config=tabebm_config)
model.train(output_dir=paths["split_dir"], synthetic_dir=paths["synthetic_dir"])

x_synth, y_synth = model.sample(1000)
x_synth[target_col] = y_synth.values

synthetic_df = save_synthetic(x_synth, train_df, paths, categorical_cols=config.categorical_cols)
evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)