import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabebm.models import TabEBMModel, TabEBMConfig
import pandas as pd

config = RunConfig(
    dataset_name     = "covtype",
    model_name       = "tabebm",
    categorical_cols = [],
    continuous_cols  = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
        'Soil_Type1',  'Soil_Type2',  'Soil_Type3',  'Soil_Type4',  'Soil_Type5',
        'Soil_Type6',  'Soil_Type7',  'Soil_Type8',  'Soil_Type9',  'Soil_Type10',
        'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
        'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
        'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
        'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
        'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',
    ],
    target_col_raw   = "Cover_Type",
    constraints      = {
        'Elevation':                          (1859, 3858),
        'Aspect':                             (0,    360),
        'Slope':                              (0,    66),
        'Horizontal_Distance_To_Hydrology':   (0,    1397),
        'Vertical_Distance_To_Hydrology':     (-173, 601),
        'Horizontal_Distance_To_Roadways':    (0,    7117),
        'Hillshade_9am':                      (0,    254),
        'Hillshade_Noon':                     (0,    254),
        'Hillshade_3pm':                      (0,    254),
        'Horizontal_Distance_To_Fire_Points': (0,    7173),
    },
    max_train_rows   = None,   # ← required to avoid pandas groupby bug
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Subsample x_train/y_train to avoid OOM (covertype ~464k rows, 54 cols)
_groups = []
for cls_val in train_df[target_col].unique():
    _g = train_df[train_df[target_col] == cls_val]
    _groups.append(_g.sample(n=min(len(_g), 2000), random_state=42))
_sub = pd.concat(_groups).reset_index(drop=True)
_sub.drop(columns=[target_col]).to_csv(
    os.path.join(paths["split_dir"], "x_train.csv"), index=False
)
_sub[[target_col]].to_csv(
    os.path.join(paths["split_dir"], "y_train.csv"), index=False
)

tabebm_config = TabEBMConfig(
    max_data_size            = 1000,  # testing only
    starting_point_noise_std = 0.01,
    sgld_step_size           = 0.01,
    sgld_noise_std           = 0.01,
    sgld_steps               = 200,
    distance_negative_class  = 5.0,
    seed                     = 42,
)

model = TabEBMModel(target_col=target_col, config=tabebm_config)
model.train(output_dir=paths["split_dir"], synthetic_dir=paths["synthetic_dir"])

x_synth, y_synth = model.sample(1000)  # testing only
x_synth[target_col] = y_synth.values
synthetic_df = save_synthetic(x_synth, train_df, paths, categorical_cols=config.categorical_cols)

eval_train_df = train_df.sample(n=5000, random_state=42)
evaluate(model, config, eval_train_df, synthetic_df, target_col, paths, test_df)