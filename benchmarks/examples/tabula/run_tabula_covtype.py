import logging
import os
import sys
import warnings
from time import perf_counter

import pandas as pd
import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.tabula.models import TABULA

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
start_time = perf_counter()

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

config = RunConfig(
    dataset_name="covtype",
    model_name="tabula",
    categorical_cols=[],
    continuous_cols=[
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
    target_col_raw="Cover_Type",
    constraints={
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
    max_train_rows=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Subsample — covtype is 464k rows, 54 cols; text rows are very long so keep small.
# Remove this block and run on full dataset on GPU to reproduce paper results.
_groups = []
for cls_val in train_df[target_col].unique():
    _g = train_df[train_df[target_col] == cls_val]
    _groups.append(_g.sample(n=min(len(_g), 200), random_state=42))
_sub = pd.concat(_groups).reset_index(drop=True)
_sub.drop(columns=[target_col]).to_csv(
    os.path.join(paths["split_dir"], "x_train.csv"), index=False
)
_sub[[target_col]].to_csv(
    os.path.join(paths["split_dir"], "y_train.csv"), index=False
)

model = TABULA(
    categorical_columns=[],
    epochs=50,  # paper: 50 epochs for large datasets (Zhao et al., 2023, Table 2)
)

model.train(
    dataset_dir=paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    device=device,
    n_samples=1000,
    k=8,  # reduced from 16 — covtype rows are 54 cols wide, lower k keeps memory manageable
    max_length=512,
    max_rounds=150,
)

x_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "x_synth.csv"))
y_synth = pd.read_csv(os.path.join(paths["synthetic_dir"], "y_synth.csv"))
synthetic_df = pd.concat([x_synth, y_synth], axis=1)
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=[])

eval_train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
evaluate(model, config, eval_train_df, synthetic_df, target_col, paths, test_df)

end_time = perf_counter()
print(f"\ntabula has taken {end_time - start_time:.2f} seconds to run the covtype dataset.")