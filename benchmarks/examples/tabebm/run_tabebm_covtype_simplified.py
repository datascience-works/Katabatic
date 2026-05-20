import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, preprocess_and_split, save_synthetic, evaluate
from katabatic.models.tabebm.models import TabEBMModel

config = RunConfig(
    dataset_name     = "covtype_privbayes",
    model_name       = "tabebm",
    categorical_cols = ["Wilderness_Area", "Soil_Type"],
    continuous_cols  = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"],
    target_col_raw   = "Cover_Type",
    max_train_rows   = 10000,
    constraints      = {"Elevation": (1859, 3858), "Aspect": (0, 360), "Slope": (0, 66), "Horizontal_Distance_To_Hydrology": (0, 1397), "Vertical_Distance_To_Hydrology": (-173, 601), "Horizontal_Distance_To_Roadways": (0, 7117), "Hillshade_9am": (0, 254), "Hillshade_Noon": (0, 254), "Hillshade_3pm": (0, 254), "Horizontal_Distance_To_Fire_Points": (0, 7173)},
)

train_df, test_df, target_col, paths = preprocess_and_split(config)
print("STEP 3 — Train TabEBM")
model = TabEBMModel(target_col=target_col)
model.train(paths["split_dir"], categorical_cols=config.categorical_cols, continuous_cols=config.continuous_cols)
print("TabEBM training complete.")
print("STEP 4 — Generate synthetic data")
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)
evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
