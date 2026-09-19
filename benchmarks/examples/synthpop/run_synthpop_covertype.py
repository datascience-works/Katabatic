import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic
from katabatic.models.synthpop import SynthPop

config = RunConfig(
    dataset_name="covtype",
    model_name="synthpop",
    categorical_cols=[],
    continuous_cols=[
        "Elevation", "Aspect", "Slope",
        "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4",
        "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5",
        "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10",
        "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15",
        "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20",
        "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25",
        "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30",
        "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35",
        "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40",
    ],
    target_col_raw="Cover_Type",
    max_train_rows=None,
    constraints={
        "Elevation": (1859, 3858),
        "Aspect": (0, 360),
        "Slope": (0, 66),
        "Horizontal_Distance_To_Hydrology": (0, 1397),
        "Vertical_Distance_To_Hydrology": (-173, 601),
        "Horizontal_Distance_To_Roadways": (0, 7117),
        "Hillshade_9am": (0, 254),
        "Hillshade_Noon": (0, 254),
        "Hillshade_3pm": (0, 254),
        "Horizontal_Distance_To_Fire_Points": (0, 7173),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Per-class subsampling — explicit loop avoids pandas 2.x groupby key-drop bug
MAX_PER_CLASS = 200
subsampled = []
for cls_val, group in train_df.groupby(target_col):
    subsampled.append(group.sample(n=min(len(group), MAX_PER_CLASS), random_state=42))
train_df = pd.concat(subsampled).reset_index(drop=True)
print(f"Subsampled train_df: {len(train_df)} rows | class counts:\n{train_df[target_col].value_counts()}")

os.makedirs(paths["synthetic_dir"], exist_ok=True)

train_csv_path = os.path.join(paths["split_dir"], "train_synthpop.csv")
train_df.to_csv(train_csv_path, index=False)

synthetic_csv_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")

model = SynthPop(seed=42)
model.train(dataset_path=train_csv_path, synthetic_path=synthetic_csv_path)
synthetic_df = model.sample()
synthetic_df = save_synthetic(synthetic_df, train_df, paths, categorical_cols=config.categorical_cols)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
print("SynthPop Covertype benchmarking completed.")