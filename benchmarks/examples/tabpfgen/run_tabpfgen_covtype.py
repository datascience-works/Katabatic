import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd
from katabatic.models.tabpfgen.models import TabPFGenModel
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

# max_train_rows=1000: TabPFN is designed for small datasets; this also keeps
# generation fast. runner.py caps x_train.csv to this size.
config = RunConfig(
    max_train_rows=999,
    dataset_name="covtype",
    model_name="tabpfgen",
    categorical_cols=[],
    continuous_cols=[
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",
        "Soil_Type1",
        "Soil_Type2",
        "Soil_Type3",
        "Soil_Type4",
        "Soil_Type5",
        "Soil_Type6",
        "Soil_Type7",
        "Soil_Type8",
        "Soil_Type9",
        "Soil_Type10",
        "Soil_Type11",
        "Soil_Type12",
        "Soil_Type13",
        "Soil_Type14",
        "Soil_Type15",
        "Soil_Type16",
        "Soil_Type17",
        "Soil_Type18",
        "Soil_Type19",
        "Soil_Type20",
        "Soil_Type21",
        "Soil_Type22",
        "Soil_Type23",
        "Soil_Type24",
        "Soil_Type25",
        "Soil_Type26",
        "Soil_Type27",
        "Soil_Type28",
        "Soil_Type29",
        "Soil_Type30",
        "Soil_Type31",
        "Soil_Type32",
        "Soil_Type33",
        "Soil_Type34",
        "Soil_Type35",
        "Soil_Type36",
        "Soil_Type37",
        "Soil_Type38",
        "Soil_Type39",
        "Soil_Type40",
    ],
    target_col_raw="Cover_Type",
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

print("\n" + "=" * 60)
print("STEP 3 — Generate synthetic data with TabPFGen")
print("=" * 60)
model = TabPFGenModel(
    task="classification",
    n_sgld_steps=300,
    sgld_step_size=0.01,
    sgld_noise_scale=0.01,
    balance_classes=True,
    device="auto",
    random_state=config.seed,
)
result = model.train(
    paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print(f"\nTabPFGen generation complete. SGLD steps: {result['n_sgld_steps']}")

print("\n" + "=" * 60)
print("STEP 4 — Load and validate synthetic data")
print("=" * 60)
synthetic_df = pd.read_csv(result["synthetic_csv"])
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
