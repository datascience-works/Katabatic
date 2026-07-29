import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.medgan.models import MEDGAN

config = RunConfig(
    dataset_name="creditcard",
    model_name="medgan",
    categorical_cols=[],
    continuous_cols=[
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ],
    target_col_raw="Class",
    constraints={
        "Time": (0.0, 172792.0),
        "V1": (-56.41, 2.45),
        "V2": (-72.72, 22.06),
        "V3": (-48.33, 9.38),
        "V4": (-5.68, 16.88),
        "V5": (-113.74, 34.80),
        "V6": (-26.16, 73.30),
        "V7": (-43.56, 120.59),
        "V8": (-73.22, 20.01),
        "V9": (-13.43, 15.60),
        "V10": (-24.59, 23.75),
        "V11": (-4.80, 12.02),
        "V12": (-18.68, 7.85),
        "V13": (-5.79, 7.13),
        "V14": (-19.21, 10.53),
        "V15": (-4.50, 8.88),
        "V16": (-14.13, 17.32),
        "V17": (-25.16, 9.25),
        "V18": (-9.50, 5.04),
        "V19": (-7.21, 5.59),
        "V20": (-54.50, 39.42),
        "V21": (-34.83, 27.20),
        "V22": (-10.93, 10.50),
        "V23": (-44.81, 22.53),
        "V24": (-2.84, 4.58),
        "V25": (-10.30, 7.52),
        "V26": (-2.60, 3.52),
        "V27": (-22.57, 31.61),
        "V28": (-15.43, 33.85),
        "Amount": (0.0, 25691.16),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

print("\n" + "=" * 60)
print("STEP 3 — Train MedGAN")
print("=" * 60)
model = MEDGAN(ae_pretrain_epochs=100, gan_epochs=1000, batch_size=256, random_state=42)
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nMedGAN training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(len(train_df))
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
