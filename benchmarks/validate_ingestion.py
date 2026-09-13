import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner import RunConfig, preprocess_and_split

DATASETS = ["adult", "car", "magic", "nursery", "shuttle"]

for name in DATASETS:
    print("\n" + "#" * 60)
    print(f"# Validating ingestion: {name}")
    print("#" * 60)
    try:
        config = RunConfig(dataset_name=name, model_name="ingestion_check")
        train_df, test_df, target_col, paths = preprocess_and_split(config)
        print(
            f"[OK] {name}: train={train_df.shape}, test={test_df.shape}, target='{target_col}'"
        )
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
