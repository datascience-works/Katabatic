import json
import os

import numpy as np
import pandas as pd


def prepare_npy_for_tabsyn(split_dir: str, categorical_cols: list, continuous_cols: list):
    for split in ["train", "test"]:
        x_df = pd.read_csv(os.path.join(split_dir, f"x_{split}.csv"))
        y_df = pd.read_csv(os.path.join(split_dir, f"y_{split}.csv"))

        num_cols = [c for c in x_df.columns if c in continuous_cols]
        cat_cols = [c for c in x_df.columns if c in categorical_cols]

        if num_cols:
            X_num = x_df[num_cols].to_numpy(dtype=np.float64)
            np.save(os.path.join(split_dir, f"X_num_{split}.npy"), X_num)

        if cat_cols:
            X_cat = x_df[cat_cols].astype(str).to_numpy(dtype=object)
            np.save(os.path.join(split_dir, f"X_cat_{split}.npy"), X_cat)

        y = y_df.iloc[:, 0].astype(str).to_numpy(dtype=object)
        np.save(os.path.join(split_dir, f"y_{split}.npy"), y)

        print(f"[{split}] num={len(num_cols)} cols, cat={len(cat_cols)} cols, y={len(y)} rows")

    x_train_df = pd.read_csv(os.path.join(split_dir, "x_train.csv"))
    all_cols = list(x_train_df.columns)
    num_idx = [all_cols.index(c) for c in continuous_cols if c in all_cols]
    cat_idx = [all_cols.index(c) for c in categorical_cols if c in all_cols]

    y_train_df = pd.read_csv(os.path.join(split_dir, "y_train.csv"))
    n_classes = y_train_df.iloc[:, 0].nunique()
    task_type = "binclass" if n_classes == 2 else "multiclass"

    info = {
        "task_type": task_type,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": [len(all_cols)],
    }

    with open(os.path.join(split_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved info.json: {info}")