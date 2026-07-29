import json
import os
import sys
from dataclasses import dataclass, field

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline
from katabatic.utils.preprocess import preprocess_dataset
from katabatic.utils.split_dataset import split_dataset


@dataclass
class RunConfig:
    dataset_name: str
    model_name: str
    categorical_cols: list
    continuous_cols: list
    target_col_raw: str
    constraints: dict | None = None
    test_size: float = 0.2
    seed: int = 42
    max_train_rows: int | None = 50000  # set to None to use the full training set
    dimensions: list = field(
        default_factory=lambda: [
            "fidelity",
            "utility",
            "diversity",
            "privacy",
            "consistency",
            "stability",
        ]
    )


def build_paths(config: RunConfig) -> dict:
    benchmarks_dir = os.path.join(REPO_ROOT, "benchmarks")
    return {
        "raw_data": os.path.join(REPO_ROOT, "datasets", f"{config.dataset_name}.csv"),
        "processed_data": os.path.join(
            benchmarks_dir, "processed", f"{config.dataset_name}_processed.csv"
        ),
        "split_dir": os.path.join(benchmarks_dir, "splits", config.dataset_name),
        "synthetic_dir": os.path.join(
            benchmarks_dir, "synthetic", config.dataset_name, config.model_name
        ),
        "results_dir": os.path.join(
            benchmarks_dir, "results", config.dataset_name, config.model_name
        ),
    }


def _write_meta(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _preprocess_cache_valid(meta_path: str, target_col: str) -> bool:
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("target_col") == target_col


def _split_cache_valid(meta_path: str, test_size: float, seed: int) -> bool:
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("test_size") == test_size and meta.get("seed") == seed


def preprocess_and_split(config: RunConfig):
    """Preprocess raw CSV and split into train/test. Returns (train_df, test_df, target_col, paths)."""
    paths = build_paths(config)

    if not os.path.exists(paths["raw_data"]):
        raise FileNotFoundError(
            f"Raw dataset not found: {paths['raw_data']}\n"
            f"Place {config.dataset_name}.csv in the datasets/ folder and re-run."
        )

    print("\n" + "=" * 60)
    print("STEP 1 — Preprocess")
    print("=" * 60)
    os.makedirs(os.path.dirname(paths["processed_data"]), exist_ok=True)
    processed_meta_path = paths["processed_data"].replace(".csv", "_meta.json")
    preprocess_ran = False

    if os.path.exists(paths["processed_data"]) and _preprocess_cache_valid(
        processed_meta_path, config.target_col_raw
    ):
        print(
            "Processed data already exists and parameters match, skipping preprocessing."
        )
    else:
        if os.path.exists(paths["processed_data"]):
            print("[INFO] target_col_raw changed — regenerating processed data.")
        preprocess_dataset(
            paths["raw_data"], paths["processed_data"], config.target_col_raw
        )
        _write_meta(processed_meta_path, {"target_col": config.target_col_raw})
        preprocess_ran = True

    processed_df = pd.read_csv(paths["processed_data"])
    target_col = processed_df.columns[-1]
    n_features = processed_df.shape[1] - 1
    print(f"\nProcessed shape : {processed_df.shape}")
    print(f"Feature columns : {n_features}")
    print(f"Target column   : '{target_col}'")
    print(f"Target classes  : {sorted(processed_df[target_col].unique())}")

    print("\n" + "=" * 60)
    print("STEP 2 — Train / test split (80 / 20, stratified)")
    print("=" * 60)
    train_full_path = os.path.join(paths["split_dir"], "train_full.csv")
    split_meta_path = os.path.join(paths["split_dir"], "split_meta.json")

    if (
        not preprocess_ran
        and os.path.exists(train_full_path)
        and _split_cache_valid(split_meta_path, config.test_size, config.seed)
    ):
        print("Split data already exists and parameters match, skipping split.")
    else:
        if os.path.exists(train_full_path) and not preprocess_ran:
            print("[INFO] test_size or seed changed — regenerating split.")
        split_dataset(
            paths["processed_data"],
            paths["split_dir"],
            test_size=config.test_size,
            seed=config.seed,
        )
        _write_meta(
            split_meta_path, {"test_size": config.test_size, "seed": config.seed}
        )

    train_df = pd.read_csv(os.path.join(paths["split_dir"], "train_full.csv"))
    test_df = pd.read_csv(os.path.join(paths["split_dir"], "test_full.csv"))
    print(f"\nTrain rows: {len(train_df)}   Test rows: {len(test_df)}")

    if config.max_train_rows is not None and len(train_df) > config.max_train_rows:
        print(
            f"\n[INFO] Training set ({len(train_df):,} rows) exceeds max_train_rows={config.max_train_rows:,}."
        )
        train_df = (
            train_df.groupby(target_col, group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, round(config.max_train_rows * len(g) / len(train_df))),
                    random_state=config.seed,
                )
            )
            .reset_index(drop=True)
        )
        print(f"[INFO] Stratified sample applied -> {len(train_df):,} rows retained.")
        print(
            f"       Class distribution: {train_df[target_col].value_counts().to_dict()}"
        )

        # Write sample to disk so models read consistent data.
        # train_full.csv is preserved as the immutable full split.
        train_df.to_csv(
            os.path.join(paths["split_dir"], "train_sample.csv"), index=False
        )
        print("[INFO] Subsampled train written to train_sample.csv in split dir.")

    # Always write x_train.csv / y_train.csv so models using the X/y split format can find them.
    train_df.drop(columns=[target_col]).to_csv(
        os.path.join(paths["split_dir"], "x_train.csv"), index=False
    )
    train_df[[target_col]].to_csv(
        os.path.join(paths["split_dir"], "y_train.csv"), index=False
    )

    return train_df, test_df, target_col, paths


def save_synthetic(
    synthetic_df: pd.DataFrame,
    train_df: pd.DataFrame,
    paths: dict,
    categorical_cols: list = None,
) -> pd.DataFrame:
    """Validate, align columns to training data, and save synthetic CSV."""
    if not isinstance(synthetic_df, pd.DataFrame):
        raise TypeError("synthetic_df must be a pandas DataFrame")

    if synthetic_df.empty:
        raise ValueError("synthetic_df is empty")

    shared_cols = [c for c in train_df.columns if c in synthetic_df.columns]
    synthetic_df = synthetic_df[shared_cols]
    os.makedirs(paths["synthetic_dir"], exist_ok=True)
    synthetic_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"\nGenerated {len(synthetic_df)} synthetic rows")
    print(f"Saved to  : {synthetic_path}")
    print(f"Shape     : {synthetic_df.shape}")
    print(f"\nSample (first 3 rows):\n{synthetic_df.head(3).to_string()}")
    return synthetic_df


def evaluate(
    model,
    config: RunConfig,
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: str,
    paths: dict,
    test_df: pd.DataFrame = None,
):
    """Run the evaluation pipeline and print the final summary. Returns EvaluationReport."""
    print("\n" + "=" * 60)
    print("STEP 5 — Evaluate synthetic data")
    print("=" * 60)

    pipeline = SyntheticEvaluationPipeline(
        dimensions=config.dimensions,
        categorical_cols=config.categorical_cols,
        continuous_cols=config.continuous_cols,
    )
    report = pipeline.run(
        real_data=train_df,
        synthetic_data=synthetic_df,
        target_col=target_col,
        test_data=test_df,
        constraints=config.constraints,
        model=model,
        output_dir=paths["results_dir"],
        report_prefix=f"{config.model_name}_{config.dataset_name}_",
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Composite score : {report.composite_score:.4f}")
    print("Dimension scores:")
    for dim, score in report.dimension_scores.items():
        print(f"  {dim:<14} {score:.4f}")

    return report
