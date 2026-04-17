import json
import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from katabatic.utils.preprocess import encode_preprocess
from katabatic.utils.split_dataset import split_dataset
from katabatic.pipeline.evaluation_pipeline import SyntheticEvaluationPipeline


@dataclass
class RunConfig:
    dataset_name: str
    model_name: str
    categorical_cols: list
    continuous_cols: list
    target_col_raw: str
    constraints: Optional[dict] = None
    test_size: float = 0.2
    seed: int = 42
    max_train_rows: Optional[int] = 50000   # set to None to use the full training set
    dimensions: list = field(default_factory=lambda: [
        'fidelity', 'utility', 'diversity', 'privacy', 'consistency', 'stability'
    ])


def build_paths(config: RunConfig) -> dict:
    benchmarks_dir = os.path.join(REPO_ROOT, "benchmarks")
    return {
        "raw_data":       os.path.join(REPO_ROOT, "datasets", f"{config.dataset_name}.csv"),
        "processed_data": os.path.join(benchmarks_dir, "processed", f"{config.dataset_name}_processed.csv"),
        "split_dir":      os.path.join(benchmarks_dir, "splits", config.dataset_name),
        "synthetic_dir":  os.path.join(benchmarks_dir, "synthetic", config.dataset_name, config.model_name),
        "results_dir":    os.path.join(benchmarks_dir, "results", config.dataset_name, config.model_name),
    }


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
    mappings_path = paths["processed_data"].replace(".csv", "_mappings.json")
    if os.path.exists(paths["processed_data"]) and os.path.exists(mappings_path):
        print("Processed data and mappings already exist, skipping preprocessing.")
    else:
        encode_preprocess(paths["raw_data"], paths["processed_data"], config.target_col_raw)

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
    split_dataset(paths["processed_data"], paths["split_dir"],
                  test_size=config.test_size, seed=config.seed)

    train_df = pd.read_csv(os.path.join(paths["split_dir"], "train_full.csv"))
    test_df  = pd.read_csv(os.path.join(paths["split_dir"], "test_full.csv"))
    print(f"\nTrain rows: {len(train_df)}   Test rows: {len(test_df)}")

    if config.max_train_rows is not None and len(train_df) > config.max_train_rows:
        print(f"\n[INFO] Training set ({len(train_df):,} rows) exceeds max_train_rows={config.max_train_rows:,}.")
        train_df = train_df.groupby(target_col, group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, round(config.max_train_rows * len(g) / len(train_df))),
                random_state=config.seed,
            )
        ).reset_index(drop=True)
        print(f"[INFO] Stratified sample applied -> {len(train_df):,} rows retained.")
        print(f"       Class distribution: {train_df[target_col].value_counts().to_dict()}")

    return train_df, test_df, target_col, paths


def save_synthetic(synthetic_df: pd.DataFrame, train_df: pd.DataFrame, paths: dict) -> pd.DataFrame:
    """Align columns to training data, save synthetic CSV, and save a human-readable decoded version."""
    shared_cols = [c for c in train_df.columns if c in synthetic_df.columns]
    synthetic_df = synthetic_df[shared_cols]
    os.makedirs(paths["synthetic_dir"], exist_ok=True)
    synthetic_path = os.path.join(paths["synthetic_dir"], "synthetic.csv")
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"\nGenerated {len(synthetic_df)} synthetic rows")
    print(f"Saved to  : {synthetic_path}")
    print(f"Shape     : {synthetic_df.shape}")
    print(f"\nSample (first 3 rows):\n{synthetic_df.head(3).to_string()}")

    # Decode to human-readable form using mappings saved by encode_preprocess
    mappings_path = paths["processed_data"].replace(".csv", "_mappings.json")
    if os.path.exists(mappings_path):
        with open(mappings_path) as f:
            mappings = json.load(f)

        readable = synthetic_df.copy()

        for col, encoding in mappings["categorical_encodings"].items():
            if col in readable.columns:
                readable[col] = readable[col].astype(int).astype(str).map(encoding)


        readable_path = os.path.join(paths["synthetic_dir"], "synthetic_readable.csv")
        readable.to_csv(readable_path, index=False)
        print(f"\nReadable version saved to: {readable_path}")
        print(f"\nReadable sample (first 3 rows):\n{readable.head(3).to_string()}")

    return synthetic_df


def evaluate(model, config: RunConfig, train_df: pd.DataFrame, synthetic_df: pd.DataFrame,
             target_col: str, paths: dict):
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
    print(f"\nFull report : {os.path.join(paths['results_dir'], f'{config.model_name}_{config.dataset_name}_evaluation_report.json')}")
    print(f"CSV summary : {os.path.join(paths['results_dir'], f'{config.model_name}_{config.dataset_name}_evaluation_summary.csv')}")

    return report
