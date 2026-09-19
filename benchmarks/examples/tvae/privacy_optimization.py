import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.tvae.models import TVAEModel

config = RunConfig(
    dataset_name="car",
    model_name="tvae",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
)
train_df, test_df, target_col, paths = preprocess_and_split(config)

configs_to_try = [
    {
        "name": "baseline",
        "epochs": 300,
        "l2scale": 1e-5,
        "compress_dims": (128, 128),
        "decompress_dims": (128, 128),
    },
    {
        "name": "strong_l2",
        "epochs": 300,
        "l2scale": 1e-3,
        "compress_dims": (128, 128),
        "decompress_dims": (128, 128),
    },
    {
        "name": "smaller_network",
        "epochs": 300,
        "l2scale": 1e-5,
        "compress_dims": (32, 32),
        "decompress_dims": (32, 32),
    },
    {
        "name": "fewer_epochs",
        "epochs": 100,
        "l2scale": 1e-5,
        "compress_dims": (128, 128),
        "decompress_dims": (128, 128),
    },
    {
        "name": "combined",
        "epochs": 100,
        "l2scale": 1e-3,
        "compress_dims": (32, 32),
        "decompress_dims": (32, 32),
    },
]

results = []
for cfg in configs_to_try:
    name = cfg.pop("name")
    print(f"\n{'=' * 60}\nRunning config: {name} -> {cfg}\n{'=' * 60}")

    model = TVAEModel(**cfg)
    model.train(
        paths["split_dir"],
        categorical_cols=config.categorical_cols,
        continuous_cols=config.continuous_cols,
    )
    synthetic_df = model.sample(len(train_df))
    synthetic_df = save_synthetic(
        synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
    )
    report = evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
    results.append({"name": name, **cfg, "report": report})

print("\n\nPRIVACY OPTIMIZATION SWEEP COMPLETE")
for r in results:
    score = getattr(r["report"], "composite_score", None)
    privacy = (
        getattr(r["report"], "dimension_scores", {}).get("privacy", None)
        if hasattr(r["report"], "dimension_scores")
        else None
    )
    print(r["name"], "-> composite:", score, " privacy:", privacy)
