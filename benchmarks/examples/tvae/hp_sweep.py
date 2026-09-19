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
    {"name": "baseline", "embedding_dim": 128, "epochs": 300, "batch_size": 500},
    {"name": "smaller_embed", "embedding_dim": 64, "epochs": 300, "batch_size": 500},
    {"name": "larger_embed", "embedding_dim": 256, "epochs": 300, "batch_size": 500},
    {"name": "fewer_epochs", "embedding_dim": 128, "epochs": 100, "batch_size": 500},
    {"name": "smaller_batch", "embedding_dim": 128, "epochs": 300, "batch_size": 128},
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

print("\n\nHP/EPOCH SWEEP COMPLETE")
for r in results:
    score = getattr(r["report"], "composite_score", None)
    print(r["name"], "-> composite:", score)
