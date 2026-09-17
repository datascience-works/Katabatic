import os
import sys
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from katabatic.models.tabkde.models import TabKDEModel
from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

config = RunConfig(
    dataset_name="car",
    model_name="tabkde",
    categorical_cols=["buying", "maint", "doors", "persons", "lug_boot", "safety"],
    continuous_cols=[],
    target_col_raw="class",
    constraints=None,
)
train_df, test_df, target_col, paths = preprocess_and_split(config)

configs_to_try = [
    {"name": "baseline",       "diffusion_epochs": 1000, "hidden_dim": 256, "diffusion_steps": 50,  "lr": 1e-3},
    {"name": "fewer_epochs",   "diffusion_epochs": 300,  "hidden_dim": 256, "diffusion_steps": 50,  "lr": 1e-3},
    {"name": "more_steps",     "diffusion_epochs": 1000, "hidden_dim": 256, "diffusion_steps": 200, "lr": 1e-3},
    {"name": "smaller_hidden", "diffusion_epochs": 1000, "hidden_dim": 128, "diffusion_steps": 50,  "lr": 1e-3},
    {"name": "lower_lr",       "diffusion_epochs": 1000, "hidden_dim": 256, "diffusion_steps": 50,  "lr": 5e-4},
]

results = []
for cfg in configs_to_try:
    name = cfg.pop("name")
    print(f"\n{'='*60}\nRunning config: {name} -> {cfg}\n{'='*60}")

    model = TabKDEModel(**cfg)
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

print("\n\nSWEEP COMPLETE")
for r in results:
    print(r["name"], "->", getattr(r["report"], "composite_score", r["report"]))
