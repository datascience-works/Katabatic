import os
import sys
import time
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

steps_values = [10, 25, 50, 100, 200]

results = []
for n_steps in steps_values:
    print(f"\n{'='*60}\nTraining with diffusion_steps={n_steps}\n{'='*60}")

    start = time.time()
    model = TabKDEModel(diffusion_epochs=300, hidden_dim=256, diffusion_steps=n_steps, lr=5e-4)
    model.train(
        paths["split_dir"],
        categorical_cols=config.categorical_cols,
        continuous_cols=config.continuous_cols,
    )
    train_time = time.time() - start

    sample_start = time.time()
    synthetic_df = model.sample(len(train_df))
    sample_time = time.time() - sample_start

    synthetic_df = save_synthetic(
        synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
    )
    report = evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)

    results.append({
        "steps": n_steps,
        "train_time_sec": round(train_time, 2),
        "sample_time_sec": round(sample_time, 2),
        "composite_score": getattr(report, "composite_score", None),
    })

print("\n\nDIFFUSION STEPS SWEEP COMPLETE")
print(f"{'Steps':<10}{'Train Time (s)':<18}{'Sample Time (s)':<18}{'Composite Score':<18}")
for r in results:
    print(f"{r['steps']:<10}{r['train_time_sec']:<18}{r['sample_time_sec']:<18}{r['composite_score']:<18}")
