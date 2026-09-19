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

epoch_values = [100, 300, 500, 1000, 2000]

results = []
for n_epochs in epoch_values:
    print(f"\n{'='*60}\nTraining with diffusion_epochs={n_epochs}\n{'='*60}")

    start = time.time()
    model = TabKDEModel(diffusion_epochs=n_epochs, hidden_dim=256, diffusion_steps=50, lr=5e-4)
    model.train(
        paths["split_dir"],
        categorical_cols=config.categorical_cols,
        continuous_cols=config.continuous_cols,
    )
    train_time = time.time() - start

    synthetic_df = model.sample(len(train_df))
    synthetic_df = save_synthetic(
        synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
    )
    report = evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)

    results.append({
        "epochs": n_epochs,
        "train_time_sec": round(train_time, 2),
        "composite_score": getattr(report, "composite_score", None),
    })

print("\n\nEPOCH SWEEP COMPLETE")
print(f"{'Epochs':<10}{'Train Time (s)':<18}{'Composite Score':<18}")
for r in results:
    print(f"{r['epochs']:<10}{r['train_time_sec']:<18}{r['composite_score']:<18}")
