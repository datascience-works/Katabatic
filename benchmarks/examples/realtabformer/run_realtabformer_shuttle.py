import os
import sys
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.realtabformer.models import REaLTabFormerModel

config = RunConfig(
    dataset_name="shuttle",
    model_name="realtabformer",
    categorical_cols=[],
    continuous_cols=[
        "time",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
    ],
    target_col_raw="class",
    constraints=None,
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Keep epochs low initially so the complete pipeline can be validated before
# running a longer experiment (matches run_realtabformer_car.py).
EPOCHS = 5
BATCH_SIZE = 8
RANDOM_STATE = 1029
DEVICE = "cpu"
N_CRITIC = 5

print("\n" + "=" * 60)
print("STEP 3 — Train REaLTabFormer")
print("=" * 60)

print(f"Training rows : {len(train_df)}")
print(f"Epochs        : {EPOCHS}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Device        : {DEVICE}")
print(f"n_critic      : {N_CRITIC}")

model = REaLTabFormerModel(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    random_state=RANDOM_STATE,
    device=DEVICE,
    n_critic=N_CRITIC,
)

start_time = time.perf_counter()

model.train(
    paths["split_dir"],
    synthetic_dir=paths["synthetic_dir"],
)

training_time = time.perf_counter() - start_time

print("\nREaLTabFormer training complete.")
print(f"Training + initial generation time: {training_time:.2f} seconds")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)

sample_start = time.perf_counter()

synthetic_df = model.sample(len(train_df))

sampling_time = time.perf_counter() - sample_start

print(f"Generated rows : {len(synthetic_df)}")
print(f"Sampling time  : {sampling_time:.2f} seconds")

synthetic_df = save_synthetic(
    synthetic_df,
    train_df,
    paths,
    categorical_cols=config.categorical_cols,
)

print("\n" + "=" * 60)
print("STEP 5 — Evaluate")
print("=" * 60)

evaluate(
    model,
    config,
    train_df,
    synthetic_df,
    target_col,
    paths,
    test_df,
)
