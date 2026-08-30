import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from runner import RunConfig, evaluate, preprocess_and_split, save_synthetic

from katabatic.models.great.models import GReaT

config = RunConfig(
    dataset_name="adult",
    model_name="great",
    categorical_cols=[
        "workclass",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native-country",
    ],
    continuous_cols=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
    target_col_raw="income",
    constraints={
        "age": (17, 90),
        "fnlwgt": (12285, 1490400),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
        "hours-per-week": (1, 99),
    },
)

train_df, test_df, target_col, paths = preprocess_and_split(config)

# Training hyperparameters
LLM = "gpt2"  # HuggingFace model checkpoint
EPOCHS = 2  # fine-tuning epochs (keep low for eval; raise for quality)
BATCH_SIZE = 2  # per-device training batch size
EXPERIMENT_DIR = "trainer_great_adult"  # output dir for HuggingFace Trainer checkpoints
EFFICIENT_FINETUNING = ""  # "" = full fine-tune; "lora" = LoRA (requires peft)
FLOAT_PRECISION = None  # decimal places for floats in text encoding; None = full

# Sampling hyperparameters
TEMPERATURE = 0.7  # generation temperature (lower = more conservative)
MAX_LENGTH = 100  # max tokens per generated row
K = 100  # rows attempted per generation batch
DEVICE = "cuda"  # "cpu" or "cuda"
GUIDED_SAMPLING = False  # True = feature-by-feature (slower, sometimes more reliable)
RANDOM_FEATURE_ORDER = True  # shuffle column order in guided sampling prompts
DROP_NAN = False  # drop rows with any NaN in the output
SEED = config.seed  # generation seed for reproducibility

print("\n" + "=" * 60)
print("STEP 3 — Train GReaT")
print("=" * 60)
model = GReaT(
    llm=LLM,
    experiment_dir=EXPERIMENT_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    efficient_finetuning=EFFICIENT_FINETUNING,
    float_precision=FLOAT_PRECISION,
)
model.train(
    paths["split_dir"],
    categorical_cols=config.categorical_cols,
    continuous_cols=config.continuous_cols,
)
print("\nGReaT training complete.")

print("\n" + "=" * 60)
print("STEP 4 — Generate synthetic data")
print("=" * 60)
synthetic_df = model.sample(
    len(train_df),
    temperature=TEMPERATURE,
    max_length=MAX_LENGTH,
    k=K,
    device=DEVICE,
    guided_sampling=GUIDED_SAMPLING,
    random_feature_order=RANDOM_FEATURE_ORDER,
    drop_nan=DROP_NAN,
    seed=SEED,
)
synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
