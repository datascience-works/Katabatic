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

# Training hyperparameters — matched to paper (Borisov et al., 2022, Appendix C)
LLM = "gpt2"             # full GReaT variant — 355M params
EPOCHS = 310             # paper: 310 epochs for full GReaT on Adult
BATCH_SIZE = 128          
EXPERIMENT_DIR = "trainer_great_adult"
EFFICIENT_FINETUNING = ""
FLOAT_PRECISION = None

# Sampling hyperparameters
TEMPERATURE = 0.7        # paper: T=0.7 for all experiments
MAX_LENGTH = 100
K = 100
DEVICE = "cuda"
GUIDED_SAMPLING = False
RANDOM_FEATURE_ORDER = True
DROP_NAN = False

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
    save_steps=100000,   # to avoid disk space errors
)

# MUST use fit(), not train()
# model.train() triggers pipeline mode which silently overrides epochs to 2
model.fit(train_df)

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
)

synthetic_df = save_synthetic(
    synthetic_df, train_df, paths, categorical_cols=config.categorical_cols
)

evaluate(model, config, train_df, synthetic_df, target_col, paths, test_df)
