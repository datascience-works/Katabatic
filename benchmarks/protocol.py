"""Shared benchmark protocol: seeds, splits and training budgets.

Single source of truth for the rules in benchmarks/protocol.md. Benchmark
scripts can import the constants, and benchmarks/check_protocol.py compares
every script against ``MODEL_SPECS``.

DRAFT: the epoch numbers and size groups are proposals that still need
approval from the stream lead.
"""

from dataclasses import dataclass

# Data split (same values as RunConfig in benchmarks/runner.py)
SEED = 42
TEST_SIZE = 0.2
MAX_TRAIN_ROWS = 50_000

# Dataset size groups, by number of rows in the packaged CSV
DATASET_SIZE = {
    "car": "small",
    "nursery": "medium",
    "magic": "medium",
    "adult": "large",
    "shuttle": "large",
}

# Standard training epochs for models on the "standard" policy
EPOCHS_BY_SIZE = {
    "small": 100,
    "medium": 100,
    "large": 150,
}

# Budget policies
STANDARD = "standard"  # epochs must equal epochs_for(dataset)
SPECIFIC = "specific"  # model-specific budget, kept and documented
FIXED = "fixed"  # no training loop, default settings are kept


@dataclass(frozen=True)
class ModelSpec:
    """What the protocol expects from the scripts of one model."""

    policy: str
    budget_params: tuple = ()  # keyword names that set the training budget
    budget_default: int | None = None  # value used when a script sets none
    seed_params: tuple = ()  # keyword names that set the random seed
    seed_default: int | None = None  # seed used when a script passes none
    seed_supported: bool = True  # False if the model has no seed argument
    note: str = ""


MODEL_SPECS = {
    "arf": ModelSpec(FIXED, seed_params=("seed",)),
    "codi": ModelSpec(
        STANDARD,
        budget_params=("epochs",),
        seed_params=("random_state",),
        seed_default=42,
    ),
    "ctgan": ModelSpec(STANDARD, ("epochs",), seed_params=("seed",)),
    "fairtabdiffusion": ModelSpec(
        STANDARD,
        ("epochs",),
        seed_params=("seed",),
        note="Was 200 epochs; now follows the standard.",
    ),
    "ganblr": ModelSpec(
        STANDARD,
        ("epochs",),
        150,
        seed_params=("seed",),
        note="Seed applies to sampling only; training has no seed argument.",
    ),
    "gmm": ModelSpec(FIXED, seed_params=("seed", "random_state")),
    "great": ModelSpec(
        SPECIFIC,
        ("epochs", "EPOCHS"),
        100,
        seed_params=("seed",),
        note="Seed applies to sampling. Adult 310 vs 2 epochs elsewhere (deliverable 13).",
    ),
    "histogram": ModelSpec(FIXED, seed_params=("seed", "random_state")),
    "kde": ModelSpec(FIXED, seed_params=("seed",)),
    "medgan": ModelSpec(
        SPECIFIC,
        ("ae_pretrain_epochs", "gan_epochs"),
        seed_params=("random_state",),
        note="Two stages: 100 pre-train, 1000 GAN.",
    ),
    "mst": ModelSpec(FIXED, seed_supported=False, note="Model has no seed argument."),
    "naivebayes": ModelSpec(FIXED, seed_params=("seed",)),
    "pategan": ModelSpec(
        SPECIFIC,
        ("niter",),
        seed_params=("random_state",),
        note="Iterations, not epochs.",
    ),
    "privtree": ModelSpec(FIXED, seed_params=("seed", "random_state")),
    "realtabformer": ModelSpec(
        SPECIFIC,
        ("epochs", "EPOCHS"),
        100,
        seed_params=("random_state",),
        seed_default=1029,
        note="Seed set to 42; the model default is 1029.",
    ),
    "smote": ModelSpec(FIXED, seed_params=("random_state",)),
    "synthpop": ModelSpec(FIXED, seed_params=("seed",)),
    "tabddpm": ModelSpec(
        SPECIFIC,
        ("steps",),
        seed_params=("seed",),
        seed_default=42,
        note="Training steps, not epochs. Model default seed is 42.",
    ),
    "tabebm": ModelSpec(
        SPECIFIC, ("sgld_steps",), seed_params=("seed",), note="Sampling steps."
    ),
    "tabkde": ModelSpec(
        FIXED,
        seed_params=("random_state",),
        note="Seed is now passed; the model default is no seed.",
    ),
    "tabsyn": ModelSpec(
        SPECIFIC,
        ("diffusion_epochs",),
        500,
        seed_params=("seed",),
        seed_default=42,
        note="Scripts use model defaults (50 decoder, 500 diffusion epochs).",
    ),
    "tvaegan": ModelSpec(
        STANDARD,
        ("epochs",),
        50,
        seed_params=("seed",),
        seed_default=42,
        note="Was 50 epochs by default and 200 on two datasets; now standard.",
    ),
}


def epochs_for(dataset_name: str) -> int:
    """Return the standard number of training epochs for a dataset."""
    return EPOCHS_BY_SIZE[DATASET_SIZE[dataset_name]]
