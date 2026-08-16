"""Train + sample TabDDPM on one dataset, then run TSTR evaluation.

Driven by env vars so scripts/run_tabddpm.sh can loop over datasets:
  DATASET  - one of car, magic, nursery, shuttle, adult
  SMOKE    - "1" for a fast, low-fidelity correctness check (default),
             "0" for full-config training.
"""
from __future__ import annotations

import os

from katabatic.models.tabddpm.models import Tabddpm
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

dataset = os.environ.get("DATASET", "car")
smoke = os.environ.get("SMOKE", "1") == "1"

dataset_dir = f"sample_data/{dataset}"
synthetic_dir = f"synthetic/{dataset}/tabddpm"

config = None
if smoke:
    config = dict(
        steps=200,
        num_timesteps=100,
        batch_size=32,
        use_ema=False,
        d_layers=(64, 64),
        eval_batches=3,
    )

model = Tabddpm()
model.train(dataset_dir, synthetic_dir=synthetic_dir, config=config)

evaluation = TSTREvaluation(
    synthetic_dir=synthetic_dir,
    real_test_dir=dataset_dir,
)
results = evaluation.evaluate()
print(f"[{dataset}] synthetic data written to {synthetic_dir}")
print(f"[{dataset}] TSTR evaluation result: {results}")
