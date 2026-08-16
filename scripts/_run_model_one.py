"""Train + sample + TSTR-evaluate one model on one dataset.

Env vars:
  MODEL    - one of: gmm, bayesian_network, tabdiff, tabmt
  DATASET  - one of: car, magic, nursery, shuttle, adult
  SMOKE    - "1" (default) for a fast low-fidelity correctness check
"""
from __future__ import annotations

import os

from katabatic.evaluate.tstr.evaluation import TSTREvaluation

MODEL_CLASSES = {
    "gmm": ("katabatic.models.gmm.models", "Gmm"),
    "bayesian_network": ("katabatic.models.bayesian_network.models", "BayesianNetworkModel"),
    "tabdiff": ("katabatic.models.tabdiff.models", "Tabdiff"),
    "tabmt": ("katabatic.models.tabmt.models", "Tabmt"),
}

SMOKE_CONFIG = {
    "tabdiff": dict(steps=150, num_timesteps=60),
    "tabmt": dict(steps=150),
}

model_name = os.environ.get("MODEL", "gmm")
dataset = os.environ.get("DATASET", "car")
smoke = os.environ.get("SMOKE", "1") == "1"

module_path, class_name = MODEL_CLASSES[model_name]
module = __import__(module_path, fromlist=[class_name])
ModelClass = getattr(module, class_name)

dataset_dir = f"sample_data/{dataset}"
synthetic_dir = f"synthetic/{dataset}/{model_name}"

if model_name in ("gmm", "bayesian_network"):
    model = ModelClass()
    model.train(dataset_dir, synthetic_dir=synthetic_dir)
else:
    config = SMOKE_CONFIG.get(model_name) if smoke else None
    model = ModelClass(config=config)
    model.train(dataset_dir, synthetic_dir=synthetic_dir)

evaluation = TSTREvaluation(synthetic_dir=synthetic_dir, real_test_dir=dataset_dir)
results = evaluation.evaluate()
print(f"[{model_name}/{dataset}] synthetic data written to {synthetic_dir}")
print(f"[{model_name}/{dataset}] TSTR evaluation result: {results}")
