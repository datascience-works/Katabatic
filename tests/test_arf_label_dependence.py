"""ARF must generate the label jointly with the features."""

import numpy as np
import pandas as pd

from katabatic.models.arf.models import ARFModel


def _training_table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n)
    hours = rng.normal(40, 10, n)
    odds = (age - 40) / 8 + (hours - 40) / 10
    label = np.where(rng.random(n) < 1 / (1 + np.exp(-odds)), "high", "low")
    return pd.DataFrame({"age": age, "hours": hours, "class": label})


def test_synthetic_label_depends_on_features(tmp_path):
    df = _training_table()
    df.to_csv(tmp_path / "train_full.csv", index=False)

    model = ARFModel(verbose=False)
    model.train(str(tmp_path), synthetic_dir=str(tmp_path / "synth"))
    synthetic = model.sample(2000)

    assert synthetic.columns.tolist() == df.columns.tolist()
    real_gap = df.groupby("class")["age"].mean().diff().abs().iloc[-1]
    synth_gap = synthetic.groupby("class")["age"].mean().diff().abs().iloc[-1]
    # An independently sampled label gives a gap close to zero. This ARF
    # keeps only part of the real dependence, so the bound is deliberately low.
    assert synth_gap > 0.08 * real_gap
