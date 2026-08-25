# KDE

**KDE** (Kernel Density Estimation) is a class-conditional, non-parametric synthetic tabular data generator. It has no training loop and no GPU requirement — every "fit" is a closed-form density estimate.

## Overview

For each class in the target column:

- **Continuous features**: fit a 1D Gaussian KDE (`sklearn.neighbors.KernelDensity`) per (feature, class).
- **Categorical features**: use a class-conditional empirical histogram.
- **Class distribution**: matched to the real data's class proportions.

Sampling draws a class per synthetic row (proportional to real class frequencies), then draws each feature independently from that class's fitted KDE or histogram.

### Key Features

- No training instability — no adversarial loss, no convergence tuning
- Handles mixed categorical and continuous features
- Uses `info.json` (`cat_col_idx`) when present to distinguish categorical codes from real continuous values, since Katabatic's pipeline data is usually already integer-encoded — dtype alone can't tell them apart
- Falls back to dtype-based detection (object/category dtype) when no `info.json` is available
- Only depends on `scikit-learn`, already a core Katabatic dependency — no extra install required

## Installation

No extra needed — `scikit-learn` is a core dependency:

```bash
poetry install
```

## Quick Start

### Standalone Usage

```python
from katabatic.models.kde import KDESynthesizer

model = KDESynthesizer(kernel="gaussian", bandwidth=None, seed=42)
model.train(data_dir="sample_data/car", synthetic_dir="synthetic/car/kde")

synthetic_df = model.sample(n=1000)
```

### Pipeline Usage (Recommended)

```python
from katabatic.artifacts import LocalArtifactStore
from katabatic.models.kde import KDESynthesizer
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

store = LocalArtifactStore("artifacts")
pipeline = TrainTestSplitPipeline(model=KDESynthesizer())

results = pipeline.run(
    input_csv="raw_data/car.csv",
    dataset_name="car",
    artifact_store=store,
    model_name="kde",
)
# results["model_ref"], results["evaluation_refs"] — TSTR metrics on disk
```

## Configuration

- **kernel** (str, default: `"gaussian"`): kernel passed to `sklearn.neighbors.KernelDensity`.
- **bandwidth** (float or `None`, default: `None`): fixed bandwidth for every continuous feature's KDE. If `None`, uses a per-feature rule-of-thumb (Scott's rule: `std * n^(-1/5)`).
- **seed** (int, default: `42`): random seed for both KDE sampling and categorical resampling.

## Model Contract (Katabatic Framework)

### Inputs

`train(data_dir, synthetic_dir=None)` reads, in order of preference:

- `data_dir/train_full.csv` (target assumed to be the last column), or
- `data_dir/x_train.csv` + `data_dir/y_train.csv` (target is `y_train.csv`'s single column)

If `data_dir/info.json` exists with a `cat_col_idx` key (Katabatic's dataset-registry convention), those column positions are treated as categorical regardless of dtype. Otherwise categorical columns are detected by dtype (`object` or `category`).

### Outputs

- `synthetic_dir/x_synth.csv`: synthetic features
- `synthetic_dir/y_synth.csv`: synthetic labels
- `synthetic_dir/metadata.json`: schema, detected categorical columns, and training config

## Limitations

- Continuous KDE samples are not clipped to the real data's observed range — a feature that peaks near zero can occasionally sample a small negative value. Not corrected in this version; flagged here for whoever picks up the Validation & Benchmark pass.
- Per-feature KDEs are independent given the class — cross-feature correlation within a class is not modeled beyond what the shared class label induces.
- No conditional generation on arbitrary feature values yet (only via the class label, same limitation noted in PATE-GAN's README).

## Reference

Ported from the `katabatic-mentorship/katabatic-mentorship-repo` registry, `Rishi_Goyal` branch (`katabatic/models/kde_Rishi/kde_model.py`), and adapted to Katabatic's `Model` interface, artifact I/O conventions, and `info.json`-aware categorical detection.

## License

KDE implementation is part of the Katabatic framework and follows the project's MIT license.
