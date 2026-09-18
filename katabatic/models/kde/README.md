# KDE

**KDE** (Kernel Density Estimation) is a class-conditional, non-parametric synthetic tabular data generator. It has no training loop and no GPU requirement — every "fit" is a closed-form density estimate rather than an optimization.

***

## Overview

This implementation integrates the following core ideas, **adapted specifically for Katabatic**:

- **Class-conditional modelling**: continuous features are fit with a 1D Gaussian KDE per (feature, class); categorical features use a class-conditional empirical histogram.
- **Target distribution preservation**: the class distribution itself is matched to the real data's class proportions.
- **Pipeline-aware categorical detection**: uses the dataset's `info.json` (`cat_col_idx`) when present to distinguish categorical codes from real continuous values, since Katabatic's pipeline data is usually already integer-encoded — dtype alone can't tell them apart. Falls back to dtype-based detection (`object`/`category`) otherwise.
- **No training instability**: no adversarial loss and no convergence tuning — only depends on `scikit-learn`, already a core Katabatic dependency.

***

## Implementation Details

**Source Implementation**: Adapted from the KDE implementation contributed by **Rishi Goyal** to the Katabatic mentorship repository.

**Repository branch**: https://github.com/katabatic-mentorship/katabatic-mentorship-repo/tree/Rishi_Goyal

**Original model location**: `katabatic/models/kde_Rishi/kde_model.py`

The original detects categorical columns by dtype, which breaks on Katabatic's pipeline data since columns are already integer-encoded there — a categorical code and a real continuous value are indistinguishable by dtype alone. This version accepts an explicit `categorical_cols` set instead, sourced from the dataset's `info.json` where available.

### Original / Reference Implementation Recipe

```text
KDEModel(
    target_col,
    categorical_cols=None,
    kernel="gaussian",
    bandwidth=None,
    random_state=42,
)

fit(df)

For each target class:
    Fit a 1D Gaussian KDE per continuous feature
    Build a class-conditional histogram per categorical feature

generate(n_rows)
```

### Katabatic Implementation

**Training Process**:

```text
1. Identify the target and feature columns.
2. Detect continuous and categorical feature types (info.json cat_col_idx, else dtype).
3. Calculate the target-class distribution.
4. Separate the training data by target class.
5. Fit a 1D Gaussian KDE for each (continuous feature, class) pair.
6. Build an empirical histogram for each (categorical feature, class) pair.
```

**Synthetic Data Generation**:

```text
1. Calculate how many synthetic rows are required for each target class.
2. Sample continuous features from each class's fitted KDE.
3. Sample categorical features from each class's empirical histogram.
4. Round features that were originally integers.
5. Restore the target column.
6. Return the requested number of synthetic rows.
```

***

## Hyperparameter Comparison

| Parameter | Original Implementation | Katabatic |
|---|---|---|
| kernel | gaussian | **gaussian** |
| bandwidth | rule-of-thumb (Scott's rule) | **`None` → Scott's rule per feature, or a fixed float** |
| random_state / seed | 42 | **42** |
| target_col | Required | **Inferred (last column of `train_full.csv`, or `y_train.csv`'s column)** |

Unlike neural-network-based generative models, KDE does not require epochs or batch sizes.

***

## Data Processing

**Categorical features**:
- Detected via the dataset's `info.json` (`cat_col_idx`) when available, else by dtype (`object`/`category`).
- Modeled as a class-conditional empirical histogram; sampling draws from that class's observed value/probability pairs.

**Continuous features**:
- Modeled with a 1D Gaussian KDE (`sklearn.neighbors.KernelDensity`) per class.
- Bandwidth defaults to a per-feature rule-of-thumb (Scott's rule: `std * n^(-1/5)`) when not fixed.
- Columns that were originally integers are rounded back to integer values after sampling.

**Target column**:
- A separate set of per-feature KDEs/histograms is fit for each target class.
- Synthetic class counts follow the target-class proportions in the training data.

***

## Benchmark Results

Evaluated end-to-end via `benchmarks/examples/kde/run_kde_<dataset>.py` against the framework's 6-dimension evaluation pipeline (fidelity, utility, diversity, privacy, consistency, stability), across all 5 standard datasets:

| Dataset | Composite | Fidelity | Utility | Diversity | Privacy | Consistency | Stability |
|---|---|---|---|---|---|---|---|
| car | 0.8435 | 0.9827 | 0.8874 | 0.9993 | 0.4612 | 0.6900 | 0.9825 |
| adult | 0.9179 | 0.9932 | 0.9780 | 0.9588 | 0.9660 | 0.3676 | 0.9950 |
| magic | 0.8916 | 0.9960 | 0.9952 | 0.8676 | 0.4216 | 0.9431 | 1.0000 |
| nursery | 0.8642 | 0.9949 | 0.9136 | 0.9988 | 0.4587 | 0.7731 | 0.9950 |
| shuttle | 0.9081 | 0.9964 | 0.9980 | 0.9486 | 0.5915 | 0.7608 | 1.0000 |

Fidelity, diversity, and stability are consistently strong across all 5. **Privacy is the model's clearest weak point on the low-cardinality categorical datasets** (car, magic, nursery — all 0.42–0.59): class-conditional histograms over a small category space reproduce the training distribution closely enough that this trades off against privacy. `adult`, with higher-cardinality and mixed continuous features, scores far better on privacy (0.97) but noticeably worse on consistency (0.37) — worth keeping in mind before using KDE where either property specifically matters (see PATE-GAN's README for a model built around a differential-privacy guarantee instead).

***

## Installation

No extra dependencies needed — `scikit-learn` is a core dependency:

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

***

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

***

## Limitations

- Continuous KDE samples are not clipped to the real data's observed range — a feature that peaks near zero can occasionally sample a small negative value. Not corrected in this version; flagged here for whoever picks up the Validation & Benchmark pass.
- Per-feature KDEs are independent given the class — cross-feature correlation within a class is not modeled beyond what the shared class label induces.
- No conditional generation on arbitrary feature values yet (only via the class label, same limitation noted in PATE-GAN's README).

***

## Dependencies

The KDE implementation uses:

- NumPy
- pandas
- scikit-learn

The model does not require an external account, API key, external service, or proprietary licence to operate.

***

## References

- Rosenblatt, M. (1956). *Remarks on Some Nonparametric Estimates of a Density Function*. Annals of Mathematical Statistics, 27(3), 832-837.
- Scott, D. W. (1992). *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
- Source implementation by Rishi Goyal: https://github.com/katabatic-mentorship/katabatic-mentorship-repo/tree/Rishi_Goyal

## Generative AI Acknowledgement

AI was used to assist with interpreting and structuring the KDE implementation based on the existing source implementation, established kernel density estimation methodology, and Katabatic model requirements.

All generated content was manually verified, modified, and extended, including code restructuring, Katabatic pipeline integration, debugging, documentation, and experimental validation. The final implementation reflects the author's independent work.
