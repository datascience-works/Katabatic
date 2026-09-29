# REaLTabFormer Model

## Model Overview

REaLTabFormer (REalistic and Accurate Learning of TABular data using TransFORMERs) is a transformer-based model for generating synthetic tabular data.

The model uses an autoregressive transformer architecture to learn the structure and relationships present in real tabular datasets. Instead of generating each column independently, REaLTabFormer learns patterns across columns and uses the learned representation to generate new synthetic rows that resemble the original dataset.

REaLTabFormer supports both tabular and relational data generation. The current Katabatic implementation focuses on the tabular model only.

This integration is currently experimental within Katabatic.

---

### Key Idea

REaLTabFormer is used as a tabular synthetic data generator with the following model interface:

1. `train(...)` loads the real training data and trains the REaLTabFormer tabular model.

2. `sample(n)` generates `n` new synthetic rows from the trained model.

3. During the Katabatic training workflow, the model generates the same number of synthetic rows as the original training dataset.

4. The generated features and target values may be saved as:

   - `x_synth.csv`

   - `y_synth.csv`

   - `metadata.json`

---

### Research Paper

REaLTabFormer is based on:

> A. V. Solatorio and O. Dupriez,
>
> **“REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers,”**
>
> 2023.
>
> arXiv:2302.02041

---

### Parameters Kept from the Original Method

The Katabatic implementation exposes several parameters from REaLTabFormer, including:

- number of training epochs;

- training batch size;

- random seed;

- gradient accumulation steps;

- logging interval;

- training and sampling device;

- additional arguments that can be passed to the underlying `fit()` method.

The wrapper uses REaLTabFormer's `tabular` model type.

---

### Project-Specific Interpretation

Within Katabatic, REaLTabFormer is used as an experimental synthetic tabular data generator.

The wrapper adapts REaLTabFormer to Katabatic's existing model interface and dataset structure. Training data can be provided as either a complete training dataset or as separate feature and target files.

The final column of the combined training data is treated as the target column. After synthetic data is generated, Katabatic separates the generated features and target values and saves them using the project's standard synthetic output format.

CPU is used as the default device in this integration to provide safer execution across local development and CI environments.

---

## Approach

The synthetic data generation pipeline is:

1. Load the real training dataset.

2. If `train_full.csv` is available, load the complete training dataset directly.

3. Otherwise, load `x_train.csv` and `y_train.csv` and combine them into a single training DataFrame.

4. Validate that the training dataset is not empty and that the expected columns are valid.

5. Treat the final column as the target column.

6. Initialise REaLTabFormer using `model_type="tabular"` and the configured training parameters.

7. Train the REaLTabFormer model on the complete training DataFrame.

8. Use the trained model to generate synthetic rows.

9. Separate the generated target column from the generated feature columns.

10. Save the generated features, target values and metadata using Katabatic's synthetic output structure.

---

### Convergence Criteria

REaLTabFormer training is controlled primarily by the configured number of training epochs.

The current Katabatic wrapper does not implement an additional project-specific convergence criterion. Training behaviour is handled by the underlying REaLTabFormer implementation and the supplied training arguments.

---

## Hyperparameters

The main parameters exposed by the Katabatic REaLTabFormer wrapper are:

| Hyperparameter | Default Value | Description |
|---|---:|---|
| `epochs` | `100` | Number of training epochs. |
| `batch_size` | `8` | Batch size used during model training. |
| `random_state` | `1029` | Random seed used by REaLTabFormer. |
| `device` | `"cpu"` | Device used for training and sampling. |
| `gradient_accumulation_steps` | `1` | Number of gradient accumulation steps used during training. |
| `logging_steps` | `100` | Logging interval used during training. |
| `fit_kwargs` | `None` | Optional additional arguments passed to REaLTabFormer's `fit()` method. |
| `n_critic` | `5` | Interval between critic/sensitivity assessments during REaLTabFormer training. |
---

### Expected Files

Following the project template:

- `x_train.csv`

- `y_train.csv`

Or alternatively:

- `train_full.csv`

When separate feature and target files are provided, `y_train.csv` must contain exactly one target column.

---

### Numerical Features

REaLTabFormer can model numerical columns as part of a tabular dataset.

Numerical values are handled by the underlying REaLTabFormer preprocessing and generation workflow rather than requiring the Katabatic wrapper to implement a separate numerical-only generation method.

---

### Categorical Features

REaLTabFormer can also model categorical columns within tabular datasets.

This allows the model to work with datasets containing numerical, categorical or mixed feature types.

The current Katabatic wrapper passes the complete training DataFrame to REaLTabFormer and relies on the underlying model to process the tabular columns.

---

## Output

Generated files:

- `x_synth.csv`

- `y_synth.csv`

- `metadata.json`

The metadata records information including the model name, target column, number of generated rows and model parameters.

---

## Strengths

- Designed specifically for realistic synthetic tabular data generation.

- Uses a transformer-based architecture capable of learning relationships between columns.

- Can work with numerical and categorical tabular data.

- Supports configurable training and sampling behaviour.

- Integrates with Katabatic's existing model interface and synthetic output structure.

- Provides reproducibility through a configurable random seed.

---

## Limitations

- Transformer-based training can be computationally expensive compared with simpler synthetic data generation methods.

- CPU is used by default in the Katabatic wrapper, which may result in longer training times for larger datasets or higher epoch counts.

- Only REaLTabFormer's tabular mode is currently integrated.

- REaLTabFormer's relational data generation functionality is not currently supported by the Katabatic wrapper.

- The current `evaluate()` implementation returns a placeholder value and is expected to be supplemented by Katabatic's benchmarking and evaluation pipeline.

- This integration is experimental and requires further integration and benchmark testing before it can be promoted to an officially supported Katabatic model.

---

## Installation

REaLTabFormer is integrated into Katabatic as an optional model dependency.

From the Katabatic repository root, install the required dependencies using:

```bash
poetry install -E realtabformer

---

## Usage

### Project Model Interface

```python
from katabatic.models.realtabformer.models import REaLTabFormerModel

model = REaLTabFormerModel(
    epochs=100,
    batch_size=8,
    random_state=1029,
    device="cpu",
    n_critic=5,
)

model.train(
    data_dir=paths["split_dir"],
    synthetic_dir=paths.get("synthetic_dir"),
)

synthetic_data = model.sample(1000)
```

---

### Evaluation Pipeline Benchmark Scripts

The current REaLTabFormer benchmark runner is:

- `benchmarks/examples/realtabformer/run_realtabformer_car.py`

The Car benchmark uses the tabular REaLTabFormer model with:

- epochs: `5`
- batch size: `8`
- random state: `1029`
- device: `cpu`
- n_critic: `5`
---

## Model Evaluation Benchmarks Results

### Car Dataset

A real end-to-end benchmark was completed using the Car Evaluation dataset.

| Metric | Score |
|---|---:|
| Fidelity | 0.9830 |
| Utility | 0.9802 |
| Diversity | 0.9992 |
| Privacy | 0.4539 |
| Consistency | 0.8499 |
| Stability | 0.9905 |
| Composite | 0.8913 |

Training and initial generation took approximately 128 seconds on CPU.
Sampling 1,382 synthetic rows took approximately 1.6 seconds.

The model achieved strong fidelity, utility, diversity and stability on this
dataset. Privacy evaluation was weaker, with an exact duplicate rate of
81.91%. This result is specific to the small, fully categorical Car dataset
and should not be treated as a general privacy conclusion for REaLTabFormer.
Additional datasets should be evaluated for broader validation.

### Composite Score

`0.8913`

### Dimension Scores

- fidelity: `0.9830`
- utility: `0.9802`
- diversity: `0.9992`
- privacy: `0.4539`
- consistency: `0.8499`
- stability: `0.9905`

---

## Model Performance Benchmarks Results

### Car Dataset

- Training + initial generation time: approximately `127.99 seconds`
- Synthetic sampling time: approximately `1.60 seconds`
- Training rows: `1,382`
- Synthetic rows generated: `1,382`
- Epochs: `5`
- Batch size: `8`
- Device: `CPU`
- `n_critic`: `5`

The benchmark completed successfully using the full Katabatic evaluation pipeline. All five stability runs completed successfully, producing a stability score of `0.9905`.

The privacy score was lower than the other dimensions (`0.4539`), with an exact duplicate rate of `81.91%`. This result is specific to the Car dataset and should be investigated further using additional datasets.

---

## Experimental Status

REaLTabFormer is registered in Katabatic's `ModelRegistry` as:

```text
realtabformer
```

The model is currently registered with:

```text
supported = False
```

This means the integration is experimental and has not yet completed Katabatic's model promotion process.

---

## References

1. Solatorio, A. V., & Dupriez, O. (2023). **REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers.** arXiv:2302.02041.

2. REaLTabFormer open-source repository and documentation.
