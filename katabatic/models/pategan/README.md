# PATE-GAN

**PATE-GAN** (Private Aggregation of Teacher Ensembles - Generative Adversarial Network) is a differentially private synthetic data generation model for tabular data.

## Overview

PATE-GAN achieves (ε, δ)-differential privacy by combining:

- **PATE Framework**: Multiple teacher discriminators trained on disjoint data partitions
- **Noisy Aggregation**: Gaussian noise injection into teacher voting for privacy
- **WGAN-GP**: Wasserstein GAN with gradient penalty for stable training
- **Mixed Data Support**: Handles both categorical and continuous features

### Key Features

- ✅ Differential privacy guarantees
- ✅ Handles categorical and continuous features
- ✅ Stable training with WGAN-GP
- ✅ Configurable privacy-utility trade-off
- ✅ Compatible with Katabatic pipeline framework

## Installation

Install PATE-GAN dependencies:

```bash
poetry install -E pategan
```

Or with pip:

```bash
pip install katabatic[pategan]
```

## Quick Start

### Standalone Usage

```python
from katabatic.models.pategan import PATEGAN
import pandas as pd

# Load your data
X_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")

# Initialize model with privacy parameters
model = PATEGAN(
    epsilon=1.0,          # Privacy budget (lower = more private)
    delta=1e-5,          # Privacy parameter
    num_teachers=10,     # Number of teacher discriminators
    niter=10000,         # Training iterations
    batch_size=128,      # Batch size
    random_state=42      # For reproducibility
)

# Train the model
model.fit(X_train, y_train, verbose=1)

# Generate synthetic data
synthetic_data = model.sample(n=1000)
```

### Pipeline Usage (Recommended)

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models.pategan import PATEGAN

# Create pipeline with PATEGAN
pipeline = TrainTestSplitPipeline(
    model=PATEGAN,
    evaluations=None  # Uses default TSTR evaluation
)

# Run complete workflow: split -> train -> evaluate
results = pipeline.run(
    input_csv='data/car.csv',
    output_dir='sample_data/car',
    synthetic_dir='synthetic/car/pategan'
)
```

## Configuration

### Privacy Parameters

- **epsilon** (float, default: 1.0): Privacy budget

  - Lower values = stronger privacy but potentially lower utility
  - Typical range: 0.1 - 10.0
  - Common values: 0.1 (strong), 1.0 (moderate), 10.0 (weak)

- **delta** (float, default: 1e-5): Privacy parameter

  - Should be << 1/n where n is dataset size
  - Typical values: 1e-5 or 1e-6

- **num_teachers** (int, default: 10): Number of teacher discriminators
  - More teachers = better privacy but slower training
  - Range: 5-20
  - For small datasets (<1000 rows): use 5-10
  - For large datasets (>10000 rows): use 10-20

### Training Parameters

- **niter** (int, default: 10000): Number of training iterations

  - More iterations = better quality but longer training
  - Small datasets: 5000-10000
  - Large datasets: 10000-20000

- **batch_size** (int, default: 128): Training batch size

  - Larger batches = faster but more memory
  - Typical values: 64, 128, 256

- **learning_rate** (float, default: 1e-4): Adam optimizer learning rate
- **lambda_gp** (float, default: 10.0): Gradient penalty coefficient for WGAN-GP
- **z_dim** (int, optional): Latent noise dimension (default: n_features // 4)

## Examples

### Example 1: High Privacy

```python
model = PATEGAN(
    epsilon=0.1,      # Strong privacy
    delta=1e-6,
    num_teachers=15,
    niter=15000
)
```

### Example 2: Balanced Privacy-Utility

```python
model = PATEGAN(
    epsilon=1.0,      # Moderate privacy
    delta=1e-5,
    num_teachers=10,
    niter=10000
)
```

### Example 3: Fast Training for Large Datasets

```python
model = PATEGAN(
    epsilon=1.0,
    delta=1e-5,
    num_teachers=10,
    niter=5000,
    batch_size=256
)
```

### Example 4: Custom Training

```python
# Initialize model
model = PATEGAN(epsilon=1.0, delta=1e-5, random_state=42)

# Train on combined X and y
model.fit(X_train, y_train, verbose=1)

# Generate conditional samples (future feature)
synthetic_samples = model.sample(n=500)

# Quick evaluation
results = model.evaluate(X_test, y_test, model='lr')
print(f"Accuracy: {results['accuracy']:.4f}")
```

## Model Contract (Katabatic Framework)

### Inputs

- `dataset_dir/x_train.csv`: Training features
- `dataset_dir/y_train.csv`: Training labels (optional)

### Outputs

- `synthetic_dir/x_synth.csv`: Synthetic features
- `synthetic_dir/y_synth.csv`: Synthetic labels
- `synthetic_dir/metadata.json`: Model metadata, schema, and config

### Schema Fidelity

- Preserves original column order
- Maintains categorical labels (not encoded integers)
- Respects original dtypes (int, float, object)

## Privacy Considerations

### Understanding Differential Privacy

PATE-GAN provides (ε, δ)-differential privacy:

- **ε (epsilon)**: Privacy budget - measures worst-case privacy loss
  - ε ≤ 0.1: Strong privacy
  - ε ≈ 1.0: Moderate privacy
  - ε ≥ 10: Weak privacy
- **δ (delta)**: Probability of privacy breach
  - Should be << 1/n (dataset size)
  - Typical: 1e-5 or 1e-6

### Privacy-Utility Trade-off

- **Stronger Privacy** (low ε):

  - ✅ Better privacy protection
  - ❌ Lower synthetic data utility
  - ❌ More noise in teacher votes

- **Weaker Privacy** (high ε):
  - ✅ Higher synthetic data utility
  - ✅ Less noise in training
  - ❌ Reduced privacy protection

### Recommendations

1. **Start with ε=1.0, δ=1e-5** as baseline
2. **Increase num_teachers** for better privacy (at cost of speed)
3. **More iterations** generally improve quality
4. **Validate** synthetic data utility with TSTR evaluation

## Performance Tips

- **CPU Training**: Model runs on CPU by default (TensorFlow 1.x compatibility mode)
- **Memory**: Reduce `batch_size` if OOM errors occur
- **Speed**: Increase `batch_size` if you have sufficient memory
- **GPU**: Not currently optimized for GPU (uses TF 1.x compatibility)

## Limitations

- Conditional sampling not yet implemented
- Requires TensorFlow 1.x compatibility mode (TF 2.x with v1 behavior)
- Training can be slow for large datasets (10K+ rows)
- Limited to tabular data (no images, text, etc.)

## Evaluation

PATE-GAN includes a built-in `evaluate()` method for quick TSTR testing:

```python
# Train model
model.fit(X_train, y_train)

# Quick evaluation
results = model.evaluate(X_test, y_test, model='lr', task='classification')
print(results)
# Output: {'accuracy': 0.85, 'f1_macro': 0.83}
```

For comprehensive evaluation, use the pipeline with `TSTREvaluation`:

```python
from katabatic.evaluate.tstr.evaluation import TSTREvaluation

evaluator = TSTREvaluation(
    synthetic_dir="synthetic/car/pategan",
    real_test_dir="sample_data/car"
)

results = evaluator.evaluate()
# Tests with LR, MLP, RF, and XGBoost
```

## Troubleshooting

### Import Errors

```
ImportError: Missing required dependencies for PATEGAN: ['tensorflow']
```

**Solution**: Install pategan extras

```bash
poetry install -E pategan
```

### Training Issues

**Slow Training**: Reduce `niter` or increase `batch_size`

**Poor Quality**: Increase `niter`, reduce `epsilon` noise, or tune `num_teachers`

**Memory Issues**: Reduce `batch_size` or sample fewer synthetic records

### TensorFlow Warnings

PATE-GAN uses TensorFlow 1.x compatibility mode. You may see warnings like:

```
WARNING:tensorflow:From ...
```

These are expected and can be safely ignored.

## Reference

**Paper**: "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"  
**Authors**: Jinsung Yoon, James Jordon, Mihaela van der Schaar  
**Year**: 2018  
**Venue**: ICLR 2019

**Original Implementation**: https://bitbucket.org/mvdschaar/mlforhealthlabpub/

## Citation

If you use PATE-GAN in your research, please cite:

```bibtex
@inproceedings{yoon2018pategan,
  title={PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees},
  author={Yoon, Jinsung and Jordon, James and van der Schaar, Mihaela},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## License

PATE-GAN implementation is part of the Katabatic framework and follows the project's MIT license.
