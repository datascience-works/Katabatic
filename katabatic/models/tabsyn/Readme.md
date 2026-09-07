# TabSyn

## Model Overview

By training a diffusion model on a latent representation acquired by a Variational Autoencoder, TabSyn creates synthetic tabular data (mixed numerical and categorical columns). The architecture outlined in the original TabSyn paper  (Zhang et al., ICLR 2024) is followed in this implementation: a Transformer-based VAE encoder and decoder that are trained in conjunction with a diffusion model on the resulting latent space.

### Key Idea

Data generation is divided into two phases by TabSyn. Using a Transformer-based encoder that uses self-attention to record associations between columns and a corresponding Transformer decoder that reconstructs the original row, a Variational Autoencoder first learns to compress each row into a latent representation. Second, a diffusion model learns to reverse a noise-adding process in order to create new latent vectors, which are subsequently decoded back into realistic tabular rows.

An earlier, simpler version that used a frozen, randomly initialised encoder (never trained) with no appropriate VAE loss was rebuilt into this implementation. In line with the design of the study, it now incorporates a trainable Transformer-based VAE with the reparameterization technique and a KL-divergence loss weighted by a scheduled beta (Appendix D.1, Section 4.4).

### Research Paper
**Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space**

Hengrui Zhang, Jiani Zhang, Balasubramaniam Srinivasan, Zhengyuan Shen, 
Xiao Qin, Christos Faloutsos, Huzefa Rangwala, George Karypis
ICLR 2024 (Oral Presentation)

Code: https://github.com/amazon-science/tabsyn
Paper: https://arxiv.org/abs/2310.09656

**What was kept the same, what was changed:**

This implementation adheres to Appendix D.1 and Section 4.4's architecture, which consists of a 2-layer Transformer VAE encoder and decoder with single-head self-attention, a reparameterization trick for latent sampling, and a KL-divergence loss weighted by a scheduled beta (beta_max=0.01, beta_min=1e-5, decay=0.7). This is a rebuild of an earlier version that used a frozen, untrained encoder with no proper VAE loss.


## Approach

1. **Tokenize**: each column becomes a d-dimensional token, one token per 
   column (numeric columns via a linear projection, categorical columns 
   via embeddings).

2. **Encode and decode (VAE)**: a Transformer-based encoder produces a 
   mean and log-variance for each token, a latent vector is sampled via 
   the reparameterization trick, and a matching Transformer decoder 
   reconstructs the original columns from that latent.

3. **Diffusion**: a diffusion model is trained on the VAE's latent 
   vectors, learning to generate new latents by reversing a noise 
   process. New rows are produced by sampling noise, denoising it, and 
   decoding the result.

### Training Details

A combined reconstruction loss (MSE for numeric columns, cross-entropy for categorical columns) with a KL-divergence term weighted by a scheduled beta are used to jointly train the tokeniser, encoder, and decoder. The final latents for diffusion training are then calculated using the mean output of the trained encoder.

### Convergence Criteria

Both stages use early stopping via `patience`. Testing found that longer training combined with higher patience, closer to the paper's actual training regime of roughly 10,000 epochs and patience 500, substantially improved results and resolved a mode collapse issue seen under shorter training.

## Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `d_token` | 4 | Matches paper (Appendix G.1) |
| `decoder_epochs` | 50 | VAE training epochs, tokenizer, encoder, decoder trained jointly |
| `decoder_batch_size` | 2048 | Not tuned |
| `diffusion_epochs` | 500 | Ceiling only, testing showed 2000 significantly improves results over the default |
| `diffusion_batch_size` | 4096 | Not tuned |
| `diffusion_hidden_dim` | 1024 | Matches paper |
| `diffusion_steps` | 15 | Matches paper's recommendation, under 20 for optimal results |
| `lr` | 1e-3 | Not tuned |
| `weight_decay` | 0.0 | Not tuned |
| `patience` | 20 | Testing showed 500 significantly improves results when combined with more diffusion_epochs |

**Best verified configuration (Car dataset):**
```python
TabSyn(diffusion_epochs=2000, patience=500)
```
Composite score: 0.8224 (fidelity 0.96, utility 0.94, diversity 0.997, consistency 0.85), with healthy label diversity and no mode collapse.

## Input
- `X`: Tabular feature matrix (numeric and/or categorical columns)
- `y`: Target labels

### Expected Files
TabSyn expects NumPy files in `data_dir`, not CSVs directly:
- `X_num_train.npy`, `X_cat_train.npy`, `y_train.npy`
- `X_num_test.npy`, `X_cat_test.npy`, `y_test.npy`
- `info.json` describing task type and column indices

If a dataset's split directory only has CSVs (`x_train.csv`, `y_train.csv`, etc.), use `benchmarks/examples/tabsyn/prepare_npy.py` to generate the required `.npy` files and `info.json` before training.

## Preprocessing
### Numerical Features
Numeric columns are standardized (mean 0, standard deviation 1) before training. The original mean and standard deviation are stored and used to convert generated values back to real-world scale during sampling.

### Categorical Features
Categorical columns are encoded to integer indices using sklearn's `LabelEncoder`, one encoder per column. The fitted encoders are stored and used to decode generated indices back into their original category strings during sampling.

## Label Handling
The target column is treated as a categorical column internally and is placed first among the categorical columns (before the actual feature columns) when building the model's internal data layout. This ordering matters when writing code that renames or reorders TabSyn's synthetic output, since the target must be positioned to match this internal layout, not assumed to be last like in the original real data.

## Output
Generated files (when using `train()`):
- `x_synth.csv`
- `y_synth.csv`

When calling `sample()` directly, output columns are generically named (`num_0`, `num_1`, ..., `cat_0`, `cat_1`, ...) and must be renamed to match the real dataset's column order before use. See `benchmarks/examples/tabsyn/run_tabsyn_car.py` for a working example of this rename logic.

## Evaluation
`evaluate()` returns a reconstruction loss (a blend of MSE fornumeric 
columns and cross entropy for categorical columns), not accuracy or F1. Lower is better.

The full evaluation used for validation is `SyntheticEvaluationPipeline` 
(`katabatic/pipeline/evaluation_pipeline.py`), which reports 6 dimensions: 
fidelity, utility, diversity, privacy, consistency, and stability, combined into a single composite score.

## Strengths
- Handles mixed numeric and categorical columns in one unified pipeline
- Fast to train and sample compared to full diffusion-in-data-space methods
- Once bugs were fixed, produces strong fidelity scores (JSD, Wasserstein) 
  on tested datasets

## Limitations

- Architecture is a reimplementation based on the paper's description.
- Mode collapse can occur under short training; resolved in testing by increasing diffusion_epochs and patience closer to the paper's regime, not yet confirmed whether the defaults should simply change to reflect this
- Only validated on the Car dataset with this rebuilt architecture.

## Installation
```bash
poetry install --extras tabsyn
```
## Usage
Benchmark scripts for each dataset:
- Car: [benchmarks/examples/tabsyn/run_tabsyn_car.py](benchmarks/examples/tabsyn)
- Adult: [benchmarks/examples/tabsyn/run_tabsyn_adult.py](benchmarks/examples/tabsyn)

```python
from katabatic.models.tabsyn.models import TabSyn

model = TabSyn(diffusion_epochs=2000, patience=500)

model.train(
    data_dir="path_to_data",
    synthetic_dir="path_to_save"
)

X_synth, y_synth = model.sample(1000)
```

## Model Evaluation Benchmarks Results

#### Car Dataset (diffusion_epochs=2000, patience=500)

Composite score: 0.8224

Dimension scores:
- fidelity       0.9602
- utility        0.9377
- diversity      0.9970
- privacy        0.4655
- consistency    0.8466
- stability      not available, shared pipeline bug prevents this from running

Note: previous results using the old, simplified architecture (frozen encoder, tuned hyperparameters) are no longer current and have been removed. 

## Model Performance Benchmarks Results
Training and inference timing not yet measured for this model.Diffusion training typically stops early via early stopping, usually well under 100 epochs on the datasets tested so far.
