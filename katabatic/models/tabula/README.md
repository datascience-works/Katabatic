# TabuLa Model

## Model Overview
TabuLa (Tabular Language) is a language model-based synthetic tabular data generator that treats each table row as a natural language sequence and fine-tunes a compact GPT-2 variant to generate new rows autoregressively.

---

### Key Idea
TabuLa applies autoregressive language modelling to tabular data. Each row is serialized into a text string (e.g., `"age 35, income 52000, education bachelors, ..."`) with column order randomly shuffled per row to prevent the model from learning spurious positional dependencies. A distilGPT-2 model with randomly initialized weights (no pretrained knowledge) is then fine-tuned on these sequences. At generation time, new rows are sampled token-by-token and parsed back into structured tabular form.

---

### Research Paper
Zhao, Z., Birke, R., & Chen, L. (2023). *TabuLa: Harnessing Language Models for Tabular Data Synthesis*. arXiv:2310.12746.

**Parameters kept the same (per paper, Table 2):**
- Model architecture: distilGPT-2 with randomly initialized weights
- Column order: shuffled randomly per row during both training and generation
- Generation strategy: `do_sample=True`, `temperature=0.8`, `k=16` rows per round
- Epochs: 50 for large datasets, 100 for small datasets
- `max_length` scaled to the width of each dataset's serialized rows

**Parameters changed / inferred:**
- `max_rounds` (not specified in paper): set to 100–150 depending on dataset size to ensure sufficient valid rows are collected
- `n_samples` (not specified): set to 1000 to match the evaluation pipeline standard
- For large datasets (Adult, Bank Marketing, Credit Card, Covertype), a subsampling block of 50–200 rows per class is used for CPU/MPS validation runs — this must be removed for full GPU training to reproduce paper results

**Parts not specified in the paper:**
- Exact tokenizer padding and truncation behaviour — inferred from the GReaT implementation (Borisov et al., 2023), which TabuLa builds upon
- Row validity checking logic (constraint ranges, type coercion) — implemented in the Katabatic evaluation pipeline

---

## Approach
TabuLa's pipeline consists of three stages: serialization, fine-tuning, and generation with filtering.

1. Each training row is converted to a free-text string with column–value pairs in random order.
2. A distilGPT-2 model (randomly initialized) is fine-tuned on these strings using a standard causal language modelling objective.
3. At inference, the model generates text sequences which are parsed back into column–value dictionaries, validated against expected types and constraint ranges, and collected until `n_samples` valid rows are obtained.

### Training Details
The model is fine-tuned using the Hugging Face `Trainer` API with a causal language modelling loss. Each token is predicted conditioned on all preceding tokens in the serialized row. Rows are padded/truncated to `max_length` tokens. The training data is the real tabular dataset (or a subsampled version for CPU/MPS runs).

### Convergence Criteria
Training runs for a fixed number of epochs (`epochs=50` for large datasets, `epochs=100` for small datasets) as specified in the paper. There is no early stopping.

---

## Hyperparameters

| Hyperparameter | Value | Source |
|---|---|---|
| `epochs` | 50 (large datasets), 100 (small) | Zhao et al. (2023), Table 2 |
| `k` | 16 | Zhao et al. (2023), Section 4.2 — generation batch size per round |
| `temperature` | 0.8 | Zhao et al. (2023) |
| `do_sample` | True | Zhao et al. (2023) |
| `max_length` | 256–512 (dataset-dependent) | Scaled to serialized row width |
| `n_samples` | 1000 | Katabatic pipeline standard |
| `max_rounds` | 100–150 | Inferred; ensures sufficient valid rows |

---

## Input
- `X`: Tabular feature matrix
- `y`: Target labels

### Expected Files
- `x_train.csv`
- `y_train.csv`

---

## Preprocessing
No additional preprocessing beyond the standard Katabatic pipeline. Column order is shuffled randomly per row at serialization time (inside the model).

### Numerical Features
Numerical values are serialized as plain decimal strings (e.g., `"age 35"`). No normalization or binning is applied before serialization.

### Categorical Features
Categorical values are serialized as their string labels (e.g., `"education bachelors"`). No encoding is applied; the language model learns the category vocabulary directly from the text.

---

## Label Handling
The target label is included as a column in the serialized row (with random position) during training. At generation time, the model produces rows that include the label column, which is then separated into `y_synth.csv`.

---

## Output
Generated files:

- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

---

## Evaluation
`evaluate()` returns the standard Katabatic 6-dimension evaluation report:
- **Fidelity** — statistical similarity between real and synthetic distributions
- **Utility** — downstream ML performance on synthetic vs. real data
- **Diversity** — coverage of the real data's feature space
- **Privacy** — resistance to membership inference and attribute disclosure
- **Consistency** — internal logical coherence of synthetic rows
- **Stability** — variance of fidelity and diversity across multiple generation seeds

---

## Strengths
- Handles mixed-type data (categorical and numerical) natively through text serialization — no separate encoding pipelines needed.
- Random column shuffling prevents the model from learning spurious positional correlations, improving generalization.
- Lightweight architecture (distilGPT-2) trains faster than full GPT-2 while maintaining competitive quality.
- Works well on datasets with rich categorical structure where statistical models struggle to capture complex interactions.

---

## Limitations
- Valid row rate is typically low on CPU/MPS with subsampled training data; full GPU training is required to reproduce paper-quality results.
- Very wide datasets (e.g., Covertype with 54 columns) produce long serialized rows that require large `max_length` values and significantly increase generation time.
- Generation is slow compared to non-LLM models — each round generates only `k=16` candidate rows, so reaching `n_samples=1000` valid rows may require many rounds.
- Purely numerical datasets with many continuous features may see lower fidelity, as the model must learn numeric value distributions from text tokens.

---

## Installation

```bash
poetry install --extras tabula
```

---

## Usage

Benchmark scripts for each dataset:
- Adult: [benchmarks/examples/tabula/run_tabula_adult.py](benchmarks/examples/tabula/run_tabula_adult.py)
- Bank Marketing: [benchmarks/examples/tabula/run_tabula_bank_marketing.py](benchmarks/examples/tabula/run_tabula_bank_marketing.py)
- Car: [benchmarks/examples/tabula/run_tabula_car.py](benchmarks/examples/tabula/run_tabula_car.py)
- Credit Card: [benchmarks/examples/tabula/run_tabula_creditcard.py](benchmarks/examples/tabula/run_tabula_creditcard.py)
- Covertype: [benchmarks/examples/tabula/run_tabula_covtype.py](benchmarks/examples/tabula/run_tabula_covtype.py)

```python
from katabatic.models.tabula.models import TABULA

model = TABULA(
    categorical_columns=["job", "education", "marital"],
    epochs=50,
)

model.train(
    dataset_dir="path/to/split_dir",
    synthetic_dir="path/to/synthetic_dir",
    device="cuda",   # or "mps" / "cpu"
    n_samples=1000,
    k=16,
    max_length=256,
    max_rounds=100,
)
```

