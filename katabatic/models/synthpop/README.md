# SynthPop Model

## Model Overview
SynthPop is a synthetic data generation model written in R. It creates fake (synthetic) versions of real datasets by learning the patterns in the original data and generating new rows that look statistically similar. It is commonly used in research and government statistics to share data without exposing private information.

---

### Key Idea
SynthPop goes through each column in the dataset one at a time. For each column, it builds a decision tree (CART) trained on the real data, then uses that tree to generate synthetic values. Each column is generated based on the columns that came before it, so the relationships between variables are preserved.

---

### Research Paper
**Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke Creation of Synthetic Data in R. *Journal of Statistical Software*, 74(11). https://doi.org/10.18637/jss.v074.i11**

**What we kept the same as the paper:**
- `method = "cart"` — use decision trees for synthesis (Section 3.1)
- `m = 1` — generate one synthetic dataset (Section 2)
- `k = nrow(data)` — synthetic dataset has the same number of rows as the training data (Section 2)
- `cart.minbucket = 5` — each leaf node in the tree must have at least 5 rows, to avoid overfitting (Section 3.1, p. 6)

**What we added or changed:**
- Added `seed = 42` for reproducibility (the paper does not specify a fixed seed).
- Added a column name fix in `utils.py`: R automatically converts hyphens in column names to dots (e.g. `marital-status` → `marital.status`), which breaks the evaluation pipeline. We added `gsub(".", "-", colnames(syn_data$syn), fixed = TRUE)` before writing the CSV to restore the original names.

---

## Approach
1. Load the training data into R.
2. For each column (in order), train a CART decision tree using the real data.
3. Use that tree to generate a synthetic value for each row.
4. Repeat until all columns are synthesised, including the target label.
5. Write the synthetic dataset to a CSV file and read it back into Python.

### Training Details
The Python wrapper writes the training data to a CSV, generates a temporary R script, and runs it as a subprocess. The R script calls the `syn()` function from the `synthpop` package using the paper parameters, then saves the synthetic data to a CSV.

### Convergence Criteria
CART trees do not converge — they stop growing when leaf nodes reach the minimum size (`cart.minbucket = 5`). There is no training loop or epochs.

---

## Hyperparameters

| Hyperparameter | Value | Where it comes from |
|---|---|---|
| `method` | `"cart"` | Paper Section 3.1 |
| `m` | `1` | Paper Section 2 |
| `k` | `nrow(data)` | Paper Section 2 |
| `cart.minbucket` | `5` | Paper Section 3.1, p. 6 |
| `seed` | `42` | Set for reproducibility |

---

## Input
- `X`: Tabular feature matrix
- `y`: Target labels

### Expected Files
- `train_synthpop.csv` — the full training data (features + target) saved as a CSV before being passed to R

---

## Preprocessing
No extra preprocessing beyond the standard Katabatic pipeline. The training DataFrame is saved to CSV before calling the model because R reads data directly from disk, not from Python objects.

For large or imbalanced datasets, per-class subsampling is applied to keep training fast. We use an explicit `for` loop instead of `groupby().apply()` to avoid a pandas 2.x bug where the groupby column gets dropped from the result.

### Numerical Features
Passed directly to R with no normalisation. CART handles raw numerical values fine.

### Categorical Features
R automatically detects string columns and converts them to factors. No manual encoding needed.

---

## Label Handling
The target label is included in the training CSV as a regular column. SynthPop synthesises it as part of the sequential process — it is treated the same as any other column.

---

## Output
Generated files:
- `x_synth.csv`
- `y_synth.csv`
- `metadata.json`

---

## Evaluation
`evaluate()` returns a composite score across six dimensions: fidelity, utility, diversity, privacy, consistency, and stability.

---

## Strengths
- Works well with mixed data (both numerical and categorical columns).
- No assumptions about the shape of the data distribution.
- Captures relationships between columns through the sequential CART approach.
- Produces consistent results with a fixed seed.
- Backed by a well-known, peer-reviewed R package.

---

## Limitations
- Requires R and the `synthpop` package to be installed.
- Can be slow on wide datasets (many columns) because each column needs its own CART model.
- Not practical for very large datasets without subsampling first.
- R changes hyphens in column names to dots — needs a workaround in the Python wrapper.

---

## Installation

```bash
poetry install --extras synthpop
```

Also requires R (≥ 4.0) with the synthpop package:
```r
install.packages("synthpop")
```

---

## Usage

Evaluation pipeline benchmark scripts for each dataset:
- Adult — [benchmarks/examples/synthpop/run_synthpop_adult.py](benchmarks/examples/synthpop/run_synthpop_adult.py)
- Bank Marketing — [benchmarks/examples/synthpop/run_synthpop_bank_marketing.py](benchmarks/examples/synthpop/run_synthpop_bank_marketing.py)
- Car — [benchmarks/examples/synthpop/run_synthpop_car.py](benchmarks/examples/synthpop/run_synthpop_car.py)
- Credit Card — [benchmarks/examples/synthpop/run_synthpop_creditcard.py](benchmarks/examples/synthpop/run_synthpop_creditcard.py)
- Covertype — [benchmarks/examples/synthpop/run_synthpop_covertype.py](benchmarks/examples/synthpop/run_synthpop_covertype.py)

```python
from katabatic.models.synthpop import SynthPop

model = SynthPop(seed=42)

model.train(
    dataset_path="path/to/train_synthpop.csv",
    synthetic_path="path/to/synthetic.csv"
)

synthetic_df = model.sample()
```

---

## Model Evaluation Benchmarks Results

#### Adult Dataset

Composite score: 0.9456

Dimension scores:
- fidelity       0.9959
- utility        0.9950
- diversity      0.9616
- privacy        0.6996
- consistency    0.9729
- stability      1.0000

#### Bank Marketing Dataset

Composite score: 0.9674

Dimension scores:
- fidelity       0.9879
- utility        1.0000
- diversity      0.9173
- privacy        0.8933
- consistency    0.9474
- stability      1.0000

#### Car Dataset

Composite score: 0.8860

Dimension scores:
- fidelity       0.9844
- utility        0.9769
- diversity      0.9990
- privacy        0.4158
- consistency    0.8571
- stability      1.0000

#### Credit Card Dataset

Composite score: 0.9810

Dimension scores:
- fidelity       0.9833
- utility        0.9955
- diversity      0.9740
- privacy        0.9766
- consistency    0.9287
- stability      1.0000

#### Covertype Dataset

Composite score: 0.9205

Dimension scores:
- fidelity       0.9931
- utility        0.9403
- diversity      0.7128
- privacy        0.8303
- consistency    0.9734
- stability      1.0000

---

## Model Performance Benchmarks Results

Runtime depends on the number of columns and training rows. All runs tested on Apple M-series CPU with 16 GB RAM.

- **Adult** (subsampled to 1,000 rows, 15 columns): ~2–3 minutes
- **Bank Marketing** (subsampled to 1,000 rows, 16 columns): ~2–3 minutes
- **Car** (~1,700 rows, 6 columns): < 1 minute
- **Credit Card** (subsampled to 1,000 rows, 31 columns): ~3–4 minutes
- **Covertype** (subsampled to 1,400 rows, 54 columns): ~5–8 minutes
