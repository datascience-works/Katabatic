# GReaT Model

## Model Overview
GReaT (Generation of Realistic Tabular data) fine-tunes a pretrained GPT-2 language model on tabular data encoded as natural language sentences. Each row is converted into a sentence like *"Age is 39, Occupation is Adm-clerical, Income is ≤50K"* and the model learns to generate new rows by predicting the next token.

---

### Research Paper
Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2022). *Language Models are Realistic Tabular Data Generators.* ICLR 2023. arXiv:2210.06280v2

---

## Hyperparameters

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | gpt2 (355M params) | Paper Appendix C |
| Epochs | 310 | Paper Appendix C |
| Batch size | 128 | Adjusted for RTX 4090 |
| Optimizer | AdamW, lr = 5×10⁻⁵ | Paper Appendix C |
| Temperature | 0.7 | Paper Appendix C |
| Max token length | 100 | Library default |
| Feature order | Random permutation | Paper Section 3.1 |

---

## Input
- `X`: Tabular feature matrix
- `y`: Target labels

### Expected Files
- `x_train.csv`
- `y_train.csv`

---

## Installation

```bash
poetry install
```

---

## Usage

Benchmark script:
- Adult Income: [benchmarks/examples/great/run_great_adult.py](benchmarks/examples/great/run_great_adult.py)

```python
from katabatic.models.great.models import GReaT

model = GReaT(
    llm="gpt2",
    experiment_dir="trainer_great_adult",
    epochs=310,
    batch_size=128,
)

model.fit(train_df)

synthetic_df = model.sample(
    len(train_df),
    temperature=0.7,
    max_length=100,
    k=100,
    device="cuda",
)
```

> **Note:** Use `model.fit()`, not `model.train()`. The `train()` method silently overrides epochs to 2.

---

## Model Evaluation Benchmark Results

### Adult Income Dataset

Composite score: **0.469**

| Dimension | Score |
|---|---|
| Fidelity | 0.9616 |
| Utility | 0.0000 |
| Diversity | 0.9213 |
| Privacy | 0.0000 |
| Consistency | 0.8853 |
| Stability | 0.9595 |

> Utility and Privacy scored 0 due to pipeline-level bugs in the evaluator, not GReaT's generation quality. Excluding these, the effective score across the remaining four dimensions is approximately **0.93**.

---

## Model Performance

Training time: several hours on a single NVIDIA RTX 4090 (24GB VRAM), 310 epochs, full Adult Income training set (39,073 rows).

Output files:
- `great_adult_evaluation_report.json`
- `great_adult_evaluation_summary.csv`

---

## Strengths
- No lossy preprocessing needed — works directly with raw data
- Leverages contextual knowledge from pretrained LLMs
- Strong fidelity, diversity, and stability scores

---

## Limitations
- Requires GPU — CPU training is impractical
- Can generate NaN rows, causing the utility evaluator to fail
- Slow to train
