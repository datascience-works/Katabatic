# NaiveBayesSynth

A lightweight, dependency-free baseline synthetic data generator for
discrete/categorical tabular data.

## How it works

1. **Fit**: learns `P(class)` and, for every feature independently,
   `P(feature = category | class)`, with Laplace smoothing so unseen
   combinations don't get zero probability.
2. **Sample**: draws a class label from the class prior, then samples each
   feature independently from its class-conditional distribution (the
   "naive" independence assumption — features are *not* modeled jointly,
   unlike GANBLR's kDB-based dependency graph).
3. **Evaluate**: standard TSTR (Train on Synthetic, Test on Real) — trains a
   classifier (`lr` / `rf` / `mlp` / custom sklearn-style estimator) on
   sampled synthetic data, tests it on real held-out data, returns accuracy.

## Status

Experimental (`supported: False` in `katabatic/models/registry.py`) — not
yet validated across the full benchmark suite.

## Dependencies

- numpy
- pandas
- scikit-learn

No extra dependencies beyond the project's base requirements — no
`pyproject.toml`/`poetry.lock` needed for this model.

## Example

```python
from katabatic.models.naive_bayes_synth import NaiveBayesSynth

model = NaiveBayesSynth()
model.fit(X_train, y_train)

synthetic_df = model.sample(size=1000)

accuracy = model.evaluate(X_test, y_test, model="lr")
```

## Known limitations

- Assumes features are conditionally independent given the class. Real
  tabular data (e.g. the UCI Car Evaluation dataset) has feature
  interactions this model can't capture, so TSTR accuracy is expected to
  trail dependency-aware models like GANBLR.
- Categorical/discrete data only — no continuous-feature handling.
- `evaluate()` samples `max(len(x_test), 500)` synthetic rows for the TSTR
  training set; tune this in your own evaluation script if you need a
  different synthetic training-set size.

## Results (car evaluation dataset)

| Eval model | TSTR accuracy |
|---|---|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.76 |

(25% held-out real test split, `random_state=42`.)