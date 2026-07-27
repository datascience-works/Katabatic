# Katabatic Evaluation Pipeline — Team Guide

This document explains how the Katabatic evaluation pipeline scores synthetic tabular data.
Every model in the library is evaluated across 6 dimensions, each targeting a different aspect
of synthetic data quality. The results are combined into a single composite score between 0 and 1.

---

## How to Read a Score

Every dimension and every sub-metric produces a score in the range **[0, 1]**.
- **1.0** = perfect on that metric
- **0.0** = worst possible on that metric

The composite score is a weighted average of all active dimensions. Weights are re-normalised
against whichever dimensions were actually run, so the composite is always in [0, 1] regardless
of which dimensions are skipped.

**Reading order for a report:**
1. Start with `composite_score` — use this to compare models against each other.
2. Check `dimension_scores` — identify which dimensions are strong and which are weak.
3. Drill into `full_results` only for dimensions that surprise you.

---

## Composite Score Weights

| Dimension   | Default Weight | Why this weight |
|-------------|---------------|-----------------|
| Utility     | 35%           | Most practically important — measures real-world usability |
| Fidelity    | 25%           | Baseline statistical quality |
| Privacy     | 15%           | Critical for sensitive domains (healthcare, finance) |
| Diversity   | 10%           | Ensures full population coverage |
| Consistency | 10%            | Internal coherence check |
| Stability   | 5%            | Reproducibility across runs |

Weights are re-normalised when dimensions are skipped. For example if Stability is not run,
its 5% is redistributed proportionally among the other five dimensions.

---

## Dimension 1 — Fidelity
**Question: Does the synthetic data look like the real data statistically?**

This is the baseline dimension. If a model fails here, it fails everywhere. Fidelity measures
how closely the synthetic data mirrors the statistical properties of the real data, column by
column and relationship by relationship.

### Sub-metrics

#### Jensen-Shannon Divergence (JSD) — categorical columns
Compares the probability distribution of each categorical column between real and synthetic data.

- **Range:** 0 to 1 (lower is better — 0 means identical distributions)
- **How to read:** Each categorical column gets its own JSD value. The `avg` key is the average
  across all categorical columns. The `categorical_score` = 1 - avg_JSD.
- **Example:** `workclass: 0.2416` means the distribution of workclass categories in synthetic
  data differs moderately from the real data distribution.

#### Wasserstein Distance — continuous columns
Measures how much the distribution of each continuous column needs to "shift" to match the real
data. Normalised by the real column's range so all columns are on the same [0, 1] scale.

- **Range:** 0 to 1 (lower is better — 0 means identical distributions)
- **How to read:** Each continuous column gets its own value. The `avg` is the mean across all
  continuous columns. The `continuous_score` = 1 - avg_WD.
- **Example:** `age: 0.0473` means the age distribution in synthetic data is very close to real.

#### Pearson Correlation Matrix Difference — inter-column relationships
Computes the correlation matrix (how columns relate to each other) for both real and synthetic
data, then measures the average absolute difference between them element by element.

- **Range:** 0 to 1 (lower is better — 0 means identical inter-column relationships)
- **How to read:** A single value. `correlation_score` = 1 - correlation_diff.
- **Example:** `correlation_diff: 0.0143` means the relationships between columns are very well
  preserved in the synthetic data.

### How the fidelity_score is computed
```
fidelity_score = mean(categorical_score, continuous_score, correlation_score)
```

### What to look for
- High JSD on specific columns → the model struggles to capture that column's distribution
- High Wasserstein on continuous columns → the model is generating values in the wrong range
- High correlation_diff → the model captures individual columns but not their relationships

---

## Dimension 2 — Utility
**Question: Can you actually use the synthetic data to train machine learning models?**

This is the most practically important dimension. It measures how well synthetic data substitutes
for real data in downstream ML tasks using the Train on Synthetic Test on Real (TSTR) protocol.

### Protocol

**TSTR (Train on Synthetic, Test on Real)**
Five classifiers are trained on 5 cross-validation folds of the synthetic data. Each trained
classifier is then evaluated on the held-out real test set. This measures how useful the
synthetic data is as a training source.

**TRTR (Train on Real, Test on Real)**
The same five classifiers are trained on 5 folds of the real data and evaluated on the same
held-out real test set. This is the performance ceiling — the best a classifier can do with
real data.

**Delta = TRTR - TSTR**
The gap between the two protocols. A delta of 0 means synthetic data trains classifiers just
as well as real data. A large delta means the synthetic data is a poor substitute.

### Five Classifiers
1. Logistic Regression (LR)
2. Decision Tree (DT)
3. Random Forest (RF)
4. Linear SVM (LinearSVM)
5. Multi-Layer Perceptron (MLP)

### Three Metrics per Classifier
- **Accuracy** — overall correct predictions
- **F1** — weighted F1 score, balances precision and recall
- **AUC** — area under the ROC curve (binary classification only)

### How the utility_score is computed
```
utility_score = 1 - mean(all deltas across all classifiers and all metrics)
```
Clipped to [0, 1]. A score of 1.0 means synthetic data performs identically to real data.

### What to look for
- Decision Tree typically shows the largest delta — it is sensitive to distribution differences
  and tends to overfit synthetic data. This is expected behaviour.
- Linear models (LR, SVM) with large deltas indicate the synthetic data has wrong linear
  decision boundaries — a fundamental fidelity problem.
- Low `std` values across folds mean the evaluation is stable and trustworthy.

---

## Dimension 3 — Diversity
**Question: Does the synthetic data cover the full population, or just a subset of it?**

A model could generate very realistic-looking rows but always generate the same ones. That would
be useless for generalisation. Diversity checks that the synthetic data covers the full range
of variation present in the real data.

### Sub-metrics

#### Category Coverage — categorical columns
For each categorical column, what percentage of the real data's categories appear at least once
in the synthetic data.

- **Range:** 0 to 1 (higher is better — 1.0 means all real categories are represented)
- **How to read:** Each categorical column gets a coverage score. `avg` is the mean across all
  categorical columns.
- **Example:** `native-country: 0.1667` means only 7 out of 41 countries appear in the synthetic
  data — the model ignores most rare nationalities.

#### Bin Coverage — continuous columns
Each continuous column is divided into 10 equal-width bins based on the real data's range.
The score is the fraction of bins that contain at least one synthetic value.

- **Range:** 0 to 1 (higher is better — 1.0 means every part of the real range is covered)
- **How to read:** Each continuous column gets a coverage score. `avg` is the mean.
- **Example:** `capital-loss: 0.8` means 8 out of 10 bins are covered — the model misses some
  extreme capital-loss values, which are rare in the real data.

#### Gower Distance Distribution Similarity
Pairwise Gower distances are computed within a sample of real rows and within a sample of
synthetic rows. The two distributions are then compared using Wasserstein Distance. This
measures whether synthetic rows are spread across the data space in the same way as real rows.

Gower distance handles mixed-type data correctly — continuous columns use normalised absolute
difference, categorical columns use equality (0 = same, 1 = different).

- **Range:** 0 to 1 (higher is better — 1.0 means identical spread across the data space)
- **How to read:** A single value. Low values mean synthetic rows are clustered together while
  real rows are spread out (or vice versa).

### How the diversity_score is computed
```
diversity_score = mean(category_coverage_avg, bin_coverage_avg, gower_similarity)
```

### What to look for
- Low category coverage on specific columns → the model ignores rare categories in that column
- Low bin coverage → the model does not generate extreme values in that column
- Low Gower similarity → the model collapses to a narrow region of the data space

---

## Dimension 4 — Privacy
**Question: Is the synthetic data leaking information about real individuals?**

Critical for sensitive domains like healthcare and finance. Privacy measures whether the
synthetic data is too close to real records — either memorising them directly or generating
near-copies of them.

### Sub-metrics

#### Nearest Neighbour Distance Ratio (NNDR)
For each synthetic row, find the two closest real rows:
- d1 = distance to the nearest real row
- d2 = distance to the second nearest real row
- NNDR = d1 / d2

**Interpretation:**
- Ratio close to **1.0** → the synthetic row sits in a naturally dense region (private) — both
  neighbours are roughly equally close, so the row is not targeting any specific individual.
- Ratio close to **0.0** → one real row is dramatically closer than all others — strong signal
  of memorisation.

The score is the mean NNDR across all synthetic rows (higher = more private).

#### Exact Duplicate Rate
Percentage of synthetic rows that are identical to at least one real row (Gower distance = 0).
Score = 1 - rate. A score of 1.0 means zero exact copies.

#### Near-Duplicate Rate
Percentage of synthetic rows whose nearest real neighbour has a Gower distance below the
threshold (default: 0.01). Score = 1 - rate.

The threshold of 0.01 is strict — it means the synthetic row differs by less than 1% of the
possible range across all columns combined.

### How the privacy_score is computed
```
privacy_score = mean(nndr_score, exact_dup_score, near_dup_score)
```

### What to look for
- Low NNDR score → the model is memorising specific real records
- Non-zero exact duplicate rate → the model is directly copying real rows
- High near-duplicate rate → the model generates data that is very close to real records.
  Can be caused by model memorisation OR by the dataset being naturally dense (many real
  people share very similar feature combinations). Context matters here.
- A high NNDR score alongside a high near-duplicate rate suggests the second case — dense
  data space, not memorisation.

---

## Dimension 5 — Consistency
**Question: Is the synthetic data internally coherent and does it preserve the real data's structure?**

Statistical plausibility does not guarantee logical coherence. A row with age = -5 is
statistically possible but meaningless. Consistency checks whether the synthetic data is
logically valid and whether it preserves the predictive structure of the real data.

### Sub-metrics

#### Discriminator Score
Real rows (label = 1) and synthetic rows (label = 0) are combined. A Random Forest classifier
is trained via 5-fold cross-validation to distinguish them.

- **Target accuracy: ~0.5** — the classifier cannot tell real from synthetic.
- **Above 0.7:** red flag — the synthetic data is easily detectable as fake.
- **1.0:** worst possible — the classifier is perfect at detecting synthetic rows.

Score = max(0, 1 - (accuracy - 0.5) × 2). So:
- 0.5 accuracy → score 1.0 (ideal)
- 0.7 accuracy → score 0.6
- 1.0 accuracy → score 0.0 (worst)

#### Constraint Violation Rate
User-defined logical bounds per column (e.g. age must be between 17 and 90) are checked
against the synthetic data. Reports the percentage of rows violating each constraint and an
overall violation rate. Score = 1 - overall_violation_rate.

If no constraints are defined this sub-metric is skipped and does not affect the score.

#### Feature Importance Spearman Correlation
A Random Forest is trained on real data and on synthetic data separately, using the target
column. The feature importance rankings from each model are compared using Spearman rank
correlation.

- **Range:** 0 to 1 after clipping (higher is better)
- **1.0** = both models agree completely on which features are most predictive
- **0.0** = the rankings are completely different — synthetic data has distorted which features matter

### How the consistency_score is computed
```
consistency_score = mean(discriminator_score, constraint_score, spearman_score)
```
Only active sub-metrics are included in the mean. If no constraints are defined, only
discriminator and Spearman are averaged.

### What to look for
- Discriminator at 1.0 → synthetic data has a systematic signature easily detectable by a
  classifier. Often caused by model artefacts or post-processing steps.
- Zero constraint violations → the model respects logical bounds (good).
- Low Spearman → the synthetic data has distorted feature importance — models trained on it
  will rely on different features than models trained on real data.

---

## Dimension 6 — Stability
**Question: Does the model produce consistent results across multiple runs?**

A model that generates great data once and terrible data the next time is not trustworthy.
Stability measures the variance of Fidelity and Diversity scores across multiple independent
sampling runs with different random seeds.

### Protocol
The model is sampled N times (default: 5) using seeds 0, 1, 2, 3, 4. For each run:
- Fidelity is computed between the real data and that run's synthetic data
- Diversity is computed between the real data and that run's synthetic data

The standard deviation of each metric across runs is then used to compute the stability score.

### How the stability_score is computed
```
avg_std = mean(std_fidelity, std_diversity)
stability_score = max(0, 1 - avg_std / 0.10)
```
- avg_std = 0.0 → score 1.0 (perfectly stable)
- avg_std = 0.05 → score 0.5 (borderline — warning is printed)
- avg_std ≥ 0.10 → score 0.0 (unstable)

### What to look for
- High std on fidelity → the model generates data with very different statistical properties
  depending on the random seed
- High std on diversity → the model sometimes covers the full data space, sometimes collapses
  to a narrow region
- Low avg_std (< 0.05) → the model is consistently reliable across runs

---

## Interpreting Results Together

Some dimensions are in natural tension with each other. Understanding these tensions is
important for interpreting results correctly.

**Fidelity vs Privacy**
A model that perfectly reproduces real distributions is likely generating data that is close
to real records. Very high fidelity combined with very low privacy suggests memorisation.
Some tension between these two is expected and healthy.

**Utility vs Diversity**
A model can achieve high utility by generating data that trains good classifiers for the
majority class, while ignoring rare subgroups. High utility with low category coverage on
rare categories is a signal that the model is not capturing the full population.

**Discriminator Score**
A discriminator accuracy of 1.0 is a strong signal that something systematic is wrong with
the synthetic data — a pattern that a classifier can exploit. This is often caused by:
- Model artefacts (e.g. clipped values, rounding patterns)
- Post-processing steps that modify synthetic data after generation
- Fundamental distribution mismatch in a specific column

When the discriminator score is very low but other dimensions are strong, investigate the
`full_results` of fidelity and diversity for specific columns that might be causing the issue.

---

## Quick Reference — Score Interpretation

| Score Range | Interpretation |
|-------------|---------------|
| 0.90 – 1.00 | Excellent |
| 0.75 – 0.90 | Good |
| 0.60 – 0.75 | Acceptable |
| 0.40 – 0.60 | Weak — investigate full_results |
| 0.00 – 0.40 | Poor — model has a significant problem in this dimension |
