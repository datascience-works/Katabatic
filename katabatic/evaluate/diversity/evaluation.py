import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from katabatic.evaluate.base_evaluation import Evaluation
from katabatic.utils.column_types import get_column_types


class DiversityEvaluation(Evaluation):
    """
    Diversity dimension: measures whether the synthetic data covers the full
    population of the real data, not just a subset of it.

    Metrics
    -------
    - Category Coverage: for each categorical column, the % of real categories
      that also appear in the synthetic data. A model that only generates the
      most common categories will score poorly here.

    - Bin Coverage: each continuous column is divided into 10 equal-width bins
      based on the real data range. Reports the % of bins that contain at least
      one synthetic data point.

    - Gower Distance Distribution: pairwise Gower distances are computed within
      a sample of real rows and within a sample of synthetic rows. The two
      distributions are compared using Wasserstein Distance. A low WD means
      the synthetic data is spread across the data space in the same way as
      the real data.

    Gower distance handles mixed-type data correctly:
      - Continuous columns: normalised absolute difference |x-y| / range
      - Categorical columns: 0 if equal, 1 if different

    diversity_score is the mean of the three component scores.

    Parameters
    ----------
    real_data : pd.DataFrame
    synthetic_data : pd.DataFrame
    categorical_cols : list[str], optional
    continuous_cols : list[str], optional
    n_bins : int
        Number of equal-width bins for continuous coverage (default: 10).
    gower_sample_size : int
        Number of rows sampled from each dataset for Gower computation (default: 200).
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_cols: list = None,
        continuous_cols: list = None,
        n_bins: int = 10,
        gower_sample_size: int = 200,
    ):
        super().__init__(real_data, synthetic_data)
        self.n_bins = n_bins
        self.gower_sample_size = gower_sample_size

        if categorical_cols is None and continuous_cols is None:
            self.categorical_cols, self.continuous_cols = get_column_types(self.real_data, exclude_last=False)
            print(
                "[WARNING] categorical_cols and continuous_cols were not provided. "
                "Auto-detecting from dtypes — integer-encoded categorical columns will be "
                "misclassified as continuous. Pass column types explicitly for accurate results."
            )
        else:
            self.categorical_cols = categorical_cols or []
            self.continuous_cols = continuous_cols or []

    def evaluate(self) -> dict:
        cat_coverage = self._compute_category_coverage()
        bin_coverage = self._compute_bin_coverage()
        gower_similarity = self._compute_gower_similarity()

        # All three component scores are already in [0, 1].
        # Use len > 1 (not truthiness) to guard against dicts that only contain
        # the 'avg' sentinel when all columns were skipped — that would be truthy
        # but would incorrectly contribute a 1.0 score.
        active = [s for s in [
            cat_coverage.get('avg') if len(cat_coverage) > 1 else None,
            bin_coverage.get('avg') if len(bin_coverage) > 1 else None,
            gower_similarity,
        ] if s is not None]

        diversity_score = round(float(np.mean(active)), 4) if active else 0.0

        results = {
            'category_coverage': cat_coverage,
            'bin_coverage': bin_coverage,
            'gower_similarity': round(gower_similarity, 4) if gower_similarity is not None else None,
            'diversity_score': diversity_score,
        }

        self._print_summary(results)
        return results

    def _compute_category_coverage(self) -> dict:
        if not self.categorical_cols:
            return {}

        scores = {}
        for col in self.categorical_cols:
            if col not in self.synthetic_data.columns:
                continue
            real_cats = set(self.real_data[col].dropna().unique())
            synth_cats = set(self.synthetic_data[col].dropna().unique())
            if not real_cats:
                continue
            coverage = len(real_cats & synth_cats) / len(real_cats)
            scores[col] = round(coverage, 4)

        scores['avg'] = round(float(np.mean(list(scores.values()))), 4) if scores else 0.0
        return scores


    def _compute_bin_coverage(self) -> dict:
        if not self.continuous_cols:
            return {}

        scores = {}
        for col in self.continuous_cols:
            if col not in self.synthetic_data.columns:
                continue

            real_vals = self.real_data[col].dropna().values.astype(float)
            synth_vals = self.synthetic_data[col].dropna().values.astype(float)

            if len(real_vals) == 0:
                continue

            col_min, col_max = real_vals.min(), real_vals.max()
            if col_min == col_max:
                # Constant column covered if synthetic also produces that value
                coverage = 1.0 if np.any(synth_vals == col_min) else 0.0
                scores[col] = round(coverage, 4)
                continue

            # Build bins from real data range, check which bins synth hits
            bins = np.linspace(col_min, col_max, self.n_bins + 1)
            # np.digitize returns bin indices 1..n_bins; 0 and n_bins+1 are out-of-range
            synth_bin_ids = np.digitize(synth_vals, bins, right=False)
            # Clip out-of-range values into the edge bins
            synth_bin_ids = np.clip(synth_bin_ids, 1, self.n_bins)
            bins_hit = len(set(synth_bin_ids))
            scores[col] = round(bins_hit / self.n_bins, 4)

        scores['avg'] = round(float(np.mean(list(scores.values()))), 4) if scores else 0.0
        return scores


    def _compute_gower_similarity(self):
        all_cols = self.categorical_cols + self.continuous_cols
        shared = [c for c in all_cols if c in self.synthetic_data.columns]
        if len(shared) == 0:
            return None

        real_clean  = self.real_data[shared].dropna()
        synth_clean = self.synthetic_data[shared].dropna()
        n = min(self.gower_sample_size, len(real_clean), len(synth_clean))
        if n == 0:
            return None
        real_sample  = real_clean.sample(n=n, random_state=42)
        synth_sample = synth_clean.sample(n=n, random_state=42)

        # Pre-compute ranges for continuous columns (from full real data)
        cont_shared = [c for c in self.continuous_cols if c in shared]
        cat_shared = [c for c in self.categorical_cols if c in shared]
        ranges = {
            col: float(self.real_data[col].max() - self.real_data[col].min())
            for col in cont_shared
        }

        real_dists = self._pairwise_gower(real_sample, cat_shared, cont_shared, ranges)
        synth_dists = self._pairwise_gower(synth_sample, cat_shared, cont_shared, ranges)

        if len(real_dists) == 0 or len(synth_dists) == 0:
            return None

        # Wasserstein distance between the two Gower distance distributions
        # Since Gower distances are in [0,1], WD is also in [0,1]
        wd = float(wasserstein_distance(real_dists, synth_dists))
        similarity = max(0.0, 1.0 - wd)
        return round(similarity, 4)

    def _pairwise_gower(self, df, cat_cols, cont_cols, ranges):
        """
        Compute upper-triangle pairwise Gower distances for all row pairs in df.
        Returns a 1-D array of distance values.
        """
        n = len(df)
        n_cols = len(cat_cols) + len(cont_cols)
        if n_cols == 0:
            return np.array([])

        # Pre-extract numpy arrays for speed
        cont_data = df[cont_cols].values.astype(float) if cont_cols else np.empty((n, 0))
        cat_data = df[cat_cols].values if cat_cols else np.empty((n, 0), dtype=object)
        cont_ranges = np.array([ranges.get(c, 1.0) or 1.0 for c in cont_cols], dtype=float)

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = 0.0

                # Continuous: normalised absolute difference
                if cont_cols:
                    d += float(np.sum(np.abs(cont_data[i] - cont_data[j]) / cont_ranges))

                # Categorical: indicator (0 = same, 1 = different)
                if cat_cols:
                    d += float(np.sum(cat_data[i] != cat_data[j]))

                distances.append(d / n_cols)

        return np.array(distances)


    def _print_summary(self, results):
        print("\n=== Diversity Evaluation ===")
        print(f"Overall diversity score: {results['diversity_score']:.4f}")

        if results['category_coverage']:
            print(f"\nCategory Coverage (% of real categories in synth)")
            for col, val in results['category_coverage'].items():
                tag = '  avg' if col == 'avg' else f'  {col}'
                print(f"{tag:<32} {val * 100:.1f}%")

        if results['bin_coverage']:
            print(f"\nBin Coverage ({self.n_bins} bins, % bins hit by synth)")
            for col, val in results['bin_coverage'].items():
                tag = '  avg' if col == 'avg' else f'  {col}'
                print(f"{tag:<32} {val * 100:.1f}%")

        if results['gower_similarity'] is not None:
            print(f"\nGower distance distribution similarity: {results['gower_similarity']:.4f}")
