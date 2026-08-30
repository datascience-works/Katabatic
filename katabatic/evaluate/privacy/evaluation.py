import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from katabatic.evaluate.base_evaluation import Evaluation


class PrivacyEvaluation(Evaluation):
    """
    Privacy dimension: measures whether the synthetic data leaks information
    about real individuals through memorisation or near-copying.

    Metrics
    -------
    - Nearest Neighbour Distance Ratio (NNDR): for each synthetic row, finds
      the two closest real rows (d1 = nearest, d2 = second nearest) and
      computes ratio = d1 / d2. A ratio close to 1 means the synthetic row
      is not targeting any specific real individual, it simply exists in a
      naturally dense region of the data space. A ratio close to 0 means one
      real row is dramatically closer than all others, a strong signal of
      memorisation.
      Score = mean NNDR across all synthetic rows (higher = more private).

    - Exact Duplicate Rate: % of synthetic rows that are identical to at
      least one real row (after normalisation). Score = 1 - rate.

    - Near-Duplicate Rate: % of synthetic rows whose Gower distance to the
      nearest real row is below `near_dup_threshold`. Score = 1 - rate.

    privacy_score is the mean of the three component scores.

    Distance computations use proper Gower distance:
      - Continuous columns: normalised absolute difference |xi - xj| / range
      - Categorical columns: 0 if same category, 1 if different (true Gower)

    Parameters
    ----------
    real_data : pd.DataFrame
    synthetic_data : pd.DataFrame
    near_dup_threshold : float
        Gower distance threshold for near-duplicate detection (default: 0.01).
    sample_size : int or None
        If set, limits NNDR computation to a random sample of synthetic rows
        to keep runtime manageable for large datasets (default: 2000).
    categorical_cols : list[str], optional
        Columns to treat as categorical for Gower distance.
    continuous_cols : list[str], optional
        Columns to treat as continuous for Gower distance.
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        near_dup_threshold: float = 0.01,
        sample_size: int = 2000,
        categorical_cols: list = None,
        continuous_cols: list = None,
    ):
        super().__init__(real_data, synthetic_data)
        self.near_dup_threshold = near_dup_threshold
        self.sample_size = sample_size
        self.categorical_cols = categorical_cols or []
        self.continuous_cols = continuous_cols or []

    def evaluate(self) -> dict:
        real_norm, synth_norm, cat_mask = self._normalise()

        # Sample once — all three metrics operate on the same subset so their
        # scores are comparable when averaged into privacy_score.
        synth_eval = synth_norm
        if self.sample_size and len(synth_norm) > self.sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(synth_norm), self.sample_size, replace=False)
            synth_eval = synth_norm[idx]

        dist = self._gower_matrix(synth_eval, real_norm, cat_mask)

        mean_nndr = self._compute_nndr(dist)
        exact_dup_rate = self._compute_exact_duplicates(dist, synth_eval)
        near_dup_rate = self._compute_near_duplicates(dist, synth_eval)

        nndr_score = round(mean_nndr, 4)
        exact_dup_score = round(1.0 - exact_dup_rate, 4)
        near_dup_score = round(1.0 - near_dup_rate, 4)
        privacy_score = round(
            float(np.mean([nndr_score, exact_dup_score, near_dup_score])), 4
        )

        results = {
            "mean_nndr": round(mean_nndr, 4),
            "exact_duplicate_rate": round(exact_dup_rate, 4),
            "near_duplicate_rate": round(near_dup_rate, 4),
            "nndr_score": nndr_score,
            "exact_dup_score": exact_dup_score,
            "near_dup_score": near_dup_score,
            "privacy_score": privacy_score,
        }

        self._print_summary(results)
        return results

    def _normalise(self):
        """
        Return normalised arrays and a categorical mask for Gower distance.

        Continuous columns: min-max scaled to [0, 1] using real data range.
        Categorical columns: label-encoded to integer codes (used for == comparison).

        Returns
        -------
        real_out : np.ndarray
        synth_out : np.ndarray
        cat_mask : np.ndarray of bool
            True for categorical columns, False for continuous.
        """
        real = self.real_data.copy()
        synth = self.synthetic_data.copy()

        shared_cols = [c for c in real.columns if c in synth.columns]
        real = real[shared_cols]
        synth = synth[shared_cols]

        real_out = np.zeros((len(real), len(shared_cols)), dtype=float)
        synth_out = np.zeros((len(synth), len(shared_cols)), dtype=float)
        cat_mask = np.zeros(len(shared_cols), dtype=bool)

        for i, col in enumerate(shared_cols):
            if col in self.categorical_cols:
                # Encode to integer codes for equality comparison in Gower
                cat_mask[i] = True
                le = LabelEncoder()
                combined = pd.concat([real[col], synth[col]]).astype(str)
                le.fit(combined)
                real_out[:, i] = le.transform(real[col].astype(str)).astype(float)
                synth_out[:, i] = le.transform(synth[col].astype(str)).astype(float)
            else:
                # Continuous: min-max scale to [0, 1] using real data range
                real_num = pd.to_numeric(real[col], errors="coerce")
                synth_num = pd.to_numeric(synth[col], errors="coerce")
                col_min = float(real_num.min())
                col_max = float(real_num.max())
                col_range = col_max - col_min
                if col_range > 0:
                    real_out[:, i] = (real_num.values - col_min) / col_range
                    synth_out[:, i] = (
                        synth_num.fillna(col_min).values - col_min
                    ) / col_range
                # else constant column — stays 0

        return real_out, synth_out, cat_mask

    @staticmethod
    def _gower_matrix(A, B, cat_mask, batch_size=500):
        """
        Compute pairwise Gower distance matrix between rows of A (n x d)
        and rows of B (m x d).

        Gower distance per column:
          - Categorical: 0 if A[i,j] == B[k,j], else 1
          - Continuous:  |A[i,j] - B[k,j]|  (already normalised to [0,1])

        Final distance = mean over all columns → value in [0, 1].

        Computed in batches to avoid memory issues on large datasets.
        """
        n, d = A.shape
        m = B.shape[0]
        dist = np.zeros((n, m), dtype=np.float32)
        cont_mask = ~cat_mask

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            A_batch = A[start:end]  # (batch, d)

            batch_dist = np.zeros((end - start, m), dtype=np.float32)

            if np.any(cont_mask):
                # Continuous: absolute difference, shape (batch, m, n_cont)
                cont_diff = np.abs(
                    A_batch[:, np.newaxis, :][:, :, cont_mask]
                    - B[np.newaxis, :, :][:, :, cont_mask]
                )
                batch_dist += cont_diff.sum(axis=2)

            if np.any(cat_mask):
                # Categorical: 0 same, 1 different, shape (batch, m, n_cat)
                cat_diff = (
                    A_batch[:, np.newaxis, :][:, :, cat_mask]
                    != B[np.newaxis, :, :][:, :, cat_mask]
                ).astype(np.float32)
                batch_dist += cat_diff.sum(axis=2)

            dist[start:end] = batch_dist / d

        return dist

    def _compute_nndr(self, dist: np.ndarray) -> float:
        """
        For each synthetic row find the 2 nearest real rows using Gower distance.
        NNDR = d1 / d2. Mean across all synthetic rows.
        Sampling and matrix computation are handled by the caller.
        """
        if dist.shape[1] < 2:
            raise ValueError(
                f"NNDR requires at least 2 real rows to compute a nearest-neighbour ratio, "
                f"but real_data only has {dist.shape[1]} row(s). "
                "Ensure the training split contains sufficient data."
            )
        partitioned = np.partition(dist, kth=1, axis=1)
        d1 = partitioned[:, 0]
        d2 = partitioned[:, 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(d2 > 0, d1 / d2, 0.0)

        return float(np.mean(ratios))

    def _compute_exact_duplicates(
        self, dist: np.ndarray, synth_norm: np.ndarray
    ) -> float:
        """
        % of synthetic rows with Gower distance == 0 to any real row.
        """
        exact = np.sum(dist.min(axis=1) == 0.0)
        return float(exact) / len(synth_norm)

    def _compute_near_duplicates(
        self, dist: np.ndarray, synth_norm: np.ndarray
    ) -> float:
        """
        % of synthetic rows whose nearest real neighbour Gower distance is in
        (0, near_dup_threshold). Exact duplicates (distance == 0) are excluded
        so they are not double-counted with the exact duplicate metric.
        """
        min_dists = dist.min(axis=1)
        near = np.sum((min_dists > 0) & (min_dists < self.near_dup_threshold))
        return float(near) / len(synth_norm)

    def _print_summary(self, results):
        print("\n=== Privacy Evaluation ===")
        print(f"Overall privacy score: {results['privacy_score']:.4f}")

        print(
            f"\nNNDR (mean ratio d1/d2):      {results['mean_nndr']:.4f}  (1.0 = fully private, 0.0 = memorised)"
        )
        print(f"NNDR score:                   {results['nndr_score']:.4f}")

        print(
            f"\nExact duplicate rate:         {results['exact_duplicate_rate'] * 100:.2f}%"
        )
        print(
            f"Near-duplicate rate (<{self.near_dup_threshold}):  {results['near_duplicate_rate'] * 100:.2f}%"
        )

        if results["exact_duplicate_rate"] > 0.01:
            print("  [WARNING] Exact duplicates detected — possible memorisation.")
        if results["near_duplicate_rate"] > 0.05:
            print("  [WARNING] High near-duplicate rate — review model privacy.")
