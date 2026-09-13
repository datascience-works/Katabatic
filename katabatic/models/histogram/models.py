import numpy as np
import pandas as pd

from .utils import get_column_distribution, sample_column


class HistogramModel:
    """
    Simple univariate histogram-based tabular data generator.

    For each column, we estimate the empirical distribution (value_counts)
    and then sample from it independently for each synthetic row.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.column_distributions = {}
        self.columns_ = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, df: pd.DataFrame):
        """
        Learn per-column empirical distributions from the real data.

        Parameters
        ----------
        df : pd.DataFrame
            Input real dataset.
        """
        self.columns_ = list(df.columns)
        self.column_distributions = {}

        for col in self.columns_:
            values, probs = get_column_distribution(df[col])

            self.column_distributions[col] = (values, probs)

    def generate(self, n_rows: int) -> pd.DataFrame:
        """
        Generate synthetic data by sampling each column independently.

        Parameters
        ----------
        n_rows : int
            Number of rows to generate.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset with the same columns as the training data.
        """
        if self.columns_ is None or not self.column_distributions:
            raise RuntimeError(
                "HistogramModel must be fitted before calling generate()."
            )

        data = {}

        for col in self.columns_:
            values, probs = self.column_distributions[col]

            sampled = sample_column(
                self._rng,
                values,
                probs,
                n_rows,
            )

            data[col] = sampled

        synthetic_df = pd.DataFrame(
            data,
            columns=self.columns_,
        )

        return synthetic_df

    def sample(self, n_rows: int, seed=None) -> pd.DataFrame:
        """
        Generate synthetic data using the Histogram model.

        Parameters
        ----------
        n_rows : int
            Number of synthetic rows to generate.

        seed : int, optional
            Random seed for reproducible sampling.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        return self.generate(n_rows)
