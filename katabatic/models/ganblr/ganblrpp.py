import numpy as np

from katabatic.models.ganblr.models import GANBLR
from katabatic.models.ganblr.utils import DMMDiscretizer


class GANBLRPP:
    """GANBLR++ wrapper with support for numerical columns."""

    def __init__(self, numerical_columns, random_state=None):
        self.numerical_columns = list(numerical_columns)
        self.random_state = random_state
        self.discretizer = DMMDiscretizer(random_state=random_state)
        self.ganblr = GANBLR()
        self._is_fitted = False

    def fit(
        self,
        x,
        y,
        k=0,
        batch_size=32,
        epochs=150,
        warmup_epochs=1,
        verbose=1,
    ):
        """Fit GANBLR++ on mixed numerical and categorical data."""
        x_array = np.asarray(x).copy()

        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array")

        if any(
            index < 0 or index >= x_array.shape[1] for index in self.numerical_columns
        ):
            raise ValueError("numerical_columns contains an invalid column index")

        if self.numerical_columns:
            numerical_data = x_array[:, self.numerical_columns].astype(float)
            discrete_numerical = self.discretizer.fit_transform(numerical_data)

            x_array[:, self.numerical_columns] = discrete_numerical

        self.ganblr.fit(
            x_array,
            y,
            k=k,
            batch_size=batch_size,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            verbose=verbose,
        )

        self._is_fitted = True
        return self

    def sample(self, size=None, verbose=1, seed=None):
        """Generate synthetic data and restore numerical columns."""
        if not self._is_fitted:
            raise RuntimeError("GANBLRPP must be fitted before sampling")

        synthetic = self.ganblr.sample(
            size=size,
            verbose=verbose,
            seed=seed,
        )

        if not self.numerical_columns:
            return synthetic

        if hasattr(synthetic, "iloc"):
            numerical_data = synthetic.iloc[:, self.numerical_columns].to_numpy(
                dtype=float
            )

            restored = self.discretizer.inverse_transform(numerical_data)

            synthetic.iloc[:, self.numerical_columns] = restored
            return synthetic

        synthetic_array = np.asarray(synthetic).copy()

        numerical_data = synthetic_array[:, self.numerical_columns].astype(float)

        restored = self.discretizer.inverse_transform(numerical_data)

        synthetic_array[:, self.numerical_columns] = restored

        return synthetic_array
