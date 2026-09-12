"""GANBLR++ support for mixed numerical and categorical tabular data."""

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler

from katabatic.models.ganblr.models import GANBLR


class DMMDiscretizer:
    """Discretize numerical columns using Bayesian Gaussian mixtures."""

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = []
        self.encoders = []
        self.means = []
        self.stds = []

    def fit_transform(self, x):
        """Fit the numerical discretizer and return discrete component IDs."""
        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array")

        x_scaled = self.scaler.fit_transform(x)

        self.models.clear()
        self.encoders.clear()
        self.means.clear()
        self.stds.clear()

        transformed = np.zeros_like(x_scaled, dtype=int)

        for column_index in range(x_scaled.shape[1]):
            column = x_scaled[:, column_index : column_index + 1]

            model = BayesianGaussianMixture(
                n_components=8,
                random_state=self.random_state,
            )

            components = model.fit_predict(column)

            encoder = LabelEncoder()
            labels = encoder.fit_transform(components)

            transformed[:, column_index] = labels

            component_ids = encoder.classes_
            means = model.means_.reshape(-1)[component_ids]
            stds = np.sqrt(model.covariances_.reshape(-1)[component_ids])

            self.models.append(model)
            self.encoders.append(encoder)
            self.means.append(means)
            self.stds.append(stds)

        return transformed

    def inverse_transform(self, x):
        """Convert discrete component IDs back to numerical values."""
        x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array")

        restored = np.zeros_like(x, dtype=float)

        for column_index in range(x.shape[1]):
            labels = x[:, column_index].astype(int)
            means = self.means[column_index]
            stds = self.stds[column_index]

            sampled = np.zeros(len(labels), dtype=float)

            for label in np.unique(labels):
                mask = labels == label
                mean = means[label]
                std = stds[label]

                if std <= 0:
                    sampled[mask] = mean
                else:
                    sampled[mask] = np.random.normal(
                        loc=mean,
                        scale=std,
                        size=np.sum(mask),
                    )

            restored[:, column_index] = sampled

        return self.scaler.inverse_transform(restored)


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
