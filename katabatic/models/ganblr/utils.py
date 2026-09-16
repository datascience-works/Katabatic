import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from pandas import read_csv


class softmax_weight(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be under softmax (vectorised)."""

    def __init__(self, feature_uniques):
        if isinstance(feature_uniques, np.ndarray):
            feature_idxs = np.concatenate([[0], np.cumsum(feature_uniques)])
        else:
            feature_idxs = np.concatenate([[0], np.cumsum(np.asarray(feature_uniques))])
        feature_idxs = feature_idxs.astype(np.int64)

        # Build a segment-id vector: each row of w gets a feature group id
        seg_ids = np.zeros(int(feature_idxs[-1]), dtype=np.int64)
        for i in range(len(feature_idxs) - 1):
            seg_ids[feature_idxs[i] : feature_idxs[i + 1]] = i

        self.feature_idxs = feature_idxs.tolist()
        self.seg_ids = tf.constant(seg_ids)
        self.n_segments = int(len(feature_idxs) - 1)

    def __call__(self, w):
        # Compute group-wise log-softmax in one vectorised pass.
        seg_max = tf.math.unsorted_segment_max(w, self.seg_ids, self.n_segments)
        # Broadcast group max back to each row
        max_per_row = tf.gather(seg_max, self.seg_ids)
        shifted = w - max_per_row
        exp_shifted = tf.exp(shifted)
        seg_sum = tf.math.unsorted_segment_sum(
            exp_shifted, self.seg_ids, self.n_segments
        )
        logsumexp_per_row = tf.gather(tf.math.log(seg_sum), self.seg_ids) + max_per_row
        return w - logsumexp_per_row

    def get_config(self):
        return {"feature_idxs": self.feature_idxs}


def elr_loss(KL_LOSS):
    def loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + KL_LOSS

    return loss


def KL_loss(prob_fake):
    return np.mean(-np.log(np.subtract(1, prob_fake)))


def get_lr(input_dim, output_dim, constraint=None, KL_LOSS=0):
    model = tf.keras.Sequential()
    # declared input shape to keras
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(
        tf.keras.layers.Dense(
            output_dim, activation="softmax", kernel_constraint=constraint
        )
    )
    model.compile(loss=elr_loss(KL_LOSS), optimizer="adam", metrics=["accuracy"])
    # log_elr = model.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=epochs)
    return model


def sample(*arrays, n=None, frac=None, random_state=None):
    """
    generate sample random arrays from given arrays. The given arrays must be same size.

    Parameters:
    --------------
    *arrays: arrays to be sampled.

    n (int): Number of random samples to generate.

    frac: Float value between 0 and 1, Returns (float value * length of given arrays). frac cannot be used with n.

    random_state: int value or numpy.random.RandomState, optional. if set to a particular integer, will return same samples in every iteration.

    Return:
    --------------
    the sampled array(s). Passing in multiple arrays will result in the return of a tuple.

    """
    random = np.random
    if isinstance(random_state, int):
        random = random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random = random_state

    arr0 = arrays[0]
    original_size = len(arr0)
    if n is None and frac is None:
        raise Exception("You must specify one of frac or size.")
    if n is None:
        n = int(len(arr0) * frac)

    idxs = random.choice(original_size, n, replace=False)
    if len(arrays) > 1:
        sampled_arrays = []
        for arr in arrays:
            assert len(arr) == original_size
            sampled_arrays.append(arr[idxs])
        return tuple(sampled_arrays)
    else:
        return arr0[idxs]


DEMO_DATASETS = {
    "adult": {
        "link": "https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv",
        "params": {"dtype": int},
    },
    "adult-raw": {
        "link": "https://drive.google.com/uc?export=download&id=1iA-_qIC1xKQJ4nL2ugX1_XJQf8__xOY0",
        "params": {},
    },
}


def get_demo_data(name="adult"):
    """
    Download demo dataset from internet.

    Parameters
    ----------
    name : str
        Name of dataset. Should be one of ['adult', 'adult-raw'].

    Returns
    -------
    data : pandas.DataFrame
        the demo dataset.
    """
    assert name in DEMO_DATASETS.keys()
    return read_csv(DEMO_DATASETS[name]["link"], **DEMO_DATASETS[name]["params"])


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


