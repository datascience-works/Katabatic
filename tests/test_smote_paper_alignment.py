import numpy as np

from katabatic.models.smote.models import _SMOTE


class ZeroLambdaRNG:
    """
    Small deterministic RNG helper used only to expose which anchors
    _interpolate() selects.

    Returning lambda=0 means each generated point is exactly its anchor.
    """

    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)

    def integers(self, *args, **kwargs):
        return self._rng.integers(*args, **kwargs)

    def permutation(self, *args, **kwargs):
        return self._rng.permutation(*args, **kwargs)

    def uniform(self, low=0.0, high=1.0, size=None):
        return np.zeros(size)


def test_smote_default_k_matches_paper():
    smote = _SMOTE()

    assert smote.k_neighbors == 5


def test_complete_passes_use_each_minority_anchor_equally():
    X_cls = np.array([
        [0.0],
        [10.0],
        [20.0],
        [30.0],
    ])

    # Neighbour identities do not affect this test because lambda=0.
    neighbour_indices = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ])

    smote = _SMOTE(
        k_neighbors=3,
        random_state=42,
    )

    smote._rng = ZeroLambdaRNG(seed=42)

    generated = smote._interpolate(
        X_cls,
        neighbour_indices,
        n_needed=8,
    )

    values, counts = np.unique(
        generated[:, 0],
        return_counts=True,
    )

    assert np.array_equal(
        values,
        np.array([0.0, 10.0, 20.0, 30.0]),
    )

    # 8 generated samples / 4 anchors = 2 complete passes.
    assert np.array_equal(
        counts,
        np.array([2, 2, 2, 2]),
    )


def test_partial_pass_uses_unique_minority_anchors():
    X_cls = np.array([
        [0.0],
        [10.0],
        [20.0],
        [30.0],
    ])

    neighbour_indices = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ])

    smote = _SMOTE(
        k_neighbors=3,
        random_state=42,
    )

    smote._rng = ZeroLambdaRNG(seed=42)

    generated = smote._interpolate(
        X_cls,
        neighbour_indices,
        n_needed=2,
    )

    # A partial pass should select two different minority anchors.
    assert len(generated) == 2
    assert len(np.unique(generated[:, 0])) == 2


def test_fit_resample_preserves_requested_class_balance():
    rng = np.random.default_rng(7)

    majority = rng.normal(
        loc=4.0,
        scale=0.5,
        size=(18, 2),
    )

    minority = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [0.9, 1.3],
        [1.4, 0.8],
        [0.8, 0.9],
        [1.1, 1.5],
    ])

    X = np.vstack([majority, minority])
    y = np.array(
        [0] * len(majority)
        + [1] * len(minority)
    )

    smote = _SMOTE(
        k_neighbors=5,
        sampling_strategy="auto",
        random_state=42,
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    classes, counts = np.unique(
        y_resampled,
        return_counts=True,
    )

    assert X_resampled.shape == (36, 2)
    assert np.array_equal(classes, np.array([0, 1]))
    assert np.array_equal(counts, np.array([18, 18]))


def test_same_seed_is_reproducible():
    rng = np.random.default_rng(7)

    majority = rng.normal(
        loc=4.0,
        scale=0.5,
        size=(18, 2),
    )

    minority = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [0.9, 1.3],
        [1.4, 0.8],
        [0.8, 0.9],
        [1.1, 1.5],
    ])

    X = np.vstack([majority, minority])
    y = np.array(
        [0] * len(majority)
        + [1] * len(minority)
    )

    first = _SMOTE(
        k_neighbors=5,
        sampling_strategy="auto",
        random_state=42,
    )

    second = _SMOTE(
        k_neighbors=5,
        sampling_strategy="auto",
        random_state=42,
    )

    X_first, y_first = first.fit_resample(X, y)
    X_second, y_second = second.fit_resample(X, y)

    assert np.allclose(X_first, X_second)
    assert np.array_equal(y_first, y_second)
