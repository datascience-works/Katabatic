import numpy as np
from sklearn.utils import check_random_state

from katabatic.models.smote.models import (
    SMOTEModel,
    _PaperAlignedSamplingMixin,
    _paper_aligned_anchor_rows,
)


def test_smote_default_k_matches_paper():
    model = SMOTEModel()
    assert model.k_neighbors == 5


def test_complete_passes_use_each_minority_anchor_equally():
    rng = check_random_state(42)

    rows = _paper_aligned_anchor_rows(
        n_anchors=4,
        n_samples=8,
        random_state=rng,
    )

    expected = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(rows, expected)


def test_partial_pass_uses_unique_minority_anchors():
    rng = check_random_state(42)

    rows = _paper_aligned_anchor_rows(
        n_anchors=6,
        n_samples=3,
        random_state=rng,
    )

    assert len(rows) == 3
    assert len(np.unique(rows)) == 3
    assert np.all((rows >= 0) & (rows < 6))


def test_complete_pass_then_partial_pass():
    rng = check_random_state(42)

    rows = _paper_aligned_anchor_rows(
        n_anchors=4,
        n_samples=6,
        random_state=rng,
    )

    np.testing.assert_array_equal(rows[:4], np.array([0, 1, 2, 3]))
    assert len(np.unique(rows[4:])) == 2


def test_same_seed_is_reproducible():
    rows_a = _paper_aligned_anchor_rows(
        n_anchors=10,
        n_samples=6,
        random_state=check_random_state(42),
    )

    rows_b = _paper_aligned_anchor_rows(
        n_anchors=10,
        n_samples=6,
        random_state=check_random_state(42),
    )

    np.testing.assert_array_equal(rows_a, rows_b)
