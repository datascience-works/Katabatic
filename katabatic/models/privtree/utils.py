from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TreeNode:
    """
    A single node in the PrivTree.

    We store:
    - depth: depth of node (root = 0)
    - noisy_count: DP-protected count of records reaching this node
    - split_feature: column name used to split at this node (None if leaf)
    - split_groups: (left_values, right_values) for a categorical/binned split
    - children: dict with keys "left" and "right" mapping to child nodes
    - leaf_distributions: per-column categorical distribution for sampling at leaf
    """

    depth: int
    noisy_count: float

    split_feature: str | None = None
    split_groups: tuple[set, set] | None = None

    children: dict[str, TreeNode] = field(default_factory=dict)

    # Only populated for leaf nodes
    leaf_distributions: dict[str, dict[Any, float]] | None = None

    def is_leaf(self) -> bool:
        return self.split_feature is None

    def set_split(
        self,
        feature: str,
        left_values: set,
        right_values: set,
    ) -> None:
        self.split_feature = feature
        self.split_groups = (set(left_values), set(right_values))
        self.leaf_distributions = None  # once split, it's not a leaf

    def set_leaf_distributions(
        self,
        distributions: dict[str, dict[Any, float]],
    ) -> None:
        self.leaf_distributions = distributions
        self.split_feature = None
        self.split_groups = None
        self.children = {}


def laplace_noise(
    scale: float,
    rng: np.random.Generator,
) -> float:
    """
    Generate Laplace(0, scale) noise using a numpy Generator.
    """
    return rng.laplace(loc=0.0, scale=scale)


def dp_count(
    true_count: int,
    epsilon: float,
    rng: np.random.Generator,
    sensitivity: float = 1.0,
) -> float:
    """
    Differentially private count using Laplace mechanism.

    noisy_count = true_count + Laplace(0, sensitivity/epsilon)

    - sensitivity for counts is usually 1
    - epsilon must be > 0
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for DP noise.")

    scale = sensitivity / epsilon
    return float(true_count) + laplace_noise(
        scale=scale,
        rng=rng,
    )


def clip_nonnegative(x: float) -> float:
    """
    Counts should not be negative after noise. We clip at 0.
    """
    return float(max(0.0, x))


class PrivTreeSplitter:
    """
    Handles split decisions for PrivTree nodes.
    """

    def __init__(
        self,
        max_depth: int,
        min_count: int,
        epsilon_per_level: float,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_count = min_count
        self.epsilon = epsilon_per_level
        self.rng = np.random.default_rng(random_state)

    def should_split(self, node: TreeNode) -> bool:
        """
        Decide whether a node should be split further.
        """
        if node.depth >= self.max_depth:
            return False

        if node.noisy_count < self.min_count:
            return False

        return True

    def choose_split(
        self,
        df: pd.DataFrame,
        node: TreeNode,
    ) -> tuple[str, set, set] | None:
        """
        Choose the best split for the current node.

        Returns:
            (feature_name, left_values, right_values)
        or None if no valid split exists.
        """
        best_feature = None
        best_left = None
        best_right = None
        best_score = -np.inf

        for col in df.columns:
            values = list(df[col].unique())

            if len(values) <= 1:
                continue  # no split possible

            # Randomly split categories into two groups
            self.rng.shuffle(values)

            mid = len(values) // 2
            left_vals = set(values[:mid])
            right_vals = set(values[mid:])

            if len(left_vals) == 0 or len(right_vals) == 0:
                continue

            # True counts
            left_count = df[df[col].isin(left_vals)].shape[0]
            right_count = df[df[col].isin(right_vals)].shape[0]

            # DP noisy counts
            noisy_left = clip_nonnegative(
                dp_count(
                    left_count,
                    self.epsilon,
                    self.rng,
                )
            )

            noisy_right = clip_nonnegative(
                dp_count(
                    right_count,
                    self.epsilon,
                    self.rng,
                )
            )

            # Score: prefer balanced splits
            score = -abs(noisy_left - noisy_right)

            if score > best_score:
                best_score = score
                best_feature = col
                best_left = left_vals
                best_right = right_vals

        if best_feature is None:
            return None

        return best_feature, best_left, best_right

    def split_dataframe(
        self,
        df: pd.DataFrame,
        feature: str,
        left_vals: set,
        right_vals: set,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into left and right subsets.
        """
        left_df = df[df[feature].isin(left_vals)].copy()
        right_df = df[df[feature].isin(right_vals)].copy()

        return left_df, right_df


class PrivTreeSampler:
    """
    Generates synthetic data from a trained PrivTree.
    """

    def __init__(
        self,
        root: TreeNode,
        columns: list[str],
        random_state: int = 42,
    ):
        if root is None:
            raise ValueError("PrivTree root node is None. Did you call fit()?")

        self.root = root
        self.columns = columns
        self.rng = np.random.default_rng(random_state)

    def generate(self, n_rows: int) -> pd.DataFrame:
        """
        Generate synthetic dataset with n_rows rows.
        """
        rows = []

        for _ in range(n_rows):
            row = self._sample_single_row()
            rows.append(row)

        return pd.DataFrame(
            rows,
            columns=self.columns,
        )

    def _sample_single_row(self) -> dict:
        """
        Sample a single synthetic row by traversing the tree.
        """
        node = self.root

        # Traverse until leaf
        while not node.is_leaf():
            # Randomly choose branch (balanced traversal)
            if self.rng.random() < 0.5:
                node = node.children["left"]
            else:
                node = node.children["right"]

        # Sample values from leaf distributions
        sampled_row = {}

        for col, dist in node.leaf_distributions.items():
            values = list(dist.keys())
            probs = list(dist.values())

            sampled_row[col] = self.rng.choice(
                values,
                p=probs,
            )

        return sampled_row
