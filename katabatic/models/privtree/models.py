from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    PrivTreeSampler,
    PrivTreeSplitter,
    TreeNode,
    clip_nonnegative,
    dp_count,
)


class PrivTreeModel:
    """
    Differentially Private Tree-based synthetic data generator.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        max_depth: int = 5,
        min_count: int = 10,
        random_state: int = 42,
    ):
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.min_count = min_count
        self.random_state = random_state

        self.epsilon_per_level = epsilon / max_depth
        self.rng = np.random.default_rng(random_state)

        self.root: TreeNode | None = None
        self.columns: list[str] | None = None

        self.splitter = PrivTreeSplitter(
            max_depth=max_depth,
            min_count=min_count,
            epsilon_per_level=self.epsilon_per_level,
            random_state=random_state,
        )

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit PrivTree on the provided (categorical / binned) dataframe.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        self.columns = list(df.columns)

        true_count = df.shape[0]
        noisy_count = clip_nonnegative(
            dp_count(
                true_count,
                self.epsilon_per_level,
                self.rng,
            )
        )

        self.root = TreeNode(
            depth=0,
            noisy_count=noisy_count,
        )

        self._build_tree(df, self.root)

    def _build_tree(
        self,
        df: pd.DataFrame,
        node: TreeNode,
    ) -> None:
        """
        Recursively build the PrivTree.
        """
        if not self.splitter.should_split(node):
            node.set_leaf_distributions(self._compute_leaf_distributions(df))
            return

        split = self.splitter.choose_split(
            df,
            node,
        )

        if split is None:
            node.set_leaf_distributions(self._compute_leaf_distributions(df))
            return

        feature, left_vals, right_vals = split

        node.set_split(
            feature,
            left_vals,
            right_vals,
        )

        left_df, right_df = self.splitter.split_dataframe(
            df,
            feature,
            left_vals,
            right_vals,
        )

        left_count = clip_nonnegative(
            dp_count(
                left_df.shape[0],
                self.epsilon_per_level,
                self.rng,
            )
        )

        right_count = clip_nonnegative(
            dp_count(
                right_df.shape[0],
                self.epsilon_per_level,
                self.rng,
            )
        )

        left_child = TreeNode(
            depth=node.depth + 1,
            noisy_count=left_count,
        )

        right_child = TreeNode(
            depth=node.depth + 1,
            noisy_count=right_count,
        )

        node.children["left"] = left_child
        node.children["right"] = right_child

        self._build_tree(
            left_df,
            left_child,
        )

        self._build_tree(
            right_df,
            right_child,
        )

    def _compute_leaf_distributions(
        self,
        df: pd.DataFrame,
    ) -> dict[str, dict[Any, float]]:
        """
        Compute per-column categorical distributions for leaf node.
        """
        distributions: dict[str, dict[Any, float]] = {}

        for col in df.columns:
            value_counts = df[col].value_counts(normalize=True)

            distributions[col] = value_counts.to_dict()

        return distributions

    def generate(
        self,
        n_rows: int,
    ) -> pd.DataFrame:
        """
        Generate synthetic data after the tree has been trained.
        """
        if self.root is None or self.columns is None:
            raise ValueError("Model not fitted yet.")

        sampler = PrivTreeSampler(
            root=self.root,
            columns=self.columns,
            random_state=self.random_state,
        )

        return sampler.generate(n_rows)

    def sample(
        self,
        n_rows: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the Katabatic sampling interface.
        """
        if self.root is None or self.columns is None:
            raise ValueError("Model not fitted yet.")

        if seed is None:
            return self.generate(n_rows)

        sampler = PrivTreeSampler(
            root=self.root,
            columns=self.columns,
            random_state=seed,
        )

        return sampler.generate(n_rows)
