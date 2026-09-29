from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from katabatic.artifacts.base import ArtifactStore
from katabatic.artifacts.refs import ModelRef
from katabatic.models.base_model import Model

from .utils import (
    PrivTreeSampler,
    PrivTreeSplitter,
    TreeNode,
    clip_nonnegative,
    dp_count,
)


class PrivTreeModel(Model):
    """
    Differentially Private Tree-based synthetic data generator.
    """

    ARTIFACT_STATE_FILES = ("privtree_model.pkl",)

    def __init__(
        self,
        epsilon: float = 1.0,
        max_depth: int = 5,
        min_count: int = 10,
        random_state: int = 42,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.max_depth = max_depth
        self.min_count = min_count
        self.random_state = random_state

        self.epsilon_per_level = epsilon / max_depth
        self.rng = np.random.default_rng(random_state)

        self.root: TreeNode | None = None
        self.columns: list[str] | None = None
        self._n_train_rows: int | None = None

        self.splitter = PrivTreeSplitter(
            max_depth=max_depth,
            min_count=min_count,
            epsilon_per_level=self.epsilon_per_level,
            random_state=random_state,
        )

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["numpy", "pandas"]

    def fit(self, X: pd.DataFrame, y=None) -> PrivTreeModel:
        """
        Fit PrivTree on the provided categorical / binned dataframe. X is the combined (feature + target), y is unused.
        """
        if X.empty:
            raise ValueError("Input dataframe is empty.")

        self.columns = list(X.columns)

        true_count = X.shape[0]
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

        self._build_tree(X, self.root)
        self.is_fitted = True
        self._n_train_rows = true_count

        return self

    def train(
        self,
        data_dir: str,
        *args,
        synthetic_dir: str | None = None,
        artifact_state_dir: str | None = None,
        n_synth: int | None = None,
        **kwargs,
    ) -> PrivTreeModel:
        """
        Train PrivTree using Katabatic's model interface.
        """
        self.check_dependencies()

        train_path = os.path.join(data_dir, "train_full.csv")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")

        df = pd.read_csv(train_path)
        self.fit(df)

        if synthetic_dir:
            os.makedirs(synthetic_dir, exist_ok=True)

            if n_synth is None:
                n_synth = len(df)

            synthetic = self.sample(n_synth)

            # Katabatic uses the last column as the target.
            x_synth = synthetic.iloc[:, :-1]
            y_synth = synthetic.iloc[:, -1]

            x_synth.to_csv(
                os.path.join(synthetic_dir, "x_synth.csv"),
                index=False,
            )
            y_synth.to_csv(
                os.path.join(synthetic_dir, "y_synth.csv"),
                index=False,
            )

        self._maybe_save_artifact_state(artifact_state_dir)

        return self

    def evaluate(self, X_real=None, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")

        raise NotImplementedError(
            "PrivTreeModel.evaluate() has no meaningful standalone metric to offer."
            "Use TSTREvaluation for cross-model metrics instead."
        )

    def _save_artifact_state(self, artifact_state_dir: str) -> None:
        os.makedirs(artifact_state_dir, exist_ok=True)

        state = {
            "epsilon": self.epsilon,
            "max_depth": self.max_depth,
            "min_count": self.min_count,
            "random_state": self.random_state,
            "epsilon_per_level": self.epsilon_per_level,
            "root": self.root,
            "columns": self.columns,
            "is_fitted": self.is_fitted,
            "n_train_rows": self._n_train_rows,
        }

        target = os.path.join(
            artifact_state_dir,
            self.ARTIFACT_STATE_FILES[0],
        )

        with open(target, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_ref(
        cls,
        store: ArtifactStore,
        ref: ModelRef,
    ) -> PrivTreeModel:
        state_path = cls._require_state_file(store, ref)

        with open(state_path, "rb") as f:
            state = pickle.load(f)  # nosec B301: loading our own saved model artifact

        instance = cls(
            epsilon=state["epsilon"],
            max_depth=state["max_depth"],
            min_count=state["min_count"],
            random_state=state["random_state"],
        )

        instance.epsilon_per_level = state["epsilon_per_level"]
        instance.root = state["root"]
        instance.columns = state["columns"]
        instance.is_fitted = state["is_fitted"]
        instance._n_train_rows = state.get("n_train_rows")

        return instance

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
        n_samples: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the Katabatic sampling interface.

        Defaults to the number of rows the model was trained on when
        n_samples is omitted.
        """
        if self.root is None or self.columns is None:
            raise ValueError("Model not fitted yet.")

        if n_samples is None:
            n_samples = self._n_train_rows or 100

        if seed is None:
            return self.generate(n_samples)

        sampler = PrivTreeSampler(
            root=self.root,
            columns=self.columns,
            random_state=seed,
        )

        return sampler.generate(n_samples)
