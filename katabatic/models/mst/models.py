"""Maximum Spanning Tree (MST) model implementation for Katabatic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from katabatic.models.mst.utils import (
    load_training_data,
    resolve_synth_dir,
    save_metadata,
    save_synthetic_data,
)


class MSTModel(BaseModel):
    """
    Differentially private synthetic data generator using SmartNoise MST.

    MST models relationships between discrete attributes using a
    maximum spanning tree and generates synthetic records under
    differential privacy constraints.
    """

    def __init__(
        self,
        *,
        epsilon: float = 3.0,
        delta: float | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        super().__init__()

        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

        if delta is not None and not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1.")

        self.epsilon = epsilon
        self.delta = delta
        self.categorical_columns = categorical_columns

        self.synthesizer: Any | None = None
        self.column_names: list[str] | None = None
        self.label: str | None = None
        self._train_df: pd.DataFrame | None = None
        self._resolved_delta: float | None = None
        self._resolved_categorical_columns: list[str] = []

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return Python import names required by MST."""
        return ["snsynth", "mbi", "opendp"]

    @staticmethod
    def _apply_opendp_compatibility_patch() -> None:
        """
        Patch SmartNoise helpers for OpenDP versions requiring nan=False.

        SmartNoise 1.0.5 constructs floating-point OpenDP domains without
        explicitly disabling NaN values. Newer OpenDP versions require
        non-NaN domains when AbsoluteDistance is used by the Gaussian
        mechanism.

        The patch is applied only to the running Python process and does not
        modify SmartNoise or OpenDP source files.
        """
        import opendp.prelude as dp
        import snsynth.mst.mst as mst_module
        from opendp.measurements import make_gaussian

        def fixed_cdp_rho(
            epsilon: float,
            delta: float,
            max_contrib: int = 1,
        ) -> float:
            budget = (epsilon, delta)

            dp.enable_features(
                "floating-point",
                "contrib",
            )

            input_domain = dp.atom_domain(
                T=float,
                nan=False,
            )

            input_metric = dp.absolute_distance(
                T=float,
            )

            def make_adp_gauss(scale: float):
                test_gauss = make_gaussian(
                    input_domain,
                    input_metric,
                    scale,
                )

                adp = dp.c.make_zCDP_to_approxDP(
                    test_gauss,
                )

                return dp.c.make_fix_delta(
                    adp,
                    delta=delta,
                )

            discovered_scale = dp.binary_search_param(
                lambda scale: make_adp_gauss(scale),
                d_in=float(max_contrib),
                d_out=budget,
            )

            gaussian = make_gaussian(
                input_domain,
                input_metric,
                discovered_scale,
            )

            return gaussian.map(d_in=1.0)

        def fixed_gaussian_noise(
            sigma: float,
            size: int | None = None,
        ):
            dp.enable_features(
                "floating-point",
                "contrib",
            )

            input_domain = dp.atom_domain(
                T=float,
                nan=False,
            )

            input_metric = dp.absolute_distance(
                T=float,
            )

            measurement = make_gaussian(
                input_domain,
                input_metric,
                sigma,
            )

            if size is None:
                return measurement(0.0)

            return [
                measurement(0.0)
                for _ in range(size)
            ]

        mst_module.cdp_rho = fixed_cdp_rho
        mst_module.gaussian_noise = fixed_gaussian_noise

    @staticmethod
    def _infer_categorical_columns(
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Infer categorical columns from pandas dtypes.

        Object, category, and boolean columns are treated as categorical.
        """
        categorical_columns = []

        for column in df.columns:
            dtype = df[column].dtype

            if (
                dtype == "object"
                or str(dtype).startswith("category")
                or str(dtype) == "bool"
            ):
                categorical_columns.append(column)

        return categorical_columns

    def _resolve_delta(
        self,
        n_rows: int,
    ) -> float:
        """
        Resolve delta for approximate differential privacy.

        If no value is supplied, use 1 / (n * sqrt(n)).
        """
        if self.delta is not None:
            return self.delta

        if n_rows <= 0:
            raise ValueError(
                "Training data must contain at least one row."
            )

        return 1 / (n_rows * (n_rows ** 0.5))

    def train(
        self,
        data_dir: str,
        synthetic_dir: str | None = None,
        *args,
        **kwargs,
    ) -> MSTModel:
        """Fit the SmartNoise MST synthesizer and save synthetic output."""
        try:
            from snsynth import Synthesizer
        except ImportError as exc:
            raise ImportError(
                "SmartNoise Synth is not installed. "
                "Install the MST optional dependencies."
            ) from exc

        try:
            import mbi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Private-PGM is required by SmartNoise MST."
            ) from exc

        df = load_training_data(data_dir)

        if df.empty:
            raise ValueError(
                "Training data must not be empty."
            )

        if df.isnull().any().any():
            raise ValueError(
                "MST training data must not contain missing values."
            )

        self.column_names = df.columns.tolist()
        self.label = df.columns[-1]
        self._train_df = df.copy()

        categorical_columns = (
            list(self.categorical_columns)
            if self.categorical_columns is not None
            else self._infer_categorical_columns(df)
        )

        unknown_columns = [
            column
            for column in categorical_columns
            if column not in df.columns
        ]

        if unknown_columns:
            raise ValueError(
                "Categorical columns were not found in training data: "
                f"{unknown_columns}"
            )

        self._resolved_categorical_columns = categorical_columns
        self._resolved_delta = self._resolve_delta(len(df))

        self._apply_opendp_compatibility_patch()


        print(
            "[MST] Initializing with "
            f"epsilon={self.epsilon}, "
            f"delta={self._resolved_delta}..."
        )

        self.synthesizer = Synthesizer.create(
            "mst",
            epsilon=self.epsilon,
            delta=self._resolved_delta,
            verbose=False,
        )

        self.synthesizer.fit(
            df,
            categorical_columns=categorical_columns,
        )

        self.is_fitted = True

        n_generated = len(df)
        synthetic_df = self.sample(
            n=n_generated,
        )

        synth_dir = resolve_synth_dir(
            synthetic_dir,
            data_dir,
            "mst",
        )

        x_path, y_path = save_synthetic_data(
            synthetic_df,
            self.label,
            synth_dir,
        )

        save_metadata(
            synth_dir=synth_dir,
            df=df,
            label=self.label,
            epsilon=self.epsilon,
            delta=self._resolved_delta,
            categorical_columns=categorical_columns,
            n_generated=n_generated,
        )

        print(
            "[MST] Synthetic data saved:\n"
            f"  X -> {x_path}\n"
            f"  y -> {y_path}"
        )

        return self

    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> float:
        """Return a placeholder evaluation score for pipeline compatibility."""
        if not self.is_fitted:
            raise RuntimeError(
                "Call train() before evaluate()."
            )

        return 0.0

    def sample(
        self,
        n: int | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic samples from the fitted MST model."""
        if not self.is_fitted or self.synthesizer is None:
            raise RuntimeError(
                "Call train() before sample()."
            )

        if n is None:
            if self._train_df is None:
                raise RuntimeError(
                    "Training data is unavailable."
                )

            n = len(self._train_df)

        if n <= 0:
            raise ValueError(
                "n must be greater than 0."
            )

        synthetic = self.synthesizer.sample(
            int(n),
        )

        if isinstance(synthetic, pd.DataFrame):
            return synthetic.reset_index(
                drop=True,
            )

        if self.column_names is None:
            raise RuntimeError(
                "Column metadata is unavailable."
            )

        return pd.DataFrame(
            synthetic,
            columns=self.column_names,
        )
