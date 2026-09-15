import traceback

import pandas as pd

from katabatic.evaluate.consistency import ConsistencyEvaluation
from katabatic.evaluate.diversity import DiversityEvaluation
from katabatic.evaluate.fidelity import FidelityEvaluation
from katabatic.evaluate.privacy import PrivacyEvaluation
from katabatic.evaluate.report.composite import EvaluationReport
from katabatic.evaluate.stability import StabilityEvaluation
from katabatic.evaluate.utility import UtilityEvaluation

_AVAILABLE_DIMENSIONS = [
    "fidelity",
    "utility",
    "diversity",
    "privacy",
    "consistency",
    "stability",
]


class SyntheticEvaluationPipeline:
    """
    Orchestrates all evaluation dimensions for a synthetic dataset and
    produces a single weighted composite score.

    Parameters
    ----------
    dimensions : list[str], optional
        Which dimensions to run. Defaults to all 5 non-stability dimensions.
        Pass 'stability' explicitly to enable it (requires model in run()).
    weights : dict[str, float], optional
        Override the default composite score weights. Any dimension not
        listed falls back to DEFAULT_WEIGHTS.
    categorical_cols : list[str], optional
        Shared column type hint passed to Fidelity and Diversity evaluators.
        Auto-detected from dtypes if omitted.
    continuous_cols : list[str], optional
        Shared column type hint passed to Fidelity and Diversity evaluators.
        Auto-detected from dtypes if omitted.
    n_folds : int
        CV folds used by Utility and Consistency evaluators (default: 5).
    n_bins : int
        Number of equal-width bins for Diversity bin coverage (default: 10).
    gower_sample_size : int
        Sample size for Gower distance computation (default: 200).
    near_dup_threshold : float
        Distance threshold for near-duplicate detection (default: 0.01).
    privacy_sample_size : int
        Max synthetic rows used in NNDR computation (default: 2000).
    n_stability_runs : int
        Number of sampling runs for the Stability evaluator (default: 5).
    stability_seeds : list[int], optional
        Seeds for each stability run. Defaults to [0, 1, ..., n_stability_runs-1].
    """

    def __init__(
        self,
        dimensions: list = None,
        weights: dict = None,
        categorical_cols: list = None,
        continuous_cols: list = None,
        n_folds: int = 5,
        n_bins: int = 10,
        gower_sample_size: int = 200,
        near_dup_threshold: float = 0.01,
        privacy_sample_size: int = 2000,
        n_stability_runs: int = 5,
        stability_seeds: list = None,
    ):
        # Default: all dimensions except stability (stability requires a model)
        self.dimensions = dimensions or [
            d for d in _AVAILABLE_DIMENSIONS if d != "stability"
        ]
        self.weights = weights or {}
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.n_folds = n_folds
        self.n_bins = n_bins
        self.gower_sample_size = gower_sample_size
        self.near_dup_threshold = near_dup_threshold
        self.privacy_sample_size = privacy_sample_size
        self.n_stability_runs = n_stability_runs
        self.stability_seeds = stability_seeds

        unknown = [d for d in self.dimensions if d not in _AVAILABLE_DIMENSIONS]
        if unknown:
            raise ValueError(
                f"Unknown dimensions: {unknown}. Available: {_AVAILABLE_DIMENSIONS}"
            )

        if categorical_cols is None or continuous_cols is None:
            print(
                "[WARNING] categorical_cols and/or continuous_cols not provided. "
                "Column types will be auto-detected from dtypes — this may be inaccurate "
                "for integer-encoded categorical columns. Pass them explicitly for reliable results."
            )

    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_col: str = None,
        test_data: pd.DataFrame = None,
        constraints: dict = None,
        model=None,
        output_dir: str = None,
        report_prefix: str = "",
    ) -> EvaluationReport:
        """
        Run all configured evaluation dimensions and return an EvaluationReport.
        Parameters
        ----------
        real_data : pd.DataFrame
            The real (training) dataset.
        synthetic_data : pd.DataFrame
            The synthetic dataset to evaluate.
        target_col : str, optional
            Target column name. Required for Utility and Consistency dimensions.
        test_data : pd.DataFrame, optional
            Held-out test split. When provided, both TSTR and TRTR in the
            Utility dimension test on this set for a fair comparison.
        constraints : dict[str, tuple], optional
            Per-column bounds for Consistency constraint checking.
            Format: {'col': (min_val, max_val)}. Use None for unbounded side.
        model : Model, optional
            A trained model instance. Required only when 'stability' is in
            dimensions. Must implement sample(n_samples, seed=int).
        output_dir : str, optional
            If provided, saves the full JSON report and CSV summary here.
        report_prefix : str
            Optional filename prefix for saved report files.
        """
        if target_col is None:
            target_col = real_data.columns[-1]
            print(
                f"[WARNING] target_col not provided. "
                f"Assuming last column '{target_col}' is the target. "
                f"Pass target_col explicitly if this is incorrect."
            )

        self._validate_inputs(real_data, synthetic_data, target_col, model)

        # Feature-only views used by evaluators that don't need the target
        if target_col and target_col in real_data.columns:
            real_features = real_data.drop(columns=[target_col])
            synth_features = synthetic_data.drop(columns=[target_col])
        else:
            real_features = real_data
            synth_features = synthetic_data

        dimension_results = {}

        for dim in self.dimensions:
            print(f"\nRunning {dim} evaluation...")
            try:
                # These dimensions operate on features only (no target column).
                # Stability generates its own synthetic data internally so it
                # also receives real_features — the target is not needed.
                if dim in ("fidelity", "diversity", "privacy", "stability"):
                    r, s = real_features, synth_features
                else:
                    r, s = real_data, synthetic_data

                evaluator = self._build_evaluator(
                    dim, r, s, target_col, constraints, model, test_data
                )
                dimension_results[dim] = evaluator.evaluate()
            except Exception as e:
                print(f"  [ERROR] {dim} evaluation failed: {e}")
                traceback.print_exc()
                dimension_results[dim] = {
                    "error": str(e),
                    f"{dim}_score": 0.0,
                }

        report = EvaluationReport(dimension_results, weights=self.weights)
        report.print_summary()

        if output_dir:
            report.save(output_dir, prefix=report_prefix)

        return report

    def _build_evaluator(
        self,
        dim,
        real_data,
        synthetic_data,
        target_col,
        constraints,
        model=None,
        test_data=None,
    ):
        shared = dict(real_data=real_data, synthetic_data=synthetic_data)

        if dim == "fidelity":
            return FidelityEvaluation(
                **shared,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
            )

        if dim == "utility":
            if not target_col:
                raise ValueError("'target_col' is required for the utility dimension.")
            return UtilityEvaluation(
                **shared,
                target_col=target_col,
                test_data=test_data,
                n_folds=self.n_folds,
            )

        if dim == "diversity":
            return DiversityEvaluation(
                **shared,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
                n_bins=self.n_bins,
                gower_sample_size=self.gower_sample_size,
            )

        if dim == "privacy":
            return PrivacyEvaluation(
                **shared,
                near_dup_threshold=self.near_dup_threshold,
                sample_size=self.privacy_sample_size,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
            )

        if dim == "consistency":
            if not target_col:
                raise ValueError(
                    "'target_col' is required for the consistency dimension."
                )
            return ConsistencyEvaluation(
                **shared,
                target_col=target_col,
                constraints=constraints,
                n_folds=self.n_folds,
            )

        if dim == "stability":
            return StabilityEvaluation(
                real_data=real_data,
                model=model,
                n_runs=self.n_stability_runs,
                seeds=self.stability_seeds,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
            )

        raise ValueError(f"No evaluator registered for dimension '{dim}'.")

    def _validate_inputs(self, real_data, synthetic_data, target_col, model=None):
        if not isinstance(real_data, pd.DataFrame):
            raise TypeError("real_data must be a pandas DataFrame.")
        if not isinstance(synthetic_data, pd.DataFrame):
            raise TypeError("synthetic_data must be a pandas DataFrame.")
        if real_data.empty:
            raise ValueError("real_data is empty.")
        if synthetic_data.empty:
            raise ValueError("synthetic_data is empty.")

        needs_target = {"utility", "consistency"} & set(self.dimensions)
        if needs_target and not target_col:
            raise ValueError(
                f"'target_col' is required when running dimensions: {needs_target}. "
                f"Pass it explicitly or ensure real_data has a last column to fall back to."
            )

        if "stability" in self.dimensions and model is None:
            raise ValueError(
                "'model' must be provided when 'stability' is in dimensions. "
                "Pass a trained model instance to pipeline.run(model=...)."
            )
