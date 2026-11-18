import csv
import json
import math
import os

# Score key returned by each evaluator's result dict
SCORE_KEYS = {
    "fidelity": "fidelity_score",
    "utility": "utility_score",
    "diversity": "diversity_score",
    "privacy": "privacy_score",
    "consistency": "consistency_score",
    "stability": "stability_score",
}

DEFAULT_WEIGHTS = {
    "utility": 0.35,
    "fidelity": 0.25,
    "privacy": 0.15,
    "diversity": 0.10,
    "consistency": 0.10,
    "stability": 0.05,
}


class EvaluationReport:
    """
    Holds the results of all evaluation dimensions and computes a single
    weighted composite score.

    The composite score is a weighted average of whichever dimensions were
    actually run. Weights are re-normalised against the active dimensions so
    the score is always in [0, 1] even when some dimensions are skipped.
    Dimensions that failed are listed in ``errors`` and left out of the
    composite; if none succeeded, the composite is NaN.

    Parameters
    ----------
    dimension_results : dict[str, dict]
        Mapping of dimension name to the result dict returned by its evaluator.
    weights : dict[str, float], optional
        Per-dimension weights. Falls back to DEFAULT_WEIGHTS for any missing key.
    """

    def __init__(self, dimension_results: dict, weights: dict = None):
        self.dimension_results = dimension_results
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

        self.dimension_scores = self._extract_scores()
        self.successful_dimensions = list(self.dimension_scores.keys())
        self.successful_dimension_count = len(self.successful_dimensions)
        self.total_dimensions = len(SCORE_KEYS)
        self.errors = {
            dim: result["error"]
            for dim, result in dimension_results.items()
            if "error" in result
        }

        self.failed_dimensions = list(self.errors)
        self.skipped_dimensions = [
            dim
            for dim in SCORE_KEYS
            if dim not in self.dimension_scores and dim not in self.errors
        ]

        self.composite_score = self._compute_composite()

    def _extract_scores(self) -> dict:
        scores = {}
        for dim, result in self.dimension_results.items():
            key = SCORE_KEYS.get(dim)
            if key and key in result:
                scores[dim] = result[key]
        return scores

    def _compute_composite(self) -> float:
        active = {d: self.weights.get(d, 0.0) for d in self.dimension_scores}
        total_weight = sum(active.values())
        if total_weight == 0:
            return float("nan") if self.errors else 0.0
        composite = sum(
            self.dimension_scores[d] * active[d] / total_weight
            for d in self.dimension_scores
        )
        return round(composite, 4)

    def save(self, output_dir: str, prefix: str = ""):
        """
        Save a JSON report (full results) and a CSV summary to output_dir.

        Parameters
        ----------
        output_dir : str
            Directory to write the files into (created if it doesn't exist).
        prefix : str
            Optional filename prefix, e.g. 'ganblr_car_'.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._save_json(output_dir, prefix)
        self._save_csv(output_dir, prefix)

    def _save_json(self, output_dir: str, prefix: str):
        active_weights = {d: self.weights.get(d, 0.0) for d in self.dimension_scores}
        total_weight = sum(active_weights.values())
        normalised_weights = {
            d: round(w / total_weight, 4) if total_weight else 0.0
            for d, w in active_weights.items()
        }
        payload = {
            "composite_score": self.composite_score,
            "dimension_scores": self.dimension_scores,
            "successful_dimensions": self.successful_dimensions,
            "successful_dimension_count": self.successful_dimension_count,
            "total_dimensions": self.total_dimensions,
            "failed_dimensions": self.failed_dimensions,
            "skipped_dimensions": self.skipped_dimensions,
            "weights_used": normalised_weights,
            "errors": self.errors,
            "full_results": self._serialisable(self.dimension_results),
        }
        path = os.path.join(output_dir, f"{prefix}evaluation_report.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Full report saved to:    {path}")

    def _save_csv(self, output_dir: str, prefix: str):
        path = os.path.join(output_dir, f"{prefix}evaluation_summary.csv")
        active_weights = {d: self.weights.get(d, 0.0) for d in self.dimension_scores}
        total_weight = sum(active_weights.values())

        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dimension", "Score", "Weight", "Weighted Score"])
            for dim, score in self.dimension_scores.items():
                w = active_weights[dim]
                normalised_w = round(w / total_weight, 4) if total_weight else 0
                writer.writerow(
                    [dim, score, normalised_w, round(score * normalised_w, 4)]
                )
            for dim in self.errors:
                writer.writerow([dim, "failed", "", ""])

            writer.writerow([])
            writer.writerow(["composite_score", self.composite_score, "", ""])

        print(f"Summary CSV saved to:    {path}")

    def print_summary(self):
        bar_width = 30
        print("\n" + "=" * 52)
        print("  SYNTHETIC DATA EVALUATION REPORT")
        print("=" * 52)

        active_weights = {d: self.weights.get(d, 0.0) for d in self.dimension_scores}
        total_weight = sum(active_weights.values())

        for dim, score in self.dimension_scores.items():
            w = active_weights[dim]
            norm_w = w / total_weight if total_weight else 0
            filled = int(score * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"  {dim:<12} [{bar}] {score:.4f}  (weight {norm_w:.0%})")

        for dim in self.errors:
            print(f"  {dim:<12} FAILED, excluded from the composite")

        print("-" * 52)
        print(
            f"  Dimensions successful: "
            f"{self.successful_dimension_count}/{self.total_dimensions}"
        )
        print(f"  Successful: {', '.join(self.successful_dimensions) or 'None'}")
        print(f"  Failed: {', '.join(self.failed_dimensions) or 'None'}")
        print(f"  Skipped: {', '.join(self.skipped_dimensions) or 'None'}")

        print("-" * 52)
        filled = (
            0
            if math.isnan(self.composite_score)
            else int(self.composite_score * bar_width)
        )
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"  {'COMPOSITE':<12} [{bar}] {self.composite_score:.4f}")
        print("=" * 52)

    def _serialisable(self, obj):
        """Recursively convert numpy types to plain Python for JSON."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialisable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


def warn_if_dimension_sets_differ(reports: dict):
    """
    Warn when models have different successful evaluation dimensions.

    Parameters
    ----------
    reports : dict[str, EvaluationReport]
        Mapping of model names to their evaluation reports.

    Returns
    -------
    bool
        True if dimension sets differ, otherwise False.
    """
    if len(reports) < 2:
        return False

    dimension_sets = {
        model: set(report.successful_dimensions) for model, report in reports.items()
    }

    reference_model = next(iter(dimension_sets))
    reference_dimensions = dimension_sets[reference_model]

    differences_found = False

    for model, dimensions in dimension_sets.items():
        if dimensions != reference_dimensions:
            differences_found = True
            missing = sorted(reference_dimensions - dimensions)
            additional = sorted(dimensions - reference_dimensions)

            print(
                f"WARNING: {model} was evaluated using a different "
                f"dimension set than {reference_model}."
            )
            print(f"  Missing dimensions: {missing or 'None'}")
            print(f"  Additional dimensions: {additional or 'None'}")

    return differences_found
