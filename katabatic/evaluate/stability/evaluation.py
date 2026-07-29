import numpy as np
import pandas as pd

from katabatic.evaluate.base_evaluation import Evaluation
from katabatic.evaluate.diversity.evaluation import DiversityEvaluation
from katabatic.evaluate.fidelity.evaluation import FidelityEvaluation

# A std of 0 gives score 1.0; at or above this threshold the score hits 0.0
_STD_CEILING = 0.10


class StabilityEvaluation(Evaluation):
    """
    Stability dimension: measures whether a trained model produces
    consistently reliable synthetic data across multiple sampling runs.

    The model is sampled N times using different random seeds. Fidelity and
    Diversity scores are computed for each run. The stability score is derived
    from the variance of those scores: low variance = stable model.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real feature dataset (target column excluded).
    model : katabatic.models.base_model.Model
        A trained model instance that implements sample(n_samples, seed=int).
    n_samples : int, optional
        Number of rows to generate per run. Defaults to len(real_data).
    n_runs : int
        Number of sampling runs (default: 5).
    seeds : list[int], optional
        Random seeds for each run. Defaults to [0, 1, 2, 3, 4].
    categorical_cols : list[str], optional
        Passed through to Fidelity and Diversity. Auto-detected if omitted.
    continuous_cols : list[str], optional
        Passed through to Fidelity and Diversity. Auto-detected if omitted.
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        model,
        n_samples: int = None,
        n_runs: int = 5,
        seeds: list = None,
        categorical_cols: list = None,
        continuous_cols: list = None,
    ):
        super().__init__(real_data=real_data, synthetic_data=None)
        self.model = model
        self.n_samples = n_samples or len(real_data)
        self.n_runs = n_runs
        self.seeds = seeds or list(range(n_runs))
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols

        if len(self.seeds) != self.n_runs:
            raise ValueError(
                f"len(seeds)={len(self.seeds)} must equal n_runs={self.n_runs}."
            )

    def evaluate(self) -> dict:
        per_run = {"fidelity": [], "diversity": []}
        run_details = []

        for i, seed in enumerate(self.seeds):
            print(f"  [Stability] run {i + 1}/{self.n_runs}  (seed={seed})")

            synth = self._sample(seed)
            if synth is None:
                print(
                    f"  [Stability] WARNING: sample() returned None for seed={seed}, skipping run."
                )
                continue

            fid_result = FidelityEvaluation(
                self.real_data,
                synth,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
            ).evaluate()

            div_result = DiversityEvaluation(
                self.real_data,
                synth,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
            ).evaluate()

            per_run["fidelity"].append(fid_result["fidelity_score"])
            per_run["diversity"].append(div_result["diversity_score"])
            run_details.append(
                {
                    "seed": seed,
                    "fidelity_score": fid_result["fidelity_score"],
                    "diversity_score": div_result["diversity_score"],
                }
            )

        if not run_details:
            return {"stability_score": 0.0, "error": "All sampling runs failed."}

        mean_scores = {
            dim: round(float(np.mean(vals)), 4) for dim, vals in per_run.items() if vals
        }
        std_scores = {
            dim: round(float(np.std(vals)), 4) for dim, vals in per_run.items() if vals
        }

        avg_std = float(np.mean(list(std_scores.values()))) if std_scores else 0.0
        stability_score = round(max(0.0, 1.0 - avg_std / _STD_CEILING), 4)

        results = {
            "n_runs": len(run_details),
            "seeds": [r["seed"] for r in run_details],
            "per_run_scores": run_details,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "avg_std": round(avg_std, 4),
            "stability_score": stability_score,
        }

        self._print_summary(results)
        return results

    def _sample(self, seed: int):
        """
        Call model.sample(n_samples, seed=seed) and return a DataFrame
        aligned to real_data columns.
        """
        try:
            result = self.model.sample(self.n_samples, seed=seed)
        except TypeError:
            # Model does not support seed argument — fall back to unseeded call.
            # All runs will be identical on deterministic models, producing std=0
            # and a misleadingly perfect stability score. A warning is printed once.
            if not getattr(self, "_unseeded_warned", False):
                print(
                    "  [Stability] WARNING: model.sample() does not accept a 'seed' argument. "
                    "All stability runs will use unseeded sampling. On deterministic models this "
                    "produces std=0 and a stability_score=1.0 that does not reflect true stability. "
                    "Implement sample(n, seed=int) in your model for meaningful stability results."
                )
                self._unseeded_warned = True
            try:
                result = self.model.sample(self.n_samples)
            except Exception as e:
                print(f"  [Stability] model.sample() raised: {e}")
                return None
        except Exception as e:
            print(f"  [Stability] model.sample() raised: {e}")
            return None

        if result is None:
            return None

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"model.sample() must return a pandas DataFrame (got {type(result).__name__}). "
                "Ensure the model fulfils the base Model contract."
            )

        synth = result.reset_index(drop=True)
        shared = [c for c in synth.columns if c in self.real_data.columns]
        return synth[shared]

    def _print_summary(self, results):
        print("\n=== Stability Evaluation ===")
        print(f"Runs completed: {results['n_runs']}   seeds: {results['seeds']}")
        print(f"\n{'Run':<6} {'Seed':<8} {'Fidelity':<12} {'Diversity':<12}")
        print("-" * 40)
        for i, r in enumerate(results["per_run_scores"]):
            print(
                f"{i + 1:<6} {r['seed']:<8} {r['fidelity_score']:<12.4f} {r['diversity_score']:<12.4f}"
            )
        print("-" * 40)
        print(
            f"{'Mean':<6} {'':<8} {results['mean_scores'].get('fidelity', '-'):<12} {results['mean_scores'].get('diversity', '-'):<12}"
        )
        print(
            f"{'Std':<6} {'':<8} {results['std_scores'].get('fidelity', '-'):<12} {results['std_scores'].get('diversity', '-'):<12}"
        )
        print(
            f"\nAverage std across dimensions: {results['avg_std']:.4f}  (target < 0.05)"
        )
        flag = (
            "  [WARNING] High variance — model is unstable."
            if results["avg_std"] >= 0.05
            else "  [OK]"
        )
        print(f"Stability score: {results['stability_score']:.4f}{flag}")
