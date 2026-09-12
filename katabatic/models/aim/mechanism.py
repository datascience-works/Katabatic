"""
Core Adaptive Iterative Mechanism (AIM) implementation for Katabatic.

This module adapts the AIM implementation from Ryan McKenna's private-pgm
research code and the concentrated-DP conversion routines from IBM's
discrete-gaussian-differential-privacy project.

The upstream implementations are licensed under the Apache License 2.0.

Adaptations for Katabatic include:
- replacing hdmm.matrix.Identity with scipy sparse identity matrices;
- removing dependencies that AIM does not require at runtime;
- deterministic random-state support;
- validation and integration-oriented error handling.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np
from mbi import FactoredInference, GraphicalModel
from scipy import sparse
from scipy.special import softmax


def powerset(iterable: Iterable[str]):
    """Return every non-empty subset of an iterable."""

    values = list(iterable)

    return itertools.chain.from_iterable(
        itertools.combinations(values, size) for size in range(1, len(values) + 1)
    )


def downward_closure(workload_cliques):
    """Return the downward closure of a collection of cliques."""

    closure = set()

    for projection in workload_cliques:
        closure.update(powerset(projection))

    return list(sorted(closure, key=len))


def hypothetical_model_size(domain, cliques) -> float:
    """Estimate graphical-model size in megabytes."""

    model = GraphicalModel(domain, cliques)

    return model.size * 8 / (2**20)


def compile_workload(workload):
    """Compile weighted workload cliques into AIM candidate scores."""

    weights = {tuple(clique): weight for clique, weight in workload}

    workload_cliques = weights.keys()

    def score(clique):
        return sum(
            weights[workload_clique] * len(set(clique) & set(workload_clique))
            for workload_clique in workload_cliques
        )

    return {clique: score(clique) for clique in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    """Remove candidates that would exceed the current model-size budget."""

    filtered = {}

    free_cliques = downward_closure(model.cliques)

    for clique in candidates:
        within_size_limit = (
            hypothetical_model_size(
                model.domain,
                model.cliques + [clique],
            )
            <= size_limit
        )

        already_represented = clique in free_cliques

        if within_size_limit or already_represented:
            filtered[clique] = candidates[clique]

    return filtered


def cdp_delta(rho: float, epsilon: float) -> float:
    """
    Compute delta such that rho-CDP implies (epsilon, delta)-DP.

    Adapted from IBM's discrete-gaussian-differential-privacy project.
    """

    if rho < 0:
        raise ValueError("rho must be non-negative.")

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    if rho == 0:
        return 0.0

    alpha_min = 1.01
    alpha_max = (epsilon + 1) / (2 * rho) + 2

    alpha = alpha_min

    for _ in range(1000):
        alpha = (alpha_min + alpha_max) / 2

        derivative = (2 * alpha - 1) * rho - epsilon + math.log1p(-1.0 / alpha)

        if derivative < 0:
            alpha_min = alpha
        else:
            alpha_max = alpha

    delta = math.exp(
        (alpha - 1) * (alpha * rho - epsilon) + alpha * math.log1p(-1 / alpha)
    ) / (alpha - 1.0)

    return min(delta, 1.0)


def cdp_rho(epsilon: float, delta: float) -> float:
    """
    Convert an (epsilon, delta)-DP guarantee into its rho-CDP budget.

    Adapted from IBM's discrete-gaussian-differential-privacy project.
    """

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    if delta <= 0:
        raise ValueError("delta must be greater than zero.")

    if delta >= 1:
        return 0.0

    rho_min = 0.0
    rho_max = epsilon + 1

    for _ in range(1000):
        rho = (rho_min + rho_max) / 2

        if cdp_delta(rho, epsilon) <= delta:
            rho_min = rho
        else:
            rho_max = rho

    return rho_min


class AIMMechanism:
    """Adaptive Iterative Mechanism for differentially private synthesis."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        *,
        seed: int | None = None,
        rounds: int | None = None,
        max_model_size: float = 80,
        max_iters: int = 1000,
        structural_zeros: dict | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1.")

        if rounds is not None and rounds <= 0:
            raise ValueError("rounds must be positive when provided.")

        if max_model_size <= 0:
            raise ValueError("max_model_size must be greater than zero.")

        if max_iters <= 0:
            raise ValueError("max_iters must be greater than zero.")

        self.epsilon = epsilon
        self.delta = delta
        self.rho = cdp_rho(epsilon, delta)

        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros or {}

        self.prng = np.random.RandomState(seed)

    def gaussian_noise(
        self,
        sigma: float,
        size: int,
    ) -> np.ndarray:
        """Generate independent Gaussian noise."""

        return self.prng.normal(
            loc=0.0,
            scale=sigma,
            size=size,
        )

    def exponential_mechanism(
        self,
        qualities,
        epsilon: float,
        sensitivity: float = 1.0,
    ):
        """Select a candidate using the exponential mechanism."""

        if sensitivity <= 0:
            raise ValueError("Exponential-mechanism sensitivity must be positive.")

        if isinstance(qualities, dict):
            keys = list(qualities.keys())

            if not keys:
                raise ValueError("The exponential mechanism received no candidates.")

            quality_values = np.array(
                [qualities[key] for key in keys],
                dtype=float,
            )
        else:
            quality_values = np.asarray(
                qualities,
                dtype=float,
            )

            if quality_values.size == 0:
                raise ValueError("The exponential mechanism received no candidates.")

            keys = list(range(quality_values.size))

        centred = quality_values - quality_values.max()

        probabilities = softmax(0.5 * epsilon / sensitivity * centred)

        selected = self.prng.choice(
            probabilities.size,
            p=probabilities,
        )

        return keys[selected]

    def worst_approximated(
        self,
        candidates,
        answers,
        model,
        epsilon: float,
        sigma: float,
    ):
        """Privately select the most poorly approximated marginal."""

        if not candidates:
            raise ValueError("AIM has no candidate marginals available for selection.")

        errors = {}
        sensitivities = {}

        for clique in candidates:
            weight = candidates[clique]
            actual = answers[clique]

            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(clique)

            estimated = model.project(clique).datavector()

            errors[clique] = weight * (
                np.linalg.norm(
                    actual - estimated,
                    1,
                )
                - bias
            )

            sensitivities[clique] = abs(weight)

        max_sensitivity = max(sensitivities.values())

        if max_sensitivity <= 0:
            raise ValueError(
                "AIM workload weights must contain at least one non-zero value."
            )

        return self.exponential_mechanism(
            errors,
            epsilon,
            max_sensitivity,
        )

    def run(
        self,
        data,
        workload,
        *,
        num_synth_rows: int | None = None,
        initial_cliques=None,
    ):
        """
        Fit AIM and generate a synthetic Private-PGM dataset.

        Parameters
        ----------
        data:
            mbi.Dataset containing discretised training data.
        workload:
            Sequence of ``(clique, weight)`` workload entries.
        num_synth_rows:
            Number of synthetic records to generate.
        initial_cliques:
            Optional initial marginal cliques.

        Returns
        -------
        tuple
            ``(model, synthetic_dataset)`` where model is the fitted
            Private-PGM graphical model.
        """

        if not workload:
            raise ValueError("AIM requires at least one workload marginal.")

        if num_synth_rows is not None and num_synth_rows <= 0:
            raise ValueError("num_synth_rows must be positive when provided.")

        rounds = self.rounds or 16 * len(data.domain)

        candidates = compile_workload(workload)

        if not candidates:
            raise ValueError("AIM workload produced no candidate marginals.")

        answers = {clique: data.project(clique).datavector() for clique in candidates}

        if not initial_cliques:
            initial_cliques = [clique for clique in candidates if len(clique) == 1]

        if not initial_cliques:
            raise ValueError("AIM requires at least one one-way marginal.")

        one_way = [clique for clique in candidates if len(clique) == 1]

        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))

        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        measurements = []

        rho_used = len(one_way) * 0.5 / sigma**2

        for clique in initial_cliques:
            actual = data.project(clique).datavector()

            noisy = actual + self.gaussian_noise(
                sigma,
                actual.size,
            )

            identity = sparse.identity(
                noisy.size,
                format="csr",
            )

            measurements.append(
                (
                    identity,
                    noisy,
                    sigma,
                    clique,
                )
            )

        engine = FactoredInference(
            data.domain,
            iters=self.max_iters,
            warm_start=True,
            structural_zeros=self.structural_zeros,
        )

        model = engine.estimate(measurements)

        terminate = False

        while not terminate:
            per_round_budget = 2 * (0.5 / sigma**2 + (1.0 / 8) * epsilon**2)

            if self.rho - rho_used < per_round_budget:
                remaining = self.rho - rho_used

                if remaining <= 0:
                    break

                sigma = np.sqrt(1 / (2 * 0.9 * remaining))

                epsilon = np.sqrt(8 * 0.1 * remaining)

                terminate = True

            rho_used += (1.0 / 8) * epsilon**2 + 0.5 / sigma**2

            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(
                candidates,
                model,
                size_limit,
            )

            if not small_candidates:
                break

            clique = self.worst_approximated(
                small_candidates,
                answers,
                model,
                epsilon,
                sigma,
            )

            marginal_size = data.domain.size(clique)

            identity = sparse.identity(
                marginal_size,
                format="csr",
            )

            actual = data.project(clique).datavector()

            noisy = actual + self.gaussian_noise(
                sigma,
                marginal_size,
            )

            measurements.append(
                (
                    identity,
                    noisy,
                    sigma,
                    clique,
                )
            )

            previous = model.project(clique).datavector()

            model = engine.estimate(measurements)

            updated = model.project(clique).datavector()

            threshold = sigma * np.sqrt(2 / np.pi) * marginal_size

            if (
                np.linalg.norm(
                    updated - previous,
                    1,
                )
                <= threshold
            ):
                sigma /= 2
                epsilon *= 2

        engine.iters = self.max_iters

        model = engine.estimate(measurements)

        synthetic = model.synthetic_data(rows=num_synth_rows)

        return model, synthetic
