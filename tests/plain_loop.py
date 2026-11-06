"""The loop a script writes, one point at a time. `measure` is the script's function of
(u, rng), returning the measured variables as an array."""

import numpy as np

import hybrid_bo as hb
from hybrid_bo.acquisition import AcquisitionSolution
from hybrid_bo.gp import GP
from hybrid_bo.hybrid_model import Scenarios


def evaluate_point(
    problem: hb.Problem, measure: hb.Measure, u: np.ndarray, rng: np.random.Generator
) -> hb.Dataset:
    """Measures the system at u (N,) and solves for its states: a one-row dataset."""
    measured = measure(u, rng)
    return hb.reconstruct_states(
        problem, u=u[np.newaxis], measured=measured[np.newaxis], rng=rng
    )


def evaluate_points(
    problem: hb.Problem,
    measure: hb.Measure,
    points: np.ndarray,
    rng: np.random.Generator,
) -> hb.Dataset:
    """The dataset of the rows of `points` (n, N), evaluated one after the other."""
    dataset = evaluate_point(problem, measure, points[0], rng)
    for u in points[1:]:
        dataset = hb.extend(dataset, evaluate_point(problem, measure, u, rng))
    return dataset


def hybrid_loop(
    problem: hb.Problem,
    measure: hb.Measure,
    dataset: hb.Dataset,
    scenarios: Scenarios,
    rng: np.random.Generator,
    *,
    n_iterations: int,
    gp: GP,
    approximation: hb.SAA | hb.SoftSAA,
) -> tuple[hb.Dataset, list[AcquisitionSolution]]:
    """Hybrid BO from `dataset`, with the scenarios drawn once, before the loop.

    Returns the grown dataset and each iteration's solution.
    """
    solutions = []
    for _ in range(n_iterations):
        fitted = hb.fit_surrogate(problem.system, dataset, gp=gp, rng=rng)
        solution = hb.hybrid_trial_point(
            problem,
            fitted,
            scenarios,
            hb.incumbents(dataset)[-1],
            rng,
            approximation=approximation,
        )
        dataset = hb.extend(dataset, evaluate_point(problem, measure, solution.u, rng))
        solutions.append(solution)
    return dataset, solutions


def standard_loop(
    problem: hb.Problem,
    measure: hb.Measure,
    dataset: hb.Dataset,
    rng: np.random.Generator,
    *,
    n_iterations: int,
    gp: GP,
) -> tuple[hb.Dataset, list[AcquisitionSolution]]:
    """Standard BO from `dataset`: a GP of the measured objective over u.

    Returns the grown dataset and each iteration's solution.
    """
    solutions = []
    for _ in range(n_iterations):
        fitted = hb.fit_gp(gp, dataset.u, dataset.f, rng)
        solution = hb.standard_trial_point(
            problem, fitted, hb.incumbents(dataset)[-1], rng
        )
        dataset = hb.extend(dataset, evaluate_point(problem, measure, solution.u, rng))
        solutions.append(solution)
    return dataset, solutions
