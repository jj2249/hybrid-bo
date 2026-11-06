"""The loop a script writes, one point at a time. `measure` is the script's function of
(u, rng), returning the measured variables as an array."""

from collections.abc import Callable

import numpy as np

import hybrid_bo as hb
from hybrid_bo.gp import GP


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
    z: np.ndarray,
    w: np.ndarray,
    rng: np.random.Generator,
    *,
    n_iterations: int,
    gp: GP,
    acquisition: Callable[[float], hb.EI | hb.SoftEI],
    unknown_states_as_variables: bool = True,
) -> tuple[hb.Dataset, list[tuple]]:
    """Hybrid BO from `dataset`, with the scenarios z and w drawn once, before the loop.

    Returns the grown dataset and each iteration's solution.
    """
    solutions = []
    for _ in range(n_iterations):
        fitted = hb.fit_surrogate(problem.system, dataset, gp=gp, rng=rng)
        incumbent = hb.incumbents(dataset)[-1]
        hybrid_nlp = hb.HybridNLP(
            problem,
            fitted,
            z,
            w,
            objective=acquisition(incumbent),
            unknown_states_as_variables=unknown_states_as_variables,
        )
        solution = hybrid_nlp.solve(rng)
        dataset = hb.extend(dataset, evaluate_point(problem, measure, solution[0], rng))
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
) -> tuple[hb.Dataset, list[tuple]]:
    """Standard BO from `dataset`: a GP of the measured objective over u.

    Returns the grown dataset and each iteration's solution.
    """
    solutions = []
    for _ in range(n_iterations):
        fitted = gp.fit(dataset.u, dataset.f, rng)
        incumbent = hb.incumbents(dataset)[-1]
        solution = hb.standard_acquisition(problem, fitted, incumbent).solve(rng)
        dataset = hb.extend(dataset, evaluate_point(problem, measure, solution[0], rng))
        solutions.append(solution)
    return dataset, solutions
