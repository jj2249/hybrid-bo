"""The loops a script writes, one point at a time."""

from collections.abc import Callable

import illustrative_example as ex
import numpy as np

import hybrid_bo as hb


def measure_points(points: np.ndarray, rng: np.random.Generator) -> hb.Dataset:
    """The rows of `points` (n, N), measured one after the other."""
    data = ex.measure(points[0], rng)
    for u in points[1:]:
        data = data.extend(ex.measure(u, rng))
    return data


def hybrid_loop(
    data: hb.Dataset,
    method: hb.SAA,
    rng: np.random.Generator,
    *,
    n_iterations: int,
    objective: Callable[[float], hb.obj.EI | hb.obj.SoftEI],
) -> tuple[hb.Dataset, list[hb.Solution]]:
    """Hybrid BO from `data`: the grown dataset and each iteration's solution."""
    solutions = []
    for _ in range(n_iterations):
        surrogate = [ex.gp.fit(data, inputs=[ex.u], output=ex.y, rng=rng)]
        solution = ex.nlp.solve(
            method,
            rng,
            surrogate=surrogate,
            objective=objective(data.incumbents[-1]),
            unknown_states_as_variables=False,
        )
        data = data.extend(ex.measure(solution.u, rng))
        solutions.append(solution)
    return data, solutions


def standard_loop(
    data: hb.Dataset, rng: np.random.Generator, *, n_iterations: int
) -> tuple[hb.Dataset, list[hb.Solution]]:
    """Standard BO from `data`, a GP of the measured f over u: the grown dataset of the
    system and each iteration's solution."""
    solutions = []
    for _ in range(n_iterations):
        observed = hb.Dataset.from_measurements(
            ex.standard_nlp, u=data.u, y=data.f[:, np.newaxis], rng=rng
        )
        surrogate = [ex.gp.fit(observed, inputs=[ex.u], output=ex.fhat, rng=rng)]
        solution = ex.standard_nlp.solve(
            hb.Exact(),
            rng,
            surrogate=surrogate,
            objective=hb.obj.EI(data.incumbents[-1]),
        )
        data = data.extend(ex.measure(solution.u, rng))
        solutions.append(solution)
    return data, solutions
