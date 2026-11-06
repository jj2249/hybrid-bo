"""Reconstruction: from measurements to the states and the objective.

A measurement maps the decision variables to the measured variables. You write it as a
function of (u, rng) that runs the system, whether a plant, an external simulator or an
oracle of your own, and returns only the measured variables as an array.
`reconstruct_states` then solves the root-finding problem, the known equations `g` for
the unmeasured states, and computes the objective.
"""

from collections.abc import Callable

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.errors import RootFindingError
from hybrid_bo.problem import Problem, Symbolic, System, lhs_points
from hybrid_bo.solvers import RootProblem, find_root

# Runs the system at u (N,) and returns the measured variables (P,) in the order of
# `system.measured`
Measure = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _root_finding_problem(system: System) -> RootProblem:
    """The root-finding problem: `g` for the unmeasured states, with u and the measured
    variables as its parameters, in that order."""
    # Every state as a symbol: the unmeasured ones are unknowns, the measured parameters
    w = Symbolic.sym("w", len(system.unmeasured))
    u = Symbolic.sym("u", system.n_u)
    measured = Symbolic.sym("measured", system.n_y)
    symbols = dict(zip(system.unmeasured, cas.vertsplit(w), strict=True))
    symbols |= dict(zip(system.measured, cas.vertsplit(measured), strict=True))
    states = cas.vertcat(*(symbols[var] for var in system.states))

    residual = system.g(states[: system.n_x], states[system.n_x :], u)
    return RootProblem(
        cas.Function("known_equations", [w, cas.vertcat(u, measured)], [residual]),
        np.array([var.lower for var in system.unmeasured]),
        np.array([var.upper for var in system.unmeasured]),
    )


def reconstruct_states(
    problem: Problem,
    *,
    u: np.ndarray,
    measured: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Dataset:
    """The Dataset of measurements: the states and objective at each row of u.

    `u` (n, N) holds the decision variables and `measured` (n, P) the measured variables
    at them, in the order of `system.measured`. Each row solves the root-finding problem
    for the unmeasured states from `n_starts` Latin hypercube starts over their start
    bounds, then computes `f`.

    An unmeasured state must lie within its bounds, which may be what picks the right
    root.

    Raises `ValueError` if a shape is wrong. Every row is tried: `RootFindingError` names
    all that gave no root or hold a value that isn't finite, with the others as
    `error.dataset`, so a caller can carry on without them.
    """
    # The checks
    system = problem.system
    u = np.asarray(u, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if u.ndim != 2 or u.shape[1] != system.n_u:
        msg = f"The decision variables u need shape (n, {system.n_u}), got {u.shape}"
        raise ValueError(msg)
    if measured.shape != (len(u), system.n_y):
        msg = (
            f"The measured variables need shape ({len(u)}, {system.n_y}), a row per "
            f"point of u and a column per variable, got {measured.shape}"
        )
        raise ValueError(msg)

    # The root-finding problem at each row, noting the rows that fail. A value that
    # isn't finite fails too: with no unmeasured states nothing would be solved
    root_finding_problem = _root_finding_problem(system)
    states: list[np.ndarray] = []
    failed: list[int] = []
    for row, (u_point, measured_point) in enumerate(zip(u, measured, strict=True)):
        starts = lhs_points(system.unmeasured, n_starts, rng)
        try:
            if not np.isfinite(np.concatenate([u_point, measured_point])).all():
                msg = f"Row {row} holds a value that isn't finite"
                raise RootFindingError(msg)
            p = np.concatenate([u_point, measured_point])
            root = find_root(root_finding_problem, p, starts, tol)
            values = dict(zip(system.unmeasured, root, strict=True))
            values |= dict(zip(system.measured, measured_point, strict=True))
            states.append(np.array([values[var] for var in system.states]))
        except RootFindingError:
            failed.append(row)

    # The rows that worked, as the result or on the error
    u = np.delete(u, failed, axis=0)
    array = np.reshape(states, (len(states), system.n_x + system.n_y))
    x, y = array[:, : system.n_x], array[:, system.n_x :]
    f = np.array([problem.f(*point).full().item() for point in zip(x, y, u)])
    dataset = Dataset(u, x, y, f)
    if failed:
        msg = f"No root, or a value that isn't finite, at rows {failed}"
        error = RootFindingError(msg)
        error.dataset = dataset
        raise error
    return dataset
