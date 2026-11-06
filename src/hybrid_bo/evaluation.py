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
from hybrid_bo.problem import Problem, Symbolic, System, start_bounds
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import _find_root

# Runs the system at u (N,) and returns the measured variables (P,) in the order of
# `system.measured`
Measure = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _reconstruct(
    system: System,
    u: np.ndarray,
    measured: np.ndarray,
    starts: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Solves the root-finding problem: `g` for the unmeasured states, given u and the
    measured variables, from each row of `starts` in turn.

    Returns every state, shape (M + P,). Raises `RootFindingError` if no start gives a
    root.
    """
    # Every state as an expression: the measured ones are numbers, the rest symbols
    w = Symbolic.sym("w", len(system.unmeasured))
    values = dict(zip(system.measured, measured, strict=True))
    symbols = dict(zip(system.unmeasured, cas.vertsplit(w), strict=True))
    column = cas.vertcat(*((values | symbols)[var] for var in system.states))
    residuals = system.g(column[: system.n_x], column[system.n_x :], u)

    # Solve, then put the states back together in state order
    root = _find_root(w, residuals, system.unmeasured, starts, tol)
    values |= dict(zip(system.unmeasured, root, strict=True))
    return np.array([values[var] for var in system.states])


def _dataset(problem: Problem, u: np.ndarray, states: list[np.ndarray]) -> Dataset:
    """The Dataset of the points u (n, N) with their states, all initial points."""
    system = problem.system
    array = np.reshape(states, (len(states), system.n_x + system.n_y))
    x, y = array[:, : system.n_x], array[:, system.n_x :]
    f = np.array([problem.f(*point).full().item() for point in zip(x, y, u)])
    return Dataset(u, x, y, f)


def _check_measurements(system: System, u: np.ndarray, measured: np.ndarray) -> None:
    """Raises `ValueError` unless u is (n, N) and measured (n, P)."""
    if u.ndim != 2 or u.shape[1] != system.n_u:
        msg = f"The decision variables u need shape (n, {system.n_u}), got {u.shape}"
        raise ValueError(msg)
    if measured.shape != (len(u), system.n_y):
        msg = (
            f"The measured variables need shape ({len(u)}, {system.n_y}), a row per "
            f"point of u and a column per variable, got {measured.shape}"
        )
        raise ValueError(msg)


def reconstruct_states(
    problem: Problem,
    *,
    u: np.ndarray,
    measured: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Dataset:
    """The Dataset of measurements the caller already has, as initial points: `evaluate`
    without the measurement.

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
    _check_measurements(system, u, measured)

    # The root-finding problem of each row, noting the rows that fail. A value that
    # isn't finite fails too: with no unmeasured states nothing would be solved
    states: list[np.ndarray] = []
    failed: list[int] = []
    for row, (u_point, measured_point) in enumerate(zip(u, measured, strict=True)):
        starts = lhs(*start_bounds(system.unmeasured), n_starts, rng)
        try:
            if not np.isfinite(np.concatenate([u_point, measured_point])).all():
                msg = f"Row {row} holds a value that isn't finite"
                raise RootFindingError(msg)
            states.append(_reconstruct(system, u_point, measured_point, starts, tol))
        except RootFindingError:
            failed.append(row)

    # The rows that worked, as the result or on the error
    dataset = _dataset(problem, np.delete(u, failed, axis=0), states)
    if failed:
        msg = f"No root, or a value that isn't finite, at rows {failed}"
        error = RootFindingError(msg)
        error.dataset = dataset
        raise error
    return dataset
