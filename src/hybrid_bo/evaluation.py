"""Evaluation: running measurements at trial points and reconstructing the states.

A measurement maps the decision variables to the measured variables (ADR 0001). After
every measurement the library solves the root-finding problem, the known equations `g`
for the unmeasured states, and computes the objective. The measurement can be:

- **A real system**: the plant itself, or an in silico simulation that stands in for it
  (e.g. a flowsheet simulator). You write the measurement as a function that runs it and
  returns only the measured variables as an array. The library sees nothing else, so
  there are no oracle values and no check against them. This is all a real system
  supplies.
- **The oracle simulation** (`measure_oracle`, in `oracle.py`), for benchmark
  experiments, which returns the measured variables too: the oracle values stay in
  `oracle.py`, for scoring afterwards.

**Measurements the caller already has.** `reconstruct_states` is `evaluate` without
the measurement: it solves the root-finding problem for given measured variables.
"""

import warnings
from collections.abc import Callable, Sequence

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.errors import (
    HybridBOError,
    OutOfBoundsWarning,
    RootFindingError,
)
from hybrid_bo.problem import Problem, Symbolic, System, Var, _states
from hybrid_bo.sampling import _start_bounds, lhs
from hybrid_bo.solvers import _find_root

# Runs the system at u (N,) and returns the measured variables (P,) in the order of
# `system.measured`, whether a real system (the plant or an in silico simulation of it)
# or the oracle simulation.
Measure = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _unmeasured(system: System) -> list[Var]:
    """The states the root-finding problem solves for, in state order."""
    return [var for var in _states(system) if var not in system.measured]


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
    states, unmeasured = _states(system), _unmeasured(system)
    w = Symbolic.sym("w", len(unmeasured))
    values = dict(zip(system.measured, measured, strict=True))
    symbols = dict(zip(unmeasured, cas.vertsplit(w), strict=True))
    column = cas.vertcat(*((values | symbols)[var] for var in states))
    residuals = system.g(column[: system.n_x], column[system.n_x :], u)

    # Solve, then put the states back together in state order
    root = _find_root(w, residuals, unmeasured, starts, tol)
    values |= dict(zip(unmeasured, root, strict=True))
    return np.array([values[var] for var in states])


def _evaluate_point(
    system: System,
    measure: Measure,
    u: np.ndarray,
    rng: np.random.Generator,
    n_starts: int,
    tol: float,
) -> np.ndarray:
    """Runs the measurement at u (N,) and reconstructs the states (M + P,).

    Raises `ValueError` if the measurement returns the wrong shape, and
    `RootFindingError` if a measured variable isn't finite or no start gives a root.
    """
    # The measurement
    measured = np.asarray(measure(u, rng), dtype=float)
    if measured.shape != (system.n_y,):
        msg = (
            f"The measurement must return the {system.n_y} measured variables with "
            f"shape ({system.n_y},), got shape {measured.shape}"
        )
        raise ValueError(msg)

    # A value that isn't finite fails here: with no unmeasured states nothing is solved,
    # and a measured variable outside g would give a root all the same
    if not np.isfinite(measured).all():
        names = ", ".join(var.name for var in system.measured)
        msg = f"At u = {u}, ({names}) = {measured} holds a value that isn't finite"
        raise RootFindingError(msg)

    # The root-finding problem, from starts over the unmeasured variables
    starts = lhs(*_start_bounds(_unmeasured(system)), n_starts, rng)
    return _reconstruct(system, u, measured, starts, tol)


def _dataset(problem: Problem, u: np.ndarray, states: list[np.ndarray]) -> Dataset:
    """The Dataset of the points u (n, N) with their states, all initial points."""
    system = problem.system
    array = np.reshape(states, (len(states), system.n_x + system.n_y))
    x, y = array[:, : system.n_x], array[:, system.n_x :]
    f = np.array([problem.f(*point).full().item() for point in zip(x, y, u)])
    return Dataset(u, x, y, f, n_initial=len(u))


def evaluate(
    problem: Problem,
    measure: Measure,
    u: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Dataset:
    """Evaluates each row of u (n, N) in turn and returns them as initial points.

    Each evaluation runs the measurement, then solves the root-finding problem for the
    unmeasured states from `n_starts` Latin hypercube starts over their start bounds,
    then computes `f`.

    Raises `ValueError` if the measurement returns the wrong shape, and
    `RootFindingError` (from the oracle simulation, a measured variable that isn't
    finite, or the root-finding problem) with the points evaluated before it as
    `error.dataset`.
    """
    states: list[np.ndarray] = []
    for u_point in u:
        try:
            point = _evaluate_point(
                problem.system, measure, u_point, rng, n_starts, tol
            )
        except HybridBOError as error:
            error.dataset = _dataset(problem, u[: len(states)], states)
            raise
        states.append(point)
    return _dataset(problem, u, states)


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


def _outside_bounds(variables: Sequence[Var], values: np.ndarray) -> list[str]:
    """A description of each variable with values (n, len(variables)) outside its
    bounds, naming the rows."""
    outside = []
    for var, column in zip(variables, values.T, strict=True):
        rows = np.flatnonzero((column < var.lower) | (column > var.upper))
        if len(rows) > 0:
            outside.append(f"{var.name} at rows {rows.tolist()}")
    return outside


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
    bounds, then computes `f`. The dataset has no oracle values.

    A `u` or a measured variable outside its bounds is still a measurement, so it is kept,
    with one `OutOfBoundsWarning`. An unmeasured state must lie within its bounds, which
    may be what picks the right root.

    Raises `ValueError` if a shape is wrong. Every row is tried: `RootFindingError` names
    all that gave no root or hold a value that isn't finite, with the others as
    `error.dataset`, so a caller can carry on without them.
    """
    # The checks
    system = problem.system
    u = np.asarray(u, dtype=float)
    measured = np.asarray(measured, dtype=float)
    _check_measurements(system, u, measured)
    outside = _outside_bounds(system.u, u) + _outside_bounds(system.measured, measured)
    if outside:
        msg = f"Measurements outside their bounds: {', '.join(outside)}"
        warnings.warn(msg, OutOfBoundsWarning, stacklevel=2)

    # The root-finding problem of each row, noting the rows that fail. A value that
    # isn't finite fails too: with no unmeasured states nothing would be solved
    unmeasured = _unmeasured(system)
    states: list[np.ndarray] = []
    failed: list[int] = []
    for row, (u_point, measured_point) in enumerate(zip(u, measured, strict=True)):
        starts = lhs(*_start_bounds(unmeasured), n_starts, rng)
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
