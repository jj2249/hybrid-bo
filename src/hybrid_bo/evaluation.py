"""Evaluation: running experiments at trial points and reconstructing the states.

An experiment maps the decision variables to the measured variables (ADR 0001); a real
application supplies only this. After every experiment the library solves the
root-finding problem, the known equations `g` for the unmeasured states, and computes
the objective. In test cases the oracle simulation is the experiment, and it also gives
the oracle values.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.errors import HybridBOError, ReconstructionWarning, RootFindingError
from hybrid_bo.history import History
from hybrid_bo.problem import Problem, Symbolic, System, Var, _trace
from hybrid_bo.sampling import lhs


@dataclass(frozen=True)
class OracleMeasurement:
    """What the oracle simulation returns for one trial point: the measured variables
    and the oracle values of the states."""

    measured: np.ndarray  # (P,)
    x: np.ndarray  # (M,)
    y: np.ndarray  # (P,)


# Runs the system at u (N,) and returns the measured variables (P,), or an
# OracleMeasurement in test cases
Experiment = Callable[[np.ndarray, np.random.Generator], np.ndarray | OracleMeasurement]


def _states(system: System) -> tuple[Var, ...]:
    """The states in the order used for arrays of states: x, then y."""
    return (*system.x, *system.y)


def _start_bounds(variables: Sequence[Var]) -> tuple[np.ndarray, np.ndarray]:
    """The lower and upper start bounds of the variables, each of shape (d,)."""
    return (
        np.array([var.start[0] for var in variables]),
        np.array([var.start[1] for var in variables]),
    )


def _find_root(
    w: Any, residuals: Any, variables: Sequence[Var], starts: np.ndarray, tol: float
) -> np.ndarray:
    """A root of `residuals`, a column expression in the column of symbols `w`.

    Runs fsolve with the CasADi Jacobian from each row of `starts` in turn and returns
    the first root, shape (d,), whose residual norm is below `tol` and that lies within
    the bounds of `variables` (one per entry of `w`). A start that fails is expected,
    so fsolve's convergence warnings are silenced.

    Raises `RootFindingError` if no start gives such a root.
    """
    residual = cas.Function("residual", [w], [residuals])
    jacobian = cas.Function("jacobian", [w], [cas.jacobian(residuals, w)])
    lower = np.array([var.lower for var in variables])
    upper = np.array([var.upper for var in variables])

    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            root = scipy.optimize.fsolve(
                lambda w: residual(w[:, np.newaxis]).full().ravel(),
                start,
                fprime=lambda w: jacobian(w[:, np.newaxis]).full(),
            )
        is_root = np.linalg.norm(residual(root[:, np.newaxis]).full()) < tol
        if is_root and np.all(lower <= root) and np.all(root <= upper):
            return root

    names = ", ".join(var.name for var in variables)
    msg = f"No root for {names} within the bounds from {len(starts)} starts"
    raise RootFindingError(msg)


def oracle_experiment(
    system: System,
    oracle: Callable[[Any], Any],
    noise_std: float = 0.0,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Experiment:
    """The oracle simulation as an experiment, for test cases.

    `oracle` is a callable over the variables like `g` (`lambda v: [cas.sin(v.u)]`)
    and gives the unknown states, y = oracle(x, u). The experiment solves `g` and the
    oracle together for every state, from `n_starts` Latin hypercube starts over the
    start bounds, reads the measured variables and adds normal noise with std
    `noise_std`. The noise is drawn even when `noise_std` is 0, so the draws don't
    depend on it.

    Raises `ValueError` if the oracle doesn't give one value per unknown state. The
    experiment raises `RootFindingError` if no start gives a root.
    """
    oracle_function = _trace("oracle", oracle, system.x, system.y, system.u)
    if oracle_function.size_out(0) != (system.n_y, 1):
        msg = (
            f"The oracle must give one value per unknown state ({system.n_y}) as a "
            f"column, got shape {oracle_function.size_out(0)}"
        )
        raise ValueError(msg)
    states = _states(system)
    is_measured = np.array([var in system.measured for var in states])

    def experiment(u: np.ndarray, rng: np.random.Generator) -> OracleMeasurement:
        # Oracle values: g and the oracle together, for every state
        w = Symbolic.sym("w", len(states))
        x, y = w[: system.n_x], w[system.n_x :]
        residuals = cas.vertcat(system.g(x, y, u), oracle_function(x, y, u) - y)
        starts = lhs(*_start_bounds(states), n_starts, rng)
        root = _find_root(w, residuals, states, starts, tol)

        # The sensors read the measured variables, with noise
        noise = rng.normal(0.0, noise_std, size=system.n_y)
        return OracleMeasurement(
            measured=root[is_measured] + noise,
            x=root[: system.n_x],
            y=root[system.n_x :],
        )

    return experiment


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
    states = _states(system)
    unmeasured = [var for var in states if var not in system.measured]

    # Every state as an expression: the measured ones are numbers, the rest symbols
    w = Symbolic.sym("w", len(unmeasured))
    values = dict(zip(system.measured, measured, strict=True))
    symbols = dict(zip(unmeasured, cas.vertsplit(w), strict=True))
    column = cas.vertcat(*((values | symbols)[var] for var in states))
    residuals = system.g(column[: system.n_x], column[system.n_x :], u)

    root = _find_root(w, residuals, unmeasured, starts, tol)
    values |= dict(zip(unmeasured, root, strict=True))
    return np.array([values[var] for var in states])


def _evaluate_point(
    system: System,
    experiment: Experiment,
    u: np.ndarray,
    rng: np.random.Generator,
    n_starts: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Runs the experiment at u (N,) and reconstructs the states.

    Returns the states (M + P,) and the oracle states (None without an oracle). Warns
    with `ReconstructionWarning` when the reconstruction is a different root from the
    one the oracle values lie on.
    """
    # The experiment: the measured variables, and the oracle values in test cases
    result = experiment(u, rng)
    oracle = result if isinstance(result, OracleMeasurement) else None
    measured = np.asarray(result if oracle is None else oracle.measured, dtype=float)
    if measured.shape != (system.n_y,):
        msg = (
            f"The experiment must return the {system.n_y} measured variables with "
            f"shape ({system.n_y},), got shape {measured.shape}"
        )
        raise ValueError(msg)

    # The root-finding problem, from starts over the unmeasured variables
    states = _states(system)
    unmeasured = [var for var in states if var not in system.measured]
    starts = lhs(*_start_bounds(unmeasured), n_starts, rng)
    reconstruction = _reconstruct(system, u, measured, starts, tol)
    if oracle is None:
        return reconstruction, None

    # Agreement: solving the same root-finding problem from the oracle values must
    # give the same root (noise moves both along the branch the oracle lies on)
    oracle_states = np.concatenate([oracle.x, oracle.y])
    is_unmeasured = np.array([var in unmeasured for var in states])
    try:
        nearest = _reconstruct(
            system, u, measured, oracle_states[is_unmeasured][np.newaxis], tol
        )
        agrees = np.allclose(nearest, reconstruction, rtol=1e-6, atol=1e-6)
    except RootFindingError:
        agrees = False
    if not agrees:
        names = ", ".join(var.name for var in states)
        msg = (
            f"At u = {u}, the root-finding problem gave ({names}) = {reconstruction}, "
            f"which is not the root the oracle values {oracle_states} lie on; the "
            f"measured variables don't determine the states uniquely"
        )
        warnings.warn(msg, ReconstructionWarning, stacklevel=3)
    return reconstruction, oracle_states


def _history(
    problem: Problem,
    u: np.ndarray,
    states: list[np.ndarray],
    oracle_states: list[np.ndarray | None],
) -> History:
    """The History of the points u (n, N) with their states, all initial points.

    The oracle values are kept only when every point has them.
    """
    system = problem.system

    def split(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """x, y and f of rows of states."""
        array = np.reshape(rows, (len(rows), system.n_x + system.n_y))
        x, y = array[:, : system.n_x], array[:, system.n_x :]
        f = np.array([problem.f(*point).full().item() for point in zip(x, y, u)])
        return x, y, f

    x, y, f = split(states)
    oracle_rows = [row for row in oracle_states if row is not None]
    has_oracle = len(oracle_rows) == len(oracle_states)
    x_oracle, y_oracle, f_oracle = split(oracle_rows) if has_oracle else (None,) * 3
    return History(u, x, y, f, x_oracle, y_oracle, f_oracle, n_initial=len(u))


def evaluate(
    problem: Problem,
    experiment: Experiment,
    u: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> History:
    """Evaluates each row of u (n, N) in turn and returns them as initial points.

    Each evaluation runs the experiment, then solves the root-finding problem for the
    unmeasured states from `n_starts` Latin hypercube starts over their start bounds,
    then computes `f`. With an oracle, it also computes `f` at the oracle values and
    warns with `ReconstructionWarning` when the reconstruction is a different root.

    Raises `ValueError` if the experiment returns the wrong shape, and
    `RootFindingError` (from the oracle simulation or the root-finding problem) with
    the points evaluated before it as `error.history`.
    """
    states: list[np.ndarray] = []
    oracle_states: list[np.ndarray | None] = []
    for u_point in u:
        try:
            point, oracle_point = _evaluate_point(
                problem.system, experiment, u_point, rng, n_starts, tol
            )
        except HybridBOError as error:
            error.history = _history(problem, u[: len(states)], states, oracle_states)
            raise
        states.append(point)
        oracle_states.append(oracle_point)
    return _history(problem, u, states, oracle_states)
