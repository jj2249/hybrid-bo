"""The oracle: the unknown equations in closed form, for benchmark experiments, which
score a method.

You write the oracle in the same CasADi namespace as `g`, and the library simulates the
system from `g` and the oracle together (`measure_oracle`). The oracle simulation
returns the noisy measured variables, which are all the method sees. A `Dataset` holds
only those and what they give, so the oracle values come from here, afterwards:
`oracle_values` gives them at any `u`, `oracle_incumbents` scores a dataset with them,
and `check_reconstruction` checks the dataset's states against them.

The difference from a real system is not whether the system is simulated: an in silico
simulation of a plant is a real system, because the method doesn't know its unknown
equations. What makes an oracle is that the benchmark experiment states the unknown
equations exactly, so the oracle values are known and results can be scored against
them. `oracle_optimum` finds the lowest objective the oracle allows.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.errors import (
    OracleOptimumWarning,
    ReconstructionWarning,
    RootFindingError,
)
from hybrid_bo.evaluation import Measure, _reconstruct, _unmeasured
from hybrid_bo.hybrid_model import _minimize_f, build_deterministic_equivalent
from hybrid_bo.problem import Problem, Symbolic, System, _states, _trace
from hybrid_bo.sampling import _start_bounds, lhs
from hybrid_bo.solvers import Ipopt, _find_root, solve_with_multistart


@dataclass(frozen=True)
class OracleValues:
    """The oracle values of a set of points, one row each, for scoring: what the states
    and the objective are, as against what was measured and reconstructed."""

    x: np.ndarray  # (n, M)
    y: np.ndarray  # (n, P)
    f: np.ndarray  # (n,)


def _oracle_root(
    system: System,
    oracle_function: cas.Function,
    u: np.ndarray,
    starts: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Solves `g` and the oracle together for every state (M + P,) at u (N,), from each
    row of `starts` in turn.

    Raises `RootFindingError` if no start gives a root.
    """
    w = Symbolic.sym("w", system.n_x + system.n_y)
    x, y = w[: system.n_x], w[system.n_x :]
    residuals = cas.vertcat(system.g(x, y, u), oracle_function(x, y, u) - y)
    return _find_root(w, residuals, _states(system), starts, tol)


def _check_optimum(f: float, optimum: float, u: np.ndarray) -> None:
    """Warns with `OracleOptimumWarning` if the oracle objective f at u is below
    `optimum`, beyond the solvers' tolerance of a relative 1e-6."""
    if f < optimum - 1e-6 * max(1.0, abs(optimum)):
        msg = (
            f"At u = {u}, the oracle objective is {f!r}, below the stated oracle "
            f"optimum {optimum!r}, so the regret is wrong"
        )
        warnings.warn(msg, OracleOptimumWarning, stacklevel=3)


def measure_oracle(
    problem: Problem,
    oracle: Callable[[Any], Any],
    noise_std: float,
    optimum: float | None = None,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> Measure:
    """The oracle simulation as a measurement, for benchmark experiments.

    `oracle` is the unknown equations in closed form, a callable over the variables
    like `g` (`lambda v: [cas.sin(v.u)]`), giving y = oracle(x, y, u); it may read y,
    since only the GP needs the explicit form. The method never sees it; only the
    measurement uses it. For a real system, including an in silico
    simulation of a plant, write your own measurement instead (see the module docs).
    The measurement solves `g` and the oracle together for every state, from
    `n_starts` Latin hypercube starts over the start bounds, reads the measured
    variables and adds normal noise with std `noise_std`. The noise is drawn even when
    `noise_std` is 0, so the draws don't depend on it. The oracle is traced once,
    here.

    With `optimum`, the stated oracle optimum, the measurement also computes the oracle
    objective at each point and warns with `OracleOptimumWarning` if that is below it.

    Raises `ValueError` if the oracle doesn't give one value per unknown state. The
    measurement raises `RootFindingError` if no start gives a root.
    """
    system = problem.system
    oracle_function = _trace_oracle(system, oracle)
    states = _states(system)
    measured_columns = [states.index(var) for var in system.measured]

    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The measured variables at u (N,), with noise."""
        # The states, and the objective there if there is an optimum to check
        starts = lhs(*_start_bounds(states), n_starts, rng)
        root = _oracle_root(system, oracle_function, u, starts, tol)
        if optimum is not None:
            f = problem.f(root[: system.n_x], root[system.n_x :], u).full().item()
            _check_optimum(f, optimum, u)

        # The sensors read the measured variables, with noise
        noise = rng.normal(0.0, noise_std, size=system.n_y)
        return root[measured_columns] + noise

    return measure


def oracle_values(
    problem: Problem,
    oracle: Callable[[Any], Any],
    u: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> OracleValues:
    """The oracle values at each row of u (n, N), for scoring a dataset afterwards.

    `oracle` is written as for `measure_oracle`, and each point solves `g` and the oracle
    together from `n_starts` Latin hypercube starts over the start bounds. Pass a
    generator of its own, so this doesn't disturb a run's draws. Raises `ValueError` if
    the oracle doesn't give one value per unknown state, and `RootFindingError` if no
    start gives a root.
    """
    system = problem.system
    oracle_function = _trace_oracle(system, oracle)
    states = _states(system)
    rows = [
        _oracle_root(
            system,
            oracle_function,
            point,
            lhs(*_start_bounds(states), n_starts, rng),
            tol,
        )
        for point in u
    ]
    array = np.reshape(rows, (len(u), len(states)))
    x, y = array[:, : system.n_x], array[:, system.n_x :]
    f = np.array([problem.f(*row).full().item() for row in zip(x, y, u, strict=True)])
    return OracleValues(x, y, f)


def oracle_incumbents(dataset: Dataset, f_oracle: np.ndarray) -> np.ndarray:
    """The benchmark metric: the oracle objective at the incumbent's point after each
    evaluation, shape (n,), from the oracle objective `f_oracle` (n,) at the rows of
    `dataset`.

    The incumbent's point is the one the optimizer believes is best, so this is
    what acting on the run would give. Under noise it can be worse than the best
    oracle value so far. A later point that only ties the incumbent doesn't replace
    it.
    """
    # A point becomes the incumbent's point when it beats every earlier one
    best_before = np.minimum.accumulate(np.concatenate([[np.inf], dataset.f[:-1]]))
    improves = dataset.f < best_before
    incumbent_index = np.maximum.accumulate(
        np.where(improves, np.arange(len(dataset.f)), 0)
    )
    return f_oracle[incumbent_index]


def check_reconstruction(
    system: System, dataset: Dataset, values: OracleValues, tol: float = 1e-10
) -> None:
    """Warns with `ReconstructionWarning` for each row of `dataset` whose reconstruction
    isn't on the root its oracle values lie on.

    Noise moves the measured variables, so the two are compared after one extra solve of
    the root-finding problem started at the oracle values (no random draws). If that
    solve gives a different root, or none, the measured variables don't determine the
    states uniquely: a method would learn from states the system didn't settle at.
    """
    states, unmeasured = _states(system), _unmeasured(system)
    is_unmeasured = np.array([var in unmeasured for var in states])
    measured_columns = [states.index(var) for var in system.measured]
    reconstructions = np.column_stack([dataset.x, dataset.y])
    oracle_rows = np.column_stack([values.x, values.y])
    for u, row, oracle_row in zip(dataset.u, reconstructions, oracle_rows, strict=True):
        # Solving the same root-finding problem from the oracle values must give the
        # same root (noise moves both along the branch the oracle lies on)
        start = oracle_row[is_unmeasured][np.newaxis]
        try:
            nearest = _reconstruct(system, u, row[measured_columns], start, tol)
            agrees = np.allclose(nearest, row, rtol=1e-6, atol=1e-6)
        except RootFindingError:
            agrees = False
        if not agrees:
            names = ", ".join(var.name for var in states)
            msg = (
                f"At u = {u}, the root-finding problem gave ({names}) = {row}, which "
                f"is not the root the oracle values {oracle_row} lie on; the measured "
                f"variables don't determine the states uniquely"
            )
            warnings.warn(msg, ReconstructionWarning, stacklevel=2)


def _trace_oracle(system: System, oracle: Callable[[Any], Any]) -> cas.Function:
    """The oracle traced like g, over the columns (x, y, u).

    Raises `ValueError` unless it gives one value per unknown state.
    """
    oracle_function = _trace("oracle", oracle, system.x, system.y, system.u)
    if oracle_function.size_out(0) != (system.n_y, 1):
        msg = (
            f"The oracle must give one value per unknown state ({system.n_y}) as a "
            f"column, got shape {oracle_function.size_out(0)}"
        )
        raise ValueError(msg)
    return oracle_function


def oracle_optimum(
    problem: Problem,
    oracle: Callable[[Any], Any],
    rng: np.random.Generator,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> float:
    """The oracle optimum: the lowest objective the oracle allows, min f(x, y, u)
    subject to g(x, y, u) = 0 and y = oracle(x, y, u), within the bounds.

    `oracle` is written as for `measure_oracle`, and may read y. This is the
    deterministic equivalent of one scenario with the unknown equations tied to the
    oracle, minimizing f, as `optimize_hybrid_model` does with the surrogate model.
    Solved by `solve_with_multistart` from `n_starts` Latin hypercube starts over the
    start bounds of u, x and y, so it is the best local optimum found, which can miss
    the global one: use it to find a benchmark experiment's optimum once, from many
    starts, and state that value in the experiment. Raises `ValueError` if the oracle
    doesn't give one value per unknown state, and `NoSolutionError` if every start
    fails.
    """
    # The oracle, and the symbols of the one scenario, whose states are rows
    system = problem.system
    oracle_function = _trace_oracle(system, oracle)
    u = Symbolic.sym("u", system.n_u)
    x = Symbolic.sym("x", 1, system.n_x)
    y = Symbolic.sym("y", 1, system.n_y)

    # The unknown equations read y, so y stays a variable; the oracle takes columns
    unknown_equations = oracle_function(x.T, y.T, u).T - y
    equivalent = build_deterministic_equivalent(
        problem, u, x, y, unknown_equations, _minimize_f, "oracle_optimum"
    )

    # The best of the starts over the start bounds of u, x and y
    starts = lhs(*equivalent.start_bounds, n_starts, rng)
    return solve_with_multistart(equivalent.nlp, starts, solver).f
