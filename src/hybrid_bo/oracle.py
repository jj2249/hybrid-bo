"""The oracle: the unknown equations in closed form, for benchmark experiments, which
score a method.

You write the oracle in the same CasADi namespace as `g`, and the library simulates the
system from `g` and the oracle together (`oracle_measure`). The oracle simulation
returns an `OracleMeasurement`: the noisy measured variables, which are all the method
sees, plus the oracle values of every state, kept for benchmarking. A custom measurement
that knows its oracle values can also return an `OracleMeasurement`.

The difference from a real system is not whether the system is simulated: an in silico
simulation of a plant is a real system, because the method doesn't know its unknown
equations. What makes an oracle is that the benchmark experiment states the unknown
equations exactly, so the oracle values are known and results can be scored against
them. `oracle_optimum` finds the lowest objective the oracle allows.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.problem import Problem, Symbolic, System, _as_text, _states, _trace
from hybrid_bo.sampling import _start_bounds, lhs
from hybrid_bo.solvers import NLP, Ipopt, _find_root, solve_with_multistart


@dataclass(frozen=True)
class OracleMeasurement:
    """What the oracle simulation returns for one trial point: the measured variables
    (with noise; all the method sees) and the oracle values of the states (for
    benchmarking only)."""

    measured: np.ndarray  # (P,), in the order of system.measured
    x: np.ndarray  # (M,)
    y: np.ndarray  # (P,)


@dataclass(frozen=True, repr=False)
class OracleSimulation:
    """The oracle simulation as a measurement, built by `oracle_measure`.

    Called as `measure(u, rng)`. It is data (the system, the traced oracle and the
    settings), and its repr describes it in the run's artifact.
    """

    system: System
    oracle: cas.Function  # traced like g, over the columns (x, y, u)
    noise_std: float
    n_starts: int
    tol: float

    def __call__(self, u: np.ndarray, rng: np.random.Generator) -> OracleMeasurement:
        """The measured variables at u (N,), with noise, and the oracle values.

        Raises `RootFindingError` if no start gives a root.
        """
        system = self.system
        states = _states(system)

        # Oracle values: g and the oracle together, for every state
        w = Symbolic.sym("w", len(states))
        x, y = w[: system.n_x], w[system.n_x :]
        residuals = cas.vertcat(system.g(x, y, u), self.oracle(x, y, u) - y)
        starts = lhs(*_start_bounds(states), self.n_starts, rng)
        root = _find_root(w, residuals, states, starts, self.tol)

        # The sensors read the measured variables, with noise
        measured_columns = [states.index(var) for var in system.measured]
        noise = rng.normal(0.0, self.noise_std, size=system.n_y)
        return OracleMeasurement(
            measured=root[measured_columns] + noise,
            x=root[: system.n_x],
            y=root[system.n_x :],
        )

    def __repr__(self) -> str:
        """The oracle as text in the variable names, and the settings."""
        system = self.system
        oracle_text = _as_text(self.oracle, system.x, system.y, system.u)
        return (
            f"OracleSimulation(y = {oracle_text}, noise_std={self.noise_std}, "
            f"n_starts={self.n_starts}, tol={self.tol})"
        )


def oracle_measure(
    system: System,
    oracle: Callable[[Any], Any],
    noise_std: float,
    n_starts: int = 100,
    tol: float = 1e-10,
) -> OracleSimulation:
    """The oracle simulation as a measurement, for benchmark experiments.

    `oracle` is the unknown equations in closed form, a callable over the variables
    like `g` (`lambda v: [cas.sin(v.u)]`), giving y = oracle(x, u). The method never
    sees it; only the measurement uses it. For a real system, including an in silico
    simulation of a plant, write your own measurement instead (see the module docs).
    The measurement solves `g` and the oracle together for every state, from
    `n_starts` Latin hypercube starts over the start bounds, reads the measured
    variables and adds normal noise with std `noise_std`. The noise is drawn even when
    `noise_std` is 0, so the draws don't depend on it. The oracle is traced once,
    here.

    Raises `ValueError` if the oracle doesn't give one value per unknown state. The
    measurement raises `RootFindingError` if no start gives a root.
    """
    oracle_function = _trace_oracle(system, oracle)
    return OracleSimulation(system, oracle_function, noise_std, n_starts, tol)


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
    subject to g(x, y, u) = 0 and y = oracle(x, u), within the bounds.

    `oracle` is written as for `oracle_measure`. Solved like the acquisition
    problem: by `solve_with_multistart` from `n_starts` Latin hypercube starts over the
    start bounds of u, x and y, so it is the best local optimum found, which can miss
    the global one: use it to find a benchmark experiment's optimum once, from many
    starts, and state that value in the experiment. Raises `ValueError` if the oracle
    doesn't give one value per unknown state, and `NoSolutionError` if every start
    fails.
    """
    # The variables w = (u, x, y), with g and the oracle as constraints
    system = problem.system
    oracle_function = _trace_oracle(system, oracle)
    variables = (*system.u, *system.x, *system.y)
    w = Symbolic.sym("w", len(variables))
    u = w[: system.n_u]
    x = w[system.n_u : system.n_u + system.n_x]
    y = w[system.n_u + system.n_x :]
    constraints = cas.vertcat(system.g(x, y, u), oracle_function(x, y, u) - y)
    nlp = NLP(
        cas.Function("f", [w], [problem.f(x, y, u)]),
        np.array([var.lower for var in variables]),
        np.array([var.upper for var in variables]),
        cas.Function("g", [w], [constraints]),
    )

    starts = lhs(*_start_bounds(variables), n_starts, rng)
    return solve_with_multistart(nlp, starts, solver).f
