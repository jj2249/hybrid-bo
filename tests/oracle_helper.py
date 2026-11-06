"""An oracle written as a script would: the unknown equations in closed form, solved with
`g` to play the real system, and to score a dataset afterwards."""

import casadi as cas
import numpy as np

from hybrid_bo.problem import Problem, Symbolic, _to_casadi_function, lhs_points
from hybrid_bo.solvers import _find_root


def _root(problem: Problem, oracle, u: np.ndarray, rng) -> np.ndarray:
    """Every state (M + P,) at u (N,), solving `g` and the oracle together."""
    system = problem.system
    states = system.states
    oracle_function = _to_casadi_function(
        "oracle", oracle, system.x, system.y, system.u
    )
    w = Symbolic.sym("w", len(states))
    x, y = w[: system.n_x], w[system.n_x :]
    residuals = cas.vertcat(system.g(x, y, u), oracle_function(x, y, u) - y)
    starts = lhs_points(states, 100, rng)
    return _find_root(w, residuals, states, starts, 1e-10)


def measure_oracle(problem: Problem, oracle, noise_std: float):
    """A measurement from the oracle: the measured variables at u, with normal noise."""
    system = problem.system
    columns = [system.states.index(var) for var in system.measured]

    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The measured variables at u (N,), with noise."""
        root = _root(problem, oracle, u, rng)
        return root[columns] + rng.normal(0.0, noise_std, size=system.n_y)

    return measure


def oracle_states(problem: Problem, oracle, u: np.ndarray, rng) -> np.ndarray:
    """Every state at each row of u, shape (n, M + P)."""
    return np.array([_root(problem, oracle, point, rng) for point in u])


def oracle_incumbents(f: np.ndarray, f_oracle: np.ndarray) -> np.ndarray:
    """The oracle objective at the point the measured f ranks best, after each point; a
    later point that only ties doesn't replace it."""
    best_before = np.minimum.accumulate(np.concatenate([[np.inf], f[:-1]]))
    improves = f < best_before
    return f_oracle[np.maximum.accumulate(np.where(improves, np.arange(len(f)), 0))]
