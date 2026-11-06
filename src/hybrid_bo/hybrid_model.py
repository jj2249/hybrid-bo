"""The hybrid model as an optimization problem: the known equations with the surrogate
model embedded, over scenarios of the GPs' prediction of the unknown states.

A `DeterministicEquivalent` is that problem: `solve` it for the next trial point, and pass
it to the plots to see what was minimized. `gp_deterministic_equivalent` builds it for a
given objective, and hybrid BO's acquisition problem (`hybrid_acquisition`) is its one use.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.gp import FittedGP
from hybrid_bo.problem import Problem, Symbolic, Var, start_bounds
from hybrid_bo.solvers import NLP, Ipopt, solve_with_multistart
from hybrid_bo.utils import lhs


@dataclass(frozen=True)
class Solution:
    """The solution of a deterministic equivalent: the trial point u (N,), the objective
    the problem minimizes there (-EI, or -log EI for `SoftSAA`) and, for hybrid BO, the
    scenario states there, x (S, M) and y (S, P), one row per scenario. Standard BO has no
    scenario states, so they are None."""

    u: np.ndarray
    value: float
    x: np.ndarray | None = None
    y: np.ndarray | None = None


@dataclass(frozen=True)
class DeterministicEquivalent:
    """A deterministic equivalent: the NLP over the decision vector, the start bounds
    `(lower, upper)` of that vector for its multistart, and the order of its variables.

    The decision vector holds the N decision variables u, then the S scenario copies of
    the M mechanistic states x (S, M), then those of the unknown states y (S, P) if they
    are variables. The defaults are those of a problem over u alone, such as standard BO's
    acquisition problem.

    `weights` (S,) are the scenario weights. `scenario_values` maps the decision vector to
    f under each scenario (S, 1) and the scenarios' unknown states y (S, P), read from the
    vector or predicted by the GPs. `ei` gives EI from values of the objective.
    """

    nlp: NLP
    start_bounds: tuple[np.ndarray, np.ndarray]
    n_u: int
    ei: Callable[[np.ndarray], np.ndarray]
    n_x: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    unknown_states_as_variables: bool = False
    scenario_values: cas.Function | None = None

    @property
    def n_scenarios(self) -> int:
        """The number of scenarios: 0 for a problem over u alone."""
        return len(self.weights)

    def solve(
        self,
        rng: np.random.Generator,
        *,
        n_starts: int = 10,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
    ) -> Solution:
        """The minimizer, from `n_starts` Latin hypercube starts over the start bounds.

        Raises `NoSolutionError` if every start fails.
        """
        starts = lhs(*self.start_bounds, n_starts, rng)
        w, value = solve_with_multistart(self.nlp, starts, solver)
        if self.scenario_values is None:
            return Solution(w, value)
        u, x, _ = self.unpack(w)
        return Solution(u, value, x, self.scenario_values(w)[1].full())

    def pack(self, solution: Solution) -> np.ndarray:
        """The decision vector of a solution of this problem."""
        y = solution.y if self.unknown_states_as_variables else None
        return _pack(solution.u, solution.x, y)

    def unpack(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The u (N,), x (S, M) and y (S, P) (None unless they are variables) held in a
        numeric decision vector w of this problem; undoes `pack`.

        Raises `ValueError` if w has the wrong length for the problem.
        """
        # One column per state of each scenario; a problem over u alone has no scenarios
        w = np.asarray(w, dtype=float).ravel()
        n_u, n_x, n_scenarios = self.n_u, self.n_x, self.n_scenarios
        n_states = (len(w) - n_u) // n_scenarios if n_scenarios else 0
        n_y = n_states - n_x
        fits = len(w) == n_u + n_scenarios * n_states and n_y >= 0
        if not fits or (n_y > 0) != self.unknown_states_as_variables:
            msg = f"A decision vector of length {len(w)} doesn't fit this problem"
            raise ValueError(msg)
        states = w[n_u:].reshape((n_scenarios, n_states), order="F")
        y = states[:, n_x:] if self.unknown_states_as_variables else None
        return w[:n_u], states[:, :n_x], y


def _pack(u: Any, x: Any = None, y: Any = None) -> Any:
    """The decision vector of u (N,), the scenario states x (S, M) and, if they are
    variables, y (S, P): numpy arrays give a numpy array, CasADi matrices a column.

    The states are stacked column by column, so each state has its S scenario copies
    in a row.
    """
    states = [block for block in (x, y) if block is not None]
    if isinstance(u, np.ndarray):
        return np.concatenate([u, *(block.ravel(order="F") for block in states)])
    return cas.vertcat(u, *(cas.vec(block) for block in states))


def gp_deterministic_equivalent(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    z: np.ndarray,
    w: np.ndarray,
    objective: Callable[[Any], Any],
    ei: Callable[[np.ndarray], np.ndarray],
    unknown_states_as_variables: bool,
) -> DeterministicEquivalent:
    """The hybrid model over the scenarios of the surrogate model, minimizing `objective`
    of f under each scenario (S, 1). The scenarios are the draws z (S, P), one row per
    scenario, and the weights w (S,), which sum to 1.

    `fitted` is the surrogate model: one fitted GP per unknown state, in the order of
    `problem.system.y`. Each scenario s has its own copies x_s and y_s of the states, and
    the known equations g(x_s, y_s, u) = 0 are constraints. The unknown states y_s are
    the GPs' prediction at the scenario's arguments of h, mean + std * z_s. With `unknown_states_as_variables` the y_s are variables
    too, tied by y_s - prediction = 0 (paper, Problem 8), which helps IPOPT. Otherwise the
    prediction is substituted for them, so they add no variables and have no bounds.

    The variables are u, the x_s and, if variables, the y_s, in `_pack`'s order, each
    boxed by its bounds and started within its start bounds.

    Raises `ValueError` if z and w don't match, a weight is negative, the weights don't
    sum to 1, or there isn't one column of z and one fitted GP per unknown state.
    """
    # The checks
    system = problem.system
    if z.ndim != 2 or w.shape != (len(z),):
        msg = (
            f"Scenarios need z of shape (S, P) and w of shape (S,), got {z.shape} "
            f"and {w.shape}"
        )
        raise ValueError(msg)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        msg = f"Scenario weights must be >= 0 and sum to 1, got {w}"
        raise ValueError(msg)
    if z.shape[1] != system.n_y:
        msg = (
            f"The scenarios need one column of z per unknown state ({system.n_y}), "
            f"got {z.shape[1]}"
        )
        raise ValueError(msg)
    if len(fitted) != system.n_y:
        msg = (
            "The surrogate model needs one fitted GP per unknown state "
            f"({system.n_y}), got {len(fitted)}"
        )
        raise ValueError(msg)

    # The symbols for u and each scenario's mechanistic states (row s: scenario s), and
    # the unknown states: variables tied to the GP, or the GP itself
    n_scenarios = len(w)
    u = Symbolic.sym("u", system.n_u)
    x = Symbolic.sym("x", n_scenarios, system.n_x)

    # Unknown state j of scenario s is GP j's prediction at the scenario's arguments of h,
    # mean_j + std_j * z_sj: one column of mean and std per unknown state, like z
    h_arguments = cas.horzcat(*system.h_arguments(cas.repmat(u.T, n_scenarios, 1), x))
    mean, std = zip(*(gp.predict(h_arguments) for gp in fitted), strict=True)
    prediction = cas.horzcat(*mean) + cas.horzcat(*std) * Symbolic(z)
    y = prediction
    if unknown_states_as_variables:
        y = Symbolic.sym("y", n_scenarios, system.n_y)

    # Each scenario: its known equations, its tie to the GP, and its f
    constraints, f_parts = [], []
    for s in range(n_scenarios):
        x_s, y_s = x[s, :].T, y[s, :].T
        constraints.append(system.g(x_s, y_s, u))
        if unknown_states_as_variables:
            constraints.append((y - prediction)[s, :].T)
        f_parts.append(problem.f(x_s, y_s, u))

    # The Var of each entry of the decision vector, each state once per scenario
    def per_scenario(variables: Sequence[Var]) -> np.ndarray:
        """The variables repeated in a row per scenario, (S, len(variables))."""
        return np.tile(np.array(variables, dtype=object), (n_scenarios, 1))

    unknown = per_scenario(system.y) if unknown_states_as_variables else None
    variables = _pack(np.array(system.u, dtype=object), per_scenario(system.x), unknown)

    f = cas.vertcat(*f_parts)
    decisions = _pack(u, x, y if unknown_states_as_variables else None)
    nlp = NLP(
        cas.Function("acquisition", [decisions], [objective(f)]),
        np.array([var.lower for var in variables]),
        np.array([var.upper for var in variables]),
        cas.Function("g", [decisions], [cas.vertcat(*constraints)]),
    )
    return DeterministicEquivalent(
        nlp,
        start_bounds(variables),
        system.n_u,
        ei,
        system.n_x,
        w,
        unknown_states_as_variables,
        cas.Function("scenario_values", [decisions], [f, y]),
    )
