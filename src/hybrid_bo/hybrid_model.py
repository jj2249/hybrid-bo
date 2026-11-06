"""The hybrid model as an optimization problem: the known equations with the surrogate
model embedded, over scenarios of the GPs' prediction of the unknown states.

Hybrid BO's acquisition function has no closed form, so
`hybrid_deterministic_equivalent` replaces it with a sample average over scenarios. A
`DeterministicEquivalent` is that problem: `solve` it for the next trial point, and pass
it to the plots to see what was minimized.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.acquisition import EI, SoftEI
from hybrid_bo.gp import FittedGP
from hybrid_bo.problem import Problem, Symbolic, Var, start_bounds
from hybrid_bo.solvers import NLP, Ipopt, Solution


@dataclass(frozen=True)
class ScenarioSolution(Solution):
    """The solution of a deterministic equivalent: as well as the decision vector w, the
    trial point u (N,) and the scenario states there, x (S, M) and y (S, P), one row per
    scenario."""

    u: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, kw_only=True)
class DeterministicEquivalent(NLP):
    """An `NLP` over a decision vector that holds the N decision variables u, then the S
    scenario copies of the M mechanistic states x (S, M), then those of the unknown states
    y (S, P) if they are variables. What it adds to an `NLP` says what those variables mean.

    `weights` (S,) are the scenario weights, and `acquisition` is the function the
    objective averages. `scenario_values` maps the decision vector to f under each
    scenario (S, 1) and the scenarios' unknown states y (S, P), read from the vector or
    predicted by the GPs.
    """

    n_u: int
    n_x: int
    weights: np.ndarray
    acquisition: EI | SoftEI
    unknown_states_as_variables: bool
    scenario_values: cas.Function

    @property
    def n_scenarios(self) -> int:
        """The number of scenarios."""
        return len(self.weights)

    def solve(
        self,
        rng: np.random.Generator,
        *,
        n_starts: int = 10,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
    ) -> ScenarioSolution:
        """The minimizer, with the trial point and the scenario states there.

        Raises `NoSolutionError` if every start fails.
        """
        solution = super().solve(rng, n_starts=n_starts, solver=solver)
        u, x, _ = self.unpack(solution.w)
        y = self.scenario_values(solution.w)[1].full()
        return ScenarioSolution(solution.w, solution.value, u, x, y)

    def unpack(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The u (N,), x (S, M) and y (S, P) (None unless they are variables) held in a
        numeric decision vector w of this problem."""
        # One column per state of each scenario
        w = np.asarray(w, dtype=float).ravel()
        states = w[self.n_u :].reshape((self.n_scenarios, -1), order="F")
        y = states[:, self.n_x :] if self.unknown_states_as_variables else None
        return w[: self.n_u], states[:, : self.n_x], y


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


def hybrid_deterministic_equivalent(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    z: np.ndarray,
    w: np.ndarray,
    incumbent: float,
    *,
    acquisition: EI | SoftEI,
    unknown_states_as_variables: bool = True,
) -> DeterministicEquivalent:
    """The acquisition problem of hybrid BO (paper, Problems 8 and 9): the hybrid model
    over the scenarios, minimizing the sample average of `acquisition` under them.

    The scenarios are the draws z (S, P), one row per scenario, and the weights w (S,),
    which sum to 1. For a sample average, `z = rng.standard_normal((S, P))` and
    `w = np.full(S, 1 / S)`. Draw z once, before the first iteration, and pass the same one
    every time: nothing here checks that.

    `fitted` is the surrogate model: one fitted GP per unknown state, in the order of
    `problem.system.y`. Each scenario s has its own copies x_s and y_s of the states, and
    the known equations g(x_s, y_s, u) = 0 are constraints. The unknown states y_s are
    the GPs' prediction at the scenario's arguments of h, mean + std * z_s. With
    `unknown_states_as_variables` the y_s are variables too, tied by y_s - prediction = 0
    (paper, Problem 8), which helps IPOPT. Otherwise the prediction is substituted for
    them, so they add no variables and have no bounds.

    The variables are u, the x_s and, if variables, the y_s, in `_pack`'s order, each
    boxed by its bounds and started within its start bounds.

    Raises `ValueError` if z and w don't match, a weight is negative, the weights don't
    sum to 1, or there isn't one column of z and one fitted GP per unknown state.
    """
    # The checks
    system = problem.system
    z, w = np.asarray(z, dtype=float), np.asarray(w, dtype=float)
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
    return DeterministicEquivalent(
        cas.Function(
            "acquisition", [decisions], [acquisition.objective(f, w, incumbent)]
        ),
        np.array([var.lower for var in variables]),
        np.array([var.upper for var in variables]),
        cas.Function("g", [decisions], [cas.vertcat(*constraints)]),
        start_bounds(variables),
        n_u=system.n_u,
        n_x=system.n_x,
        weights=w,
        acquisition=acquisition,
        unknown_states_as_variables=unknown_states_as_variables,
        scenario_values=cas.Function("scenario_values", [decisions], [f, y]),
    )
