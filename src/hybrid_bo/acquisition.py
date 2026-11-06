"""Expected improvement and the acquisition problems built from it.

Standard BO computes EI analytically from the GP's mean and standard deviation
(`standard_acquisition`). Hybrid BO approximates it over scenarios of the GP's prediction
of the unknown states (`hybrid_acquisition`): `SAA`, a weighted sample average (paper,
Problems 8 and 9), or `SoftSAA`, its smooth version.

Each gives a `DeterministicEquivalent`, and its `solve` is one iteration of BO. `replay`
evaluates a problem at given points, to draw it.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.errors import RootFindingError
from hybrid_bo.gp import FittedGP
from hybrid_bo.hybrid_model import (
    DeterministicEquivalent,
    Solution,
    gp_deterministic_equivalent,
)
from hybrid_bo.problem import Problem, Symbolic, start_bounds
from hybrid_bo.solvers import NLP, RootProblem, find_root
from hybrid_bo.utils import log_softplus

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10

# A replay's root solves each equation to this
_REPLAY_TOLERANCE = 1e-8


@dataclass(frozen=True, kw_only=True)
class SAA:
    """The sample-average approximation of hybrid BO's EI: whether each scenario's
    unknown states are variables of the acquisition problem.

    With `unknown_states_as_variables` (default True) they are variables tied to the
    GP by an equality constraint (paper, Problem 8), which helps IPOPT. With False,
    the GP's prediction is substituted for them, so the problem is smaller.
    """

    unknown_states_as_variables: bool = True

    def objective(self, f: Any, w: np.ndarray, incumbent: float) -> Any:
        """-SAA-EI of f (S, 1) with weights w (S,): the weighted average of
        min(f_s - incumbent, 0)."""
        return Symbolic(w).T @ cas.fmin(f - incumbent, 0)

    def ei(self, value: np.ndarray) -> np.ndarray:
        """EI from values of the objective."""
        return -value


@dataclass(frozen=True, kw_only=True)
class SoftSAA:
    """The soft sample-average approximation of hybrid BO's acquisition function
    (SoftSAA-EI): SAA-EI with each scenario's improvement max(a, 0) replaced by the
    smooth maximum tau log(1 + exp(a / tau)) at the temperature tau, and the problem
    minimizing -log of the average, so it has no flat regions or kinks.

    The temperature is in units of the objective, and `unknown_states_as_variables`
    means what it means for `SAA`. Raises `ValueError` unless the temperature is
    positive and finite.
    """

    temperature: float
    unknown_states_as_variables: bool = True

    def __post_init__(self) -> None:
        if not (np.isfinite(self.temperature) and self.temperature > 0):
            msg = f"SoftSAA temperature must be finite and > 0, got {self.temperature}"
            raise ValueError(msg)

    def objective(self, f: Any, w: np.ndarray, incumbent: float) -> Any:
        """-log SoftSAA-EI of f (S, 1) with weights w (S,): -log of the weighted average
        of tau log(1 + exp(a_s / tau)) with a_s = incumbent - f_s, as a logsumexp over
        the scenarios."""
        tau = self.temperature
        log_smooth = np.log(tau) + log_softplus((incumbent - f) / tau)
        return -cas.logsumexp(Symbolic(np.log(w)) + log_smooth)

    def ei(self, value: np.ndarray) -> np.ndarray:
        """EI from values of the objective, which is -log EI."""
        return np.exp(-value)


def expected_improvement(mean: Any, std: Any, incumbent: float) -> Any:
    """Expected improvement below the incumbent of a normal with this mean and std,
    for CasADi expressions."""
    improvement = incumbent - mean
    gamma = improvement / (std + _EI_STD_EPSILON)
    cdf = 0.5 * (1 + cas.erf(gamma / np.sqrt(2)))
    pdf = cas.exp(-0.5 * gamma**2) / np.sqrt(2 * np.pi)
    return improvement * cdf + std * pdf


def standard_acquisition(
    problem: Problem,
    fitted: FittedGP,
    incumbent: float,
) -> DeterministicEquivalent:
    """The acquisition problem of standard BO, over the decision variables within
    their bounds and started within their start bounds.

    The GP models u -> f directly, and the problem minimizes -EI, computed analytically
    from its mean and std.
    """
    u_vars = problem.system.u
    u = Symbolic.sym("u", len(u_vars))
    mean, std = fitted.predict(u.T)
    objective = -expected_improvement(mean, std, incumbent)
    nlp = NLP(
        cas.Function("acquisition", [u], [objective]),
        np.array([var.lower for var in u_vars]),
        np.array([var.upper for var in u_vars]),
    )
    return DeterministicEquivalent(nlp, start_bounds(u_vars), len(u_vars))


def hybrid_acquisition(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    z: np.ndarray,
    w: np.ndarray,
    incumbent: float,
    *,
    approximation: SAA | SoftSAA,
) -> DeterministicEquivalent:
    """The acquisition problem of hybrid BO (paper, Problems 8 and 9): the deterministic
    equivalent (`gp_deterministic_equivalent`) of the scenarios, whose objective is EI as
    `approximation` gives it. `fitted` is the surrogate model.

    The scenarios are the draws z (S, P) and the weights w (S,). For a sample average,
    `z = rng.standard_normal((S, P))` and `w = np.full(S, 1 / S)`. Draw z once, before the
    first iteration, and pass the same one every time: nothing here checks that.

    Raises `ValueError` if z and w don't match, the weights aren't a distribution, or there
    isn't one column of z and one fitted GP per unknown state.
    """
    z, w = np.asarray(z, dtype=float), np.asarray(w, dtype=float)
    built = gp_deterministic_equivalent(
        problem,
        fitted,
        z,
        w,
        lambda f: approximation.objective(f, w, incumbent),
        approximation.unknown_states_as_variables,
    )
    return dataclasses.replace(built, ei=approximation.ei)


@dataclass(frozen=True)
class Replay:
    """An acquisition problem at the rows of u (n, N): its objective there (n,), EI (n,)
    and, for hybrid BO, f (n, S) and the states x (n, S, M) and y (n, S, P) under each
    scenario. NaN where the scenario states weren't solved."""

    values: np.ndarray
    ei: np.ndarray
    f: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None


def replay(
    problem: DeterministicEquivalent, solution: Solution, u: np.ndarray
) -> Replay:
    """The acquisition problem with u fixed at each row of u (n, N), for drawing it: the
    function the problem minimized, whatever the equations are.

    Standard BO's is analytic. With u fixed, hybrid BO's constraints are a square system
    in the scenario states, which Newton's method solves point by point outwards from
    `solution`, each from the nearest point solved, to stay on the branch the optimizer
    chose. A point fails if the residual norm is above `_REPLAY_TOLERANCE` or a state is
    outside its bounds.
    """
    if problem.scenario_values is None:
        values = problem.nlp.f.map(len(u))(u.T).full().ravel()
        return Replay(values, problem.ei(values))

    # The constraints as a root problem: the states are unknowns and u a parameter
    nlp, n_u = problem.nlp, problem.n_u
    assert nlp.g is not None and solution.x is not None and solution.y is not None
    stored = problem.pack(solution)[n_u:]
    states, u_fixed = Symbolic.sym("v", len(stored)), Symbolic.sym("u", n_u)
    equations = cas.Function(
        "replay", [states, u_fixed], [nlp.g(cas.vertcat(u_fixed, states))]
    )
    root_problem = RootProblem(equations, nlp.lower[n_u:], nlp.upper[n_u:])

    # Outwards from the stored solution, each point from the nearest one solved
    solved_u, solved_states = [solution.u], [stored]
    n_scenarios, n_y = solution.y.shape
    values = np.full(len(u), np.nan)
    f = np.full((len(u), n_scenarios), np.nan)
    x = np.full((len(u), *solution.x.shape), np.nan)
    y = np.full((len(u), n_scenarios, n_y), np.nan)
    for i in np.argsort(np.linalg.norm(u - solution.u, axis=1)):
        nearest = np.argmin(np.linalg.norm(np.array(solved_u) - u[i], axis=1))
        start = solved_states[nearest][np.newaxis]
        try:
            root = find_root(root_problem, u[i], start, _REPLAY_TOLERANCE)
        except RootFindingError:
            continue
        w = np.concatenate([u[i], root])
        values[i] = float(nlp.f(w))
        f_i, y_i = problem.scenario_values(w)
        f[i], y[i], x[i] = f_i.full().ravel(), y_i.full(), problem.unpack(w)[1]
        solved_u.append(u[i])
        solved_states.append(root)
    return Replay(values, problem.ei(values), f, x, y)
