"""Acquisition functions and the acquisition problems built from them.

`EI` and `LCB` name the acquisition function; each method computes it its own way.
Standard BO computes it analytically from the GP's mean and standard deviation
(`standard_acquisition`). Hybrid BO computes it as a weighted sample average over
scenarios of the GP's prediction of the unknown states (`hybrid_acquisition`; paper,
Problems 8 and 9).
"""

from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.gp import GP, FittedGP, predict
from hybrid_bo.problem import Problem, Symbolic, System
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.solvers import NLP

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10


@dataclass(frozen=True)
class EI:
    """Expected improvement over the incumbent."""


@dataclass(frozen=True)
class LCB:
    """The lower confidence bound, mean - std_factor * std.

    Raises `ValueError` if `std_factor` is negative or not finite.
    """

    std_factor: float = 1.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.std_factor) and self.std_factor >= 0):
            msg = f"LCB std_factor must be finite and >= 0, got {self.std_factor}"
            raise ValueError(msg)


def _check_acquisition(acquisition: object) -> None:
    """Raises `TypeError` unless `acquisition` is an `EI` or an `LCB`.

    Every method checks its acquisition function with this.
    """
    if not isinstance(acquisition, EI | LCB):
        msg = (
            "The acquisition function must be EI() or LCB(std_factor), got "
            f"{acquisition!r}"
        )
        raise TypeError(msg)


def _expected_improvement(mean: Any, std: Any, incumbent: float) -> Any:
    """Expected improvement below the incumbent of a normal with this mean and std,
    for CasADi expressions."""
    improvement = incumbent - mean
    gamma = improvement / (std + _EI_STD_EPSILON)
    cdf = 0.5 * (1 + cas.erf(gamma / np.sqrt(2)))
    pdf = cas.exp(-0.5 * gamma**2) / np.sqrt(2 * np.pi)
    return improvement * cdf + std * pdf


def standard_acquisition(
    problem: Problem,
    gp: GP,
    fitted: FittedGP,
    incumbent: float,
    acquisition: EI | LCB,
) -> tuple[NLP, tuple[np.ndarray, np.ndarray]]:
    """The acquisition problem of standard BO, over the decision variables within
    their bounds, and the start bounds for its multistart.

    The GP models u -> f directly; the acquisition function is computed analytically
    from its mean and std (minimizing -EI, or LCB). Raises `TypeError` unless
    `acquisition` is an `EI` or an `LCB`.
    """
    _check_acquisition(acquisition)
    u_vars = problem.system.u
    u = Symbolic.sym("u", len(u_vars))
    mean, std = predict(gp, fitted, u.T)
    if isinstance(acquisition, LCB):
        objective = mean - acquisition.std_factor * std
    else:
        objective = -_expected_improvement(mean, std, incumbent)

    nlp = NLP(
        cas.Function("acquisition", [u], [objective]),
        np.array([var.lower for var in u_vars]),
        np.array([var.upper for var in u_vars]),
    )
    return nlp, _start_bounds(u_vars)


@dataclass(frozen=True)
class Scenarios:
    """The scenarios of hybrid BO's acquisition problem: the scenario draws z (S, P),
    one row per scenario, and the scenario weights w (S,), which sum to 1.

    Raises `ValueError` if the shapes don't match, or a weight is negative, or the
    weights don't sum to 1.
    """

    z: np.ndarray
    w: np.ndarray

    def __post_init__(self) -> None:
        z, w = np.asarray(self.z, dtype=float), np.asarray(self.w, dtype=float)
        if z.ndim != 2 or w.shape != (len(z),):
            msg = (
                f"Scenarios need z of shape (S, P) and w of shape (S,), got {z.shape} "
                f"and {w.shape}"
            )
            raise ValueError(msg)
        if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            msg = f"Scenario weights must be >= 0 and sum to 1, got {w}"
            raise ValueError(msg)

        # Frozen dataclass: set fields through object.__setattr__
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "w", w)


def saa_scenarios(n: int, n_y: int, rng: np.random.Generator) -> Scenarios:
    """n scenarios for the sample-average approximation: standard normal scenario
    draws z (n, n_y) and equal weights 1/n."""
    return Scenarios(rng.standard_normal((n, n_y)), np.full(n, 1 / n))


def _h_inputs(system: System, u: Any, x: Any) -> list[Any]:
    """The arguments of h at each row of u (n, N) and x (n, M), numeric or CasADi, as
    a list of columns in the order of `system.h_inputs`."""
    columns = {var: u[:, i] for i, var in enumerate(system.u)} | {
        var: x[:, j] for j, var in enumerate(system.x)
    }
    return [columns[var] for var in system.h_inputs]


def hybrid_acquisition(
    problem: Problem,
    gp: GP,
    fitted: FittedGP,
    scenarios: Scenarios,
    incumbent: float,
    acquisition: EI | LCB,
    unknown_states_as_variables: bool = True,
) -> tuple[NLP, tuple[np.ndarray, np.ndarray]]:
    """The acquisition problem of hybrid BO (paper, Problems 8 and 9), and the start
    bounds for its multistart.

    Each scenario s has its own copy x_s of the mechanistic states, and the known
    equations g(x_s, y_s, u) = 0 are constraints. Its unknown states y_s are the GP's
    prediction at the scenario's arguments of h, mean + std * z_s. With
    `unknown_states_as_variables` the y_s are variables too, tied to the prediction
    by an equality constraint (Problem 8), which helps IPOPT; otherwise the prediction
    is substituted for them.

    With f_s = f(x_s, y_s, u), the problem minimizes -SAA-EI, the weighted average of
    min(f_s - incumbent, 0), or SAA-LCB, the weighted mean of the f_s minus
    `std_factor` times their weighted standard deviation.

    The variables are u, then the x_s, then the y_s (if variables); the states are
    ordered variable by variable, each with one entry per scenario. So the trial point
    is the first N entries of the solution. Each variable is boxed by its bounds and
    started within its start bounds. Raises `TypeError` unless `acquisition` is an
    `EI` or an `LCB`, and `ValueError` unless the scenarios have one column per
    unknown state.
    """
    # The checks, and the symbols for u and each scenario's mechanistic states
    _check_acquisition(acquisition)
    system = problem.system
    if scenarios.z.shape[1] != system.n_y:
        msg = (
            f"The scenarios need one column of z per unknown state ({system.n_y}), "
            f"got {scenarios.z.shape[1]}"
        )
        raise ValueError(msg)
    n = len(scenarios.w)
    u = Symbolic.sym("u", system.n_u)
    x = Symbolic.sym("x", n, system.n_x)  # row s: the mechanistic states of scenario s

    # The scenarios of the unknown states: the GP at each scenario's arguments of h.
    # One unknown state (P = 1), so the GP's (n, 1) columns match z
    inputs = cas.horzcat(*_h_inputs(system, cas.repmat(u.T, n, 1), x))
    mean, std = predict(gp, fitted, inputs)
    predicted = mean + std * Symbolic(scenarios.z)
    y = Symbolic.sym("y", n, system.n_y) if unknown_states_as_variables else predicted

    # Each scenario: its known equations, its tie to the GP, and its objective
    constraints, f = [], []
    for s in range(n):
        x_s, y_s = x[s, :].T, y[s, :].T
        constraints.append(system.g(x_s, y_s, u))
        if unknown_states_as_variables:
            constraints.append(y_s - predicted[s, :].T)
        f.append(problem.f(x_s, y_s, u))
    f = cas.vertcat(*f)

    # The acquisition function, a weighted average over the scenarios
    weights = Symbolic(scenarios.w)
    if isinstance(acquisition, LCB):
        f_mean = weights.T @ f
        f_variance = weights.T @ (f - f_mean) ** 2
        objective = f_mean - acquisition.std_factor * cas.sqrt(f_variance)
    else:
        objective = weights.T @ cas.fmin(f - incumbent, 0)

    # The variables, with each state once per scenario, and their bounds
    states = [x, y] if unknown_states_as_variables else [x]
    w = cas.vertcat(u, *(cas.vec(block) for block in states))
    unknown = system.y if unknown_states_as_variables else ()
    variables = (*system.u, *system.x, *unknown)
    copies = [1] * system.n_u + [n] * (len(variables) - system.n_u)
    start_lower, start_upper = _start_bounds(variables)

    nlp = NLP(
        cas.Function("acquisition", [w], [objective]),
        np.repeat([var.lower for var in variables], copies),
        np.repeat([var.upper for var in variables], copies),
        cas.Function("g", [w], [cas.vertcat(*constraints)]),
    )
    return nlp, (np.repeat(start_lower, copies), np.repeat(start_upper, copies))
