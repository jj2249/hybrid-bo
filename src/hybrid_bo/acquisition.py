"""Acquisition functions and the acquisition problems built from them.

`EI` and `LCB` name the acquisition function; each method computes it its own way.
Standard BO computes it analytically from the GP's mean and standard deviation
(`standard_acquisition`). Hybrid BO approximates it as set by its approximation
(`hybrid_acquisition`): `SAA`, a weighted sample average over scenarios of the GP's
prediction of the unknown states (paper, Problems 8 and 9), or `Linearization`, EI of
the objective expanded to first order around the nominal states (Lin-EI).

`replay` evaluates an acquisition problem again after the run, with u fixed at given
points, for the iteration figures (ADR 0005).
"""

import warnings
from dataclasses import dataclass
from typing import Any, assert_never

import casadi as cas
import numpy as np

from hybrid_bo.errors import ReplayWarning
from hybrid_bo.gp import FittedGP, predict
from hybrid_bo.hybrid_model import (
    ScenarioProblem,
    Scenarios,
    _pack,
    _unpack_scenarios,
    scenario_problem,
)
from hybrid_bo.problem import Problem, Symbolic, _h_inputs
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.solvers import NLP

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10

# Below this a / tau, log log(1 + exp(a / tau)) is a / tau within about 1e-13
_LOG_SOFTPLUS_LINEAR_BELOW = -30.0

# A replay's root solves each equation to this
_RESIDUAL_TOLERANCE = 1e-8
# The replay at the trial point is the stored solution to these, loose enough for
# IPOPT's and Newton's tolerances, tight enough to tell another root
_REPLAY_RTOL = 1e-4
_REPLAY_ATOL = 1e-6


@dataclass(frozen=True, kw_only=True)
class EI:
    """Expected improvement over the incumbent."""


@dataclass(frozen=True, kw_only=True)
class LCB:
    """The lower confidence bound, mean - std_factor * std.

    Raises `ValueError` if `std_factor` is negative or not finite.
    """

    std_factor: float

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


@dataclass(frozen=True, kw_only=True)
class SAA:
    """The sample-average approximation of hybrid BO's acquisition function: the
    number of scenarios, and whether each scenario's unknown states are variables of
    the acquisition problem.

    With `unknown_states_as_variables` (default True) they are variables tied to the
    GP by an equality constraint (paper, Problem 8), which helps IPOPT. With False,
    the GP's prediction is substituted for them, so the problem is smaller.
    """

    n_scenarios: int
    unknown_states_as_variables: bool = True


@dataclass(frozen=True, kw_only=True)
class SoftSAA:
    """The soft sample-average approximation of hybrid BO's acquisition function
    (SoftSAA-EI): SAA-EI with each scenario's improvement max(a, 0) replaced by the
    smooth maximum tau log(1 + exp(a / tau)) at the temperature tau, and the problem
    minimizing -log of the average, so it has no flat regions or kinks. It gives EI
    only.

    The temperature is in units of the objective. `n_scenarios` and
    `unknown_states_as_variables` mean what they mean for `SAA`. Raises `ValueError`
    unless the temperature is positive and finite.
    """

    temperature: float
    n_scenarios: int
    unknown_states_as_variables: bool = True

    def __post_init__(self) -> None:
        if not (np.isfinite(self.temperature) and self.temperature > 0):
            msg = f"SoftSAA temperature must be finite and > 0, got {self.temperature}"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class Linearization:
    """The linearization of hybrid BO's acquisition function (Lin-EI): the objective
    expanded to first order in the scenario draws around the nominal states, which
    makes it normal, so EI has a closed form. It gives EI only.

    `unknown_states_as_variables` means what it means for `SAA`, for the nominal
    scenario's unknown states.
    """

    unknown_states_as_variables: bool = True


def _check_approximation(acquisition: EI | LCB, approximation: object) -> None:
    """Raises `TypeError` unless `approximation` is an `SAA`, a `SoftSAA` or a
    `Linearization`, `ValueError` for `LCB` with `SoftSAA`, and
    `NotImplementedError` for `LCB` with `Linearization`.

    Hybrid BO's settings and acquisition builder check their approximation with this.
    """
    if not isinstance(approximation, SAA | SoftSAA | Linearization):
        msg = (
            "The approximation must be SAA(n_scenarios, unknown_states_as_variables), "
            "SoftSAA(temperature, n_scenarios, unknown_states_as_variables) or "
            f"Linearization(unknown_states_as_variables), got {approximation!r}"
        )
        raise TypeError(msg)
    if isinstance(acquisition, LCB) and isinstance(approximation, SoftSAA):
        msg = (
            "SoftSAA softens EI's improvement, so it gives EI only: use EI() with "
            "SoftSAA, or EI() or LCB(std_factor=...) with SAA"
        )
        raise ValueError(msg)  # noqa: TRY004  (both types are valid, not together)
    if isinstance(acquisition, LCB) and isinstance(approximation, Linearization):
        msg = (
            "LCB isn't implemented with Linearization: use EI() with Linearization, "
            "or EI() or LCB(std_factor=...) with SAA"
        )
        raise NotImplementedError(msg)


def _expected_improvement(mean: Any, std: Any, incumbent: float) -> Any:
    """Expected improvement below the incumbent of a normal with this mean and std,
    for CasADi expressions."""
    improvement = incumbent - mean
    gamma = improvement / (std + _EI_STD_EPSILON)
    cdf = 0.5 * (1 + cas.erf(gamma / np.sqrt(2)))
    pdf = cas.exp(-0.5 * gamma**2) / np.sqrt(2 * np.pi)
    return improvement * cdf + std * pdf


def _log_softplus(t: Any) -> Any:
    """log log(1 + exp(t)), elementwise for CasADi expressions, finite with a
    finite gradient for any finite t.

    Far below zero, log(1 + exp(t)) underflows, so its log is taken as t; above
    zero, it is t + log(1 + exp(-t)), so exp doesn't overflow.
    """
    below = cas.log(cas.log1p(cas.exp(t)))
    above = cas.log(t + cas.log1p(cas.exp(-t)))
    exact = cas.if_else(t > 0, above, below)
    return cas.if_else(t < _LOG_SOFTPLUS_LINEAR_BELOW, t, exact)


# An acquisition problem is a scenario problem whose objective is an acquisition
# function; standard BO's is over u alone, with no scenarios
AcquisitionProblem = ScenarioProblem


def standard_acquisition(
    problem: Problem,
    fitted: FittedGP,
    incumbent: float,
    acquisition: EI | LCB,
) -> AcquisitionProblem:
    """The acquisition problem of standard BO, over the decision variables within
    their bounds and started within their start bounds.

    The GP models u -> f directly; the acquisition function is computed analytically
    from its mean and std (minimizing -EI, or LCB). Raises `TypeError` unless
    `acquisition` is an `EI` or an `LCB`.
    """
    _check_acquisition(acquisition)
    u_vars = problem.system.u
    u = Symbolic.sym("u", len(u_vars))
    mean, std = predict(fitted, u.T)
    match acquisition:
        case LCB(std_factor=std_factor):
            objective = mean - std_factor * std
        case EI():
            objective = -_expected_improvement(mean, std, incumbent)
        case _:
            assert_never(acquisition)

    nlp = NLP(
        cas.Function("acquisition", [_pack(u)], [objective]),
        np.array([var.lower for var in u_vars]),
        np.array([var.upper for var in u_vars]),
    )
    return AcquisitionProblem(nlp, _start_bounds(u_vars), len(u_vars))


@dataclass(frozen=True)
class AcquisitionSolution:
    """The solution of one iteration's acquisition problem: the acquisition
    function's value at the trial point, as the objective the problem minimizes
    (-EI for EI, so lower is better for both EI and LCB), and,
    for hybrid BO, the scenario states there, x (S, M) and y (S, P), one row per
    scenario. Standard BO has no scenario states, so they are None."""

    value: float
    x: np.ndarray | None = None
    y: np.ndarray | None = None


def saa_scenarios(n: int, n_y: int, rng: np.random.Generator) -> Scenarios:
    """n scenarios for the sample-average approximation: standard normal scenario
    draws z (n, n_y) and equal weights 1/n."""
    return Scenarios(rng.standard_normal((n, n_y)), np.full(n, 1 / n))


def _sensitivity(problem: Problem, fitted: tuple[FittedGP, ...]) -> cas.Function:
    """The total derivative df/dz of the objective in the scenario draws at the
    nominal scenario (z = 0), as a function of columns x (M, 1), y (P, 1) and u
    (N, 1) to a row a (1, P). `fitted` holds one fitted GP per unknown state.

    The states move with the draws: y = mean(x, u) + L(x, u) z, and x with y through
    g = 0. By the implicit function theorem, dx/dz = -(g_x + g_y mean_x)^-1 g_y L and
    dy/dz = mean_x dx/dz + L, so a = f_x dx/dz + f_y dy/dz. L = diag(std_1, ...,
    std_P), the GPs' stds, since the GPs are independent.
    """
    # Symbols for the point, and the GPs' means and stds there, as columns (P, 1)
    system = problem.system
    x = Symbolic.sym("x", system.n_x)
    y = Symbolic.sym("y", system.n_y)
    u = Symbolic.sym("u", system.n_u)
    inputs = cas.horzcat(*_h_inputs(system, u.T, x.T))
    means, stds = zip(*(predict(gp, inputs) for gp in fitted), strict=True)
    mean, std = cas.vertcat(*means), cas.vertcat(*stds)

    # How the states move with the draws, then the objective
    g, f, L = system.g(x, y, u), problem.f(x, y, u), cas.diag(std)
    mean_x, g_y = cas.jacobian(mean, x), cas.jacobian(g, y)
    dx = -cas.solve(cas.jacobian(g, x) + g_y @ mean_x, g_y @ L)
    dy = mean_x @ dx + L
    a = cas.jacobian(f, x) @ dx + cas.jacobian(f, y) @ dy
    return cas.Function("sensitivity", [x, y, u], [a])


def hybrid_acquisition(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    incumbent: float,
    acquisition: EI | LCB,
    approximation: SAA | SoftSAA | Linearization,
) -> AcquisitionProblem:
    """The acquisition problem of hybrid BO (paper, Problems 8 and 9): the scenario
    problem (`scenario_problem`) of the scenarios, whose objective is the acquisition
    function. `fitted` is the surrogate model, and the approximation's
    `unknown_states_as_variables` says how the problem holds the unknown states.

    With f_s = f(x_s, y_s, u), the problem minimizes -SAA-EI, the weighted average of
    min(f_s - incumbent, 0), or SAA-LCB, the weighted mean of the f_s minus
    `std_factor` times their weighted standard deviation. With `SoftSAA`, it
    minimizes -log SoftSAA-EI, -log of the weighted average of
    tau log(1 + exp(a_s / tau)) with a_s = incumbent - f_s, as a logsumexp over the
    scenarios (`_log_softplus`). With `Linearization`, the
    scenarios must be the nominal scenario, whose states are the nominal states, and
    the problem minimizes -Lin-EI: EI of the normal N(f, s^2), with s^2 = a a^T and
    a = df/dz there (`_sensitivity`).

    Raises `TypeError` unless `acquisition` is an `EI` or an `LCB` and `approximation`
    an `SAA`, a `SoftSAA` or a `Linearization`, `NotImplementedError` for `LCB` with
    `Linearization`, and `ValueError` for `LCB` with `SoftSAA`, for scenarios or
    fitted GPs without one column or one GP per unknown state, and with
    `Linearization`, for scenarios other than the nominal scenario.
    """
    # The checks
    _check_acquisition(acquisition)
    _check_approximation(acquisition, approximation)
    is_nominal = len(scenarios.w) == 1 and not np.any(scenarios.z)
    if isinstance(approximation, Linearization) and not is_nominal:
        msg = "Linearization runs over the nominal scenario alone (nominal_scenario)"
        raise ValueError(msg)

    def objective(f: Any, x: Any, y: Any, u: Any) -> tuple[Any, Any]:
        """The acquisition function: a weighted average over the scenarios, its smooth
        version's -log, or Lin-EI at the nominal scenario (the last two EI only, as
        checked); and the std of f, the weighted std over the scenarios, or Lin-EI's s.
        """
        # The weighted mean and std over the scenarios
        weights = Symbolic(scenarios.w)
        f_mean = weights.T @ f
        f_std = cas.sqrt(weights.T @ (f - f_mean) ** 2)

        # What each approximation minimizes
        match approximation:
            case Linearization():
                a = _sensitivity(problem, fitted)(x[0, :].T, y[0, :].T, u)
                f_std = cas.sqrt(a @ a.T)
                return -_expected_improvement(f[0], f_std, incumbent), f_std
            case SoftSAA(temperature=temperature):
                log_smooth = np.log(temperature) + _log_softplus(
                    (incumbent - f) / temperature
                )
                log_weights = Symbolic(np.log(scenarios.w))
                return -cas.logsumexp(log_weights + log_smooth), f_std
            case SAA():
                match acquisition:
                    case LCB(std_factor=std_factor):
                        return f_mean - std_factor * f_std, f_std
                    case EI():
                        return weights.T @ cas.fmin(f - incumbent, 0), f_std
                    case _:
                        assert_never(acquisition)
            case _:
                assert_never(approximation)

    return scenario_problem(
        problem,
        fitted,
        scenarios,
        objective,
        approximation.unknown_states_as_variables,
        "acquisition",
    )


@dataclass(frozen=True)
class Replay:
    """An acquisition problem replayed at the rows of u (n_points, N): the minimized
    acquisition function there (n_points,) and, for hybrid BO, f under each scenario
    (n_points, S) and the std of f (n_points,), as `scenario_values` gives them. NaN
    where the scenario states weren't solved."""

    values: np.ndarray
    f: np.ndarray | None = None
    f_std: np.ndarray | None = None


def replay(
    problem: AcquisitionProblem,
    solution: AcquisitionSolution,
    u: np.ndarray,
    at_trial: int,
) -> Replay:
    """The acquisition problem with u fixed at each row of u (n_points, N), whose row
    `at_trial` is the trial point the problem's stored `solution` chose (ADR 0005).

    Standard BO's acquisition function is analytic, so it is evaluated there. Hybrid
    BO's needs the scenario states first: with u fixed, the problem's constraints are
    a square system in them, which `_replayed_states` solves outwards from the stored
    solution; the objective and `scenario_values` are then evaluated at each solved
    point. Warns with `ReplayWarning` unless the replay at the trial point is the
    stored solution, its value and states to `_REPLAY_RTOL` relative and
    `_REPLAY_ATOL` absolute: a root-finder that left the optimizer's branch, or found
    no root within the bounds there.
    """
    if problem.scenario_values is None:
        return Replay(problem.nlp.f.map(len(u))(u.T).full().ravel())

    # The objective and the scenario values at the points whose states were solved,
    # in one call each
    states, solved = _replayed_states(problem, solution, u, at_trial)
    values = np.full(len(u), np.nan)
    f = np.full((len(u), problem.n_scenarios), np.nan)
    f_std = np.full(len(u), np.nan)
    points = np.flatnonzero(solved)
    if len(points) > 0:
        w = np.vstack([u[points].T, states[points].T])
        values[points] = problem.nlp.f.map(len(points))(w).full().ravel()
        f_solved, _, f_std_solved = problem.scenario_values.map(len(points))(w)
        f[points] = f_solved.full().T
        f_std[points] = f_std_solved.full().ravel()

    # The check: at the trial point, the stored solution's value and states
    replayed = None
    if solved[at_trial]:
        w_trial = np.concatenate([u[at_trial], states[at_trial]])
        _, x, y = _unpack_scenarios(problem, w_trial)
        replayed = AcquisitionSolution(float(values[at_trial]), x, y)
    _check_trial_point(replayed, solution)
    return Replay(values, f, f_std)


def _replayed_states(
    problem: AcquisitionProblem,
    solution: AcquisitionSolution,
    u: np.ndarray,
    at_trial: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The scenario states at each row of u (n_points, N), as the part of w after u
    (n_points, n_states), and whether each point was solved (n_points,).

    With u fixed, the problem's constraints (each scenario's known equations, and its
    tie to the GPs if the unknown states are variables) are a square system in the
    states, which Newton's method solves, mapped over the points of one ring at a
    time. The rings go outwards from the trial point, row `at_trial`, one median
    nearest-point spacing wide (in units of the start bounds of u); each point starts
    from the nearest point solved in an earlier ring, the trial point from the stored
    solution. That follows the branch the optimizer chose. A point fails if the
    residual is above `_RESIDUAL_TOLERANCE` or a state is outside its bounds. Where the
    equations have several roots this can leave gaps IPOPT wouldn't: it stays within
    the bounds, so it could find a root inside them where Newton found one outside,
    and past a fold, where the branch ends, it could jump to another branch where
    Newton stalls.
    """
    # The stored solution as the start at the trial point
    assert solution.x is not None and solution.y is not None
    y = solution.y if problem.unknown_states_as_variables else None
    stored = _pack(np.empty(0), solution.x, y)
    states = np.full((len(u), len(stored)), np.nan)
    states[at_trial] = stored
    solved = np.zeros(len(u), dtype=bool)

    # With no states (no mechanistic states, the GP substituted) nothing is solved
    if len(stored) == 0:
        return states, ~solved

    # The constraints with u fixed, and Newton's method on them; a mapped call reports
    # no success per instance, so failures are found from the residual
    nlp, n_u = problem.nlp, problem.n_u
    assert nlp.g is not None
    v, u_fixed = Symbolic.sym("v", len(stored)), Symbolic.sym("u", n_u)
    residual = nlp.g(cas.vertcat(u_fixed, v))
    equations = cas.Function("replay_equations", [v, u_fixed], [residual])
    newton = cas.rootfinder("replay", "newton", equations, {"error_on_fail": False})
    lower, upper = nlp.lower[n_u:, np.newaxis], nlp.upper[n_u:, np.newaxis]

    # The rings: the distance from the trial point in units of the median distance
    # between nearest points (1 if that is 0 or there is one point)
    u_lower, u_upper = (bound[:n_u] for bound in problem.start_bounds)
    scaled = (u - u_lower) / (u_upper - u_lower)
    distances = np.linalg.norm(scaled[:, np.newaxis] - scaled[np.newaxis], axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    spacing = spacing if 0 < spacing < np.inf else 1.0
    ring = np.floor(np.linalg.norm(scaled - scaled[at_trial], axis=1) / spacing)

    # Solve ring by ring, every point of a ring in one call; a point is solved if it
    # converged within the bounds
    has_start = np.zeros(len(u), dtype=bool)
    has_start[at_trial] = True
    for r in np.unique(ring):
        points = np.flatnonzero(ring == r)
        known = np.flatnonzero(has_start)
        nearest = known[np.argmin(distances[np.ix_(points, known)], axis=1)]
        nearest[points == at_trial] = at_trial
        u_ring = u[points].T
        roots = newton.map(len(points))(states[nearest].T, u_ring).full()
        errors = np.abs(equations.map(len(points))(roots, u_ring).full())
        converged = np.all(errors <= _RESIDUAL_TOLERANCE, axis=0)
        inside = np.all((roots >= lower) & (roots <= upper), axis=0)
        ring_solved = points[converged & inside]
        states[ring_solved] = roots[:, converged & inside].T
        solved[ring_solved] = has_start[ring_solved] = True

    # The states of the points not solved are no roots
    states[~solved] = np.nan
    return states, solved


def _check_trial_point(
    replayed: AcquisitionSolution | None, solution: AcquisitionSolution
) -> None:
    """Warns with `ReplayWarning` unless the replay at the trial point (None if it
    wasn't solved), its acquisition value and scenario states, is the stored
    solution, as `replay` says. The stack level points the warning at the caller of
    `replay`."""
    if replayed is None:
        msg = (
            "The replay found no root at the trial point, so the acquisition panel "
            "may not show the function the run optimized"
        )
        warnings.warn(msg, ReplayWarning, stacklevel=3)
        return

    # The value and the states as one vector each, compared entry by entry
    def entries(acquisition: AcquisitionSolution) -> np.ndarray:
        """The value, x and y of a solution in one vector."""
        assert acquisition.x is not None and acquisition.y is not None
        parts = (acquisition.value, acquisition.x, acquisition.y)
        return np.concatenate([np.ravel(part) for part in parts])

    replayed_entries, stored_entries = entries(replayed), entries(solution)
    if not np.allclose(
        replayed_entries, stored_entries, rtol=_REPLAY_RTOL, atol=_REPLAY_ATOL
    ):
        msg = (
            f"The replay's acquisition value at the trial point is {replayed.value:.6g}, "
            f"stored {solution.value:.6g}, or its states differ: the root-finder left "
            "the branch the optimizer chose, so the acquisition panel may not show "
            "the function the run optimized"
        )
        warnings.warn(msg, ReplayWarning, stacklevel=3)
