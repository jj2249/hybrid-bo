"""Acquisition functions and the acquisition problems built from them.

`EI` and `LCB` name the acquisition function; each method computes it its own way.
Standard BO computes it analytically from the GP's mean and standard deviation
(`standard_acquisition`). Hybrid BO approximates it as set by its approximation
(`hybrid_acquisition`): `SAA`, a weighted sample average over scenarios of the GP's
prediction of the unknown states (paper, Problems 8 and 9), or `Linearization`, EI of
the objective expanded to first order around the nominal states (Lin-EI).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, assert_never

import casadi as cas
import numpy as np

from hybrid_bo.gp import FittedGP, predict
from hybrid_bo.problem import Problem, Symbolic, System, Var, _h_inputs
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.solvers import NLP

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10

# Below this a / tau, log log(1 + exp(a / tau)) is a / tau within about 1e-13
_LOG_SOFTPLUS_LINEAR_BELOW = -30.0


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


@dataclass(frozen=True)
class SAA:
    """The sample-average approximation of hybrid BO's acquisition function: the
    number of scenarios, and whether each scenario's unknown states are variables of
    the acquisition problem.

    With `unknown_states_as_variables` (default True) they are variables tied to the
    GP by an equality constraint (paper, Problem 8), which helps IPOPT. With False,
    the GP's prediction is substituted for them, so the problem is smaller.
    """

    n_scenarios: int = 25
    unknown_states_as_variables: bool = True


@dataclass(frozen=True)
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
    n_scenarios: int = 25
    unknown_states_as_variables: bool = True

    def __post_init__(self) -> None:
        if not (np.isfinite(self.temperature) and self.temperature > 0):
            msg = f"SoftSAA temperature must be finite and > 0, got {self.temperature}"
            raise ValueError(msg)


@dataclass(frozen=True)
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
            "SoftSAA, or EI() or LCB() with SAA"
        )
        raise ValueError(msg)  # noqa: TRY004  (both types are valid, not together)
    if isinstance(acquisition, LCB) and isinstance(approximation, Linearization):
        msg = (
            "LCB isn't implemented with Linearization: use EI() with Linearization, "
            "or EI() or LCB() with SAA"
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


@dataclass(frozen=True)
class AcquisitionProblem:
    """An acquisition problem: the NLP over the decision vector w, the start bounds
    `(lower, upper)` of w for its multistart, and the order of w's variables.

    w holds the N decision variables u, then, for hybrid BO, the S scenario copies of
    the M mechanistic states x (S, M), then those of the unknown states y (S, P) if
    they are variables. `_pack` and `_unpack` are the only code that knows this order.
    The defaults are standard BO's, whose w is u alone.
    """

    nlp: NLP
    start_bounds: tuple[np.ndarray, np.ndarray]
    n_u: int
    n_x: int = 0
    n_scenarios: int = 0
    unknown_states_as_variables: bool = False


def _pack(u: Any, x: Any = None, y: Any = None) -> Any:
    """The decision vector w of u (N,), the scenario states x (S, M) and, if they are
    variables, y (S, P): numpy arrays give a numpy array, CasADi matrices a column.

    The states are stacked column by column, so each state has its S scenario copies
    in a row.
    """
    states = [block for block in (x, y) if block is not None]
    if isinstance(u, np.ndarray):
        return np.concatenate([u, *(block.ravel(order="F") for block in states)])
    return cas.vertcat(u, *(cas.vec(block) for block in states))


def _unpack(
    problem: AcquisitionProblem, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The u (N,), x (S, M) and y (S, P) (None unless they are variables) held in a
    numeric decision vector w of the acquisition problem; undoes `_pack`.

    Raises `ValueError` if w has the wrong length for the problem.
    """
    # One column per state of each scenario; standard BO has no scenarios
    w = np.asarray(w, dtype=float).ravel()
    n_u, n_x, n_scenarios = problem.n_u, problem.n_x, problem.n_scenarios
    n_states = (len(w) - n_u) // n_scenarios if n_scenarios else 0
    n_y = n_states - n_x
    fits = len(w) == n_u + n_scenarios * n_states and n_y >= 0
    if not fits or (n_y > 0) != problem.unknown_states_as_variables:
        msg = f"A decision vector of length {len(w)} doesn't fit this problem"
        raise ValueError(msg)
    states = w[n_u:].reshape((n_scenarios, n_states), order="F")
    y = states[:, n_x:] if problem.unknown_states_as_variables else None
    return w[:n_u], states[:, :n_x], y


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


def nominal_scenario(n_y: int) -> Scenarios:
    """The nominal scenario, linearization's only scenario: zero scenario draws z
    (1, n_y) and weight 1."""
    return Scenarios(np.zeros((1, n_y)), np.ones(1))


def _scenario_unknown_states(
    system: System,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    u: Any,
    x: Any,
) -> Any:
    """The scenarios' unknown states (S, P): unknown state j of scenario s is GP j's
    prediction at the scenario's arguments of h, mean_j + std_j * z_sj, for u (N,)
    and x (S, M) as numpy arrays, or u (N, 1) and x (S, M) as CasADi matrices.
    `fitted` holds one fitted GP per unknown state, in the order of `system.y`.
    """
    # One column of mean and std per unknown state, matching the columns of z
    n_scenarios = len(scenarios.w)
    if isinstance(x, np.ndarray):
        inputs = np.column_stack(_h_inputs(system, np.tile(u, (n_scenarios, 1)), x))
        mean, std = zip(*(predict(gp, inputs) for gp in fitted), strict=True)
        return np.column_stack(mean) + np.column_stack(std) * scenarios.z
    inputs = cas.horzcat(*_h_inputs(system, cas.repmat(u.T, n_scenarios, 1), x))
    mean, std = zip(*(predict(gp, inputs) for gp in fitted), strict=True)
    return cas.horzcat(*mean) + cas.horzcat(*std) * Symbolic(scenarios.z)


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
    approximation: SAA | SoftSAA | Linearization = SAA(),
) -> AcquisitionProblem:
    """The acquisition problem of hybrid BO (paper, Problems 8 and 9).

    `fitted` is the surrogate model: one fitted GP per unknown state, in the order of
    `problem.system.y`. Each scenario s has its own copy x_s of the mechanistic
    states, and the known equations g(x_s, y_s, u) = 0 are constraints. Its unknown
    states y_s are the GPs' prediction at the scenario's arguments of h,
    mean + std * z_s (`_scenario_unknown_states`). With the approximation's
    `unknown_states_as_variables` the y_s are variables too, tied to the prediction by
    an equality constraint (Problem 8), which helps IPOPT; otherwise the prediction is
    substituted for them.

    With f_s = f(x_s, y_s, u), the problem minimizes -SAA-EI, the weighted average of
    min(f_s - incumbent, 0), or SAA-LCB, the weighted mean of the f_s minus
    `std_factor` times their weighted standard deviation. With `SoftSAA`, it
    minimizes -log SoftSAA-EI, -log of the weighted average of
    tau log(1 + exp(a_s / tau)) with a_s = incumbent - f_s, as a logsumexp over the
    scenarios (`_log_softplus`). With `Linearization`, the
    scenarios must be the nominal scenario, whose states are the nominal states, and
    the problem minimizes -Lin-EI: EI of the normal N(f, s^2), with s^2 = a a^T and
    a = df/dz there (`_sensitivity`).

    The variables are u, the x_s and, if variables, the y_s, in `_pack`'s order. Each
    variable is boxed by its bounds and started within its start bounds. Raises
    `TypeError` unless `acquisition` is an `EI` or an `LCB` and `approximation` an
    `SAA`, a `SoftSAA` or a `Linearization`, `NotImplementedError` for `LCB` with
    `Linearization`, and `ValueError` for `LCB` with `SoftSAA`, for scenarios or
    fitted GPs without one column or one GP per unknown state, and with
    `Linearization`, for scenarios other than the nominal scenario.
    """
    # The checks, and the symbols for u and each scenario's mechanistic states
    _check_acquisition(acquisition)
    _check_approximation(acquisition, approximation)
    unknown_states_as_variables = approximation.unknown_states_as_variables
    system = problem.system
    if scenarios.z.shape[1] != system.n_y:
        msg = (
            f"The scenarios need one column of z per unknown state ({system.n_y}), "
            f"got {scenarios.z.shape[1]}"
        )
        raise ValueError(msg)
    if len(fitted) != system.n_y:
        msg = (
            "The surrogate model needs one fitted GP per unknown state "
            f"({system.n_y}), got {len(fitted)}"
        )
        raise ValueError(msg)
    is_nominal = len(scenarios.w) == 1 and not np.any(scenarios.z)
    if isinstance(approximation, Linearization) and not is_nominal:
        msg = "Linearization runs over the nominal scenario alone (nominal_scenario)"
        raise ValueError(msg)
    n_scenarios = len(scenarios.w)
    u = Symbolic.sym("u", system.n_u)
    # Row s: the mechanistic states of scenario s
    x = Symbolic.sym("x", n_scenarios, system.n_x)

    # The scenarios of the unknown states: variables tied to the GP, or the GP itself
    predicted = _scenario_unknown_states(system, fitted, scenarios, u, x)
    y = predicted
    if unknown_states_as_variables:
        y = Symbolic.sym("y", n_scenarios, system.n_y)

    # Each scenario: its known equations, its tie to the GP, and its objective
    constraints, f_parts = [], []
    for s in range(n_scenarios):
        x_s, y_s = x[s, :].T, y[s, :].T
        constraints.append(system.g(x_s, y_s, u))
        if unknown_states_as_variables:
            constraints.append(y_s - predicted[s, :].T)
        f_parts.append(problem.f(x_s, y_s, u))
    f = cas.vertcat(*f_parts)

    # The acquisition function: a weighted average over the scenarios, its smooth
    # version's -log, or Lin-EI at the nominal scenario (the last two EI only, as
    # checked)
    match approximation:
        case Linearization():
            a = _sensitivity(problem, fitted)(x[0, :].T, y[0, :].T, u)
            objective = -_expected_improvement(f[0], cas.sqrt(a @ a.T), incumbent)
        case SoftSAA(temperature=temperature):
            log_smooth = np.log(temperature) + _log_softplus(
                (incumbent - f) / temperature
            )
            objective = -cas.logsumexp(Symbolic(np.log(scenarios.w)) + log_smooth)
        case SAA():
            match acquisition:
                case LCB(std_factor=std_factor):
                    f_mean = Symbolic(scenarios.w).T @ f
                    f_variance = Symbolic(scenarios.w).T @ (f - f_mean) ** 2
                    objective = f_mean - std_factor * cas.sqrt(f_variance)
                case EI():
                    objective = Symbolic(scenarios.w).T @ cas.fmin(f - incumbent, 0)
                case _:
                    assert_never(acquisition)
        case _:
            assert_never(approximation)

    # The Var of each entry of w, each state once per scenario, for the bounds
    def per_scenario(variables: Sequence[Var]) -> np.ndarray:
        """The variables repeated in a row per scenario, (S, len(variables))."""
        return np.tile(np.array(variables, dtype=object), (n_scenarios, 1))

    unknown = per_scenario(system.y) if unknown_states_as_variables else None
    variables = _pack(np.array(system.u, dtype=object), per_scenario(system.x), unknown)

    w = _pack(u, x, y if unknown_states_as_variables else None)
    nlp = NLP(
        cas.Function("acquisition", [w], [objective]),
        np.array([var.lower for var in variables]),
        np.array([var.upper for var in variables]),
        cas.Function("g", [w], [cas.vertcat(*constraints)]),
    )
    return AcquisitionProblem(
        nlp,
        _start_bounds(variables),
        system.n_u,
        system.n_x,
        n_scenarios,
        unknown_states_as_variables,
    )


def _scenario_states(
    system: System,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    problem: AcquisitionProblem,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The scenario states x (S, M) and y (S, P) at a point w of a hybrid acquisition
    problem.

    The x_s are read from w. So are the y_s when they are variables; otherwise they
    are the GPs' prediction at each scenario's arguments of h, mean + std * z_s, from
    `fitted`, one fitted GP per unknown state.
    """
    u, x, y = _unpack(problem, w)
    if y is None:
        y = _scenario_unknown_states(system, fitted, scenarios, u, x)
    return x, y
