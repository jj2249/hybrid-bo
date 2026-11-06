"""Expected improvement and the acquisition problems built from it.

Standard BO computes EI analytically from the GP's mean and standard deviation
(`standard_acquisition`). Hybrid BO approximates it over scenarios of the GP's prediction
of the unknown states (`hybrid_acquisition`): `SAA`, a weighted sample average (paper,
Problems 8 and 9), or `SoftSAA`, its smooth version.

`hybrid_trial_point` and `standard_trial_point` solve one acquisition problem from fitted
GPs: one iteration of BO.
"""

from dataclasses import dataclass
from typing import Any, assert_never

import casadi as cas
import numpy as np

from hybrid_bo.gp import FittedGP, predict
from hybrid_bo.hybrid_model import (
    DeterministicEquivalent,
    Scenarios,
    _pack,
    _unpack_scenarios,
    gp_deterministic_equivalent,
)
from hybrid_bo.problem import Problem, Symbolic
from hybrid_bo.sampling import _start_bounds, lhs
from hybrid_bo.solvers import NLP, Ipopt, solve_with_multistart

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10

# Below this a / tau, log log(1 + exp(a / tau)) is a / tau within about 1e-13
_LOG_SOFTPLUS_LINEAR_BELOW = -30.0


@dataclass(frozen=True, kw_only=True)
class SAA:
    """The sample-average approximation of hybrid BO's EI: whether each scenario's
    unknown states are variables of the acquisition problem.

    With `unknown_states_as_variables` (default True) they are variables tied to the
    GP by an equality constraint (paper, Problem 8), which helps IPOPT. With False,
    the GP's prediction is substituted for them, so the problem is smaller.
    """

    unknown_states_as_variables: bool = True


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


# An acquisition problem is a deterministic equivalent whose objective is an acquisition
# function; standard BO's is over u alone, with no scenarios
AcquisitionProblem = DeterministicEquivalent


def standard_acquisition(
    problem: Problem,
    fitted: FittedGP,
    incumbent: float,
) -> AcquisitionProblem:
    """The acquisition problem of standard BO, over the decision variables within
    their bounds and started within their start bounds.

    The GP models u -> f directly, and the problem minimizes -EI, computed analytically
    from its mean and std.
    """
    u_vars = problem.system.u
    u = Symbolic.sym("u", len(u_vars))
    mean, std = predict(fitted, u.T)
    objective = -_expected_improvement(mean, std, incumbent)
    nlp = NLP(
        cas.Function("acquisition", [_pack(u)], [objective]),
        np.array([var.lower for var in u_vars]),
        np.array([var.upper for var in u_vars]),
    )
    return AcquisitionProblem(nlp, _start_bounds(u_vars), len(u_vars))


@dataclass(frozen=True)
class AcquisitionSolution:
    """The solution of one iteration's acquisition problem: the trial point u (N,), the
    objective the problem minimizes there (-EI, or -log EI for `SoftSAA`) and, for
    hybrid BO, the scenario states there, x (S, M) and y (S, P), one row per
    scenario. Standard BO has no scenario states, so they are None."""

    u: np.ndarray
    value: float
    x: np.ndarray | None = None
    y: np.ndarray | None = None


def saa_scenarios(n: int, n_y: int, rng: np.random.Generator) -> Scenarios:
    """n scenarios for the sample-average approximation: standard normal scenario
    draws z (n, n_y) and equal weights 1/n."""
    return Scenarios(rng.standard_normal((n, n_y)), np.full(n, 1 / n))


def hybrid_acquisition(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    incumbent: float,
    approximation: SAA | SoftSAA,
) -> AcquisitionProblem:
    """The acquisition problem of hybrid BO (paper, Problems 8 and 9): the deterministic
    equivalent (`gp_deterministic_equivalent`) of the scenarios, whose objective is EI as
    `approximation` gives it. `fitted` is the surrogate model.

    With f_s = f(x_s, y_s, u), `SAA` minimizes -SAA-EI, the weighted average of
    min(f_s - incumbent, 0). `SoftSAA` minimizes -log SoftSAA-EI, -log of the weighted
    average of tau log(1 + exp(a_s / tau)) with a_s = incumbent - f_s, as a logsumexp
    over the scenarios (`_log_softplus`).

    Raises `ValueError` for scenarios or fitted GPs without one column or one GP per
    unknown state.
    """

    def objective(f: Any) -> Any:
        """EI of f (S, 1) under the scenarios, as the approximation gives it."""
        match approximation:
            case SoftSAA(temperature=temperature):
                log_smooth = np.log(temperature) + _log_softplus(
                    (incumbent - f) / temperature
                )
                log_weights = Symbolic(np.log(scenarios.w))
                return -cas.logsumexp(log_weights + log_smooth)
            case SAA():
                return Symbolic(scenarios.w).T @ cas.fmin(f - incumbent, 0)
            case _:
                assert_never(approximation)

    return gp_deterministic_equivalent(
        problem,
        fitted,
        scenarios,
        objective,
        approximation.unknown_states_as_variables,
    )


def standard_trial_point(
    problem: Problem,
    fitted: FittedGP,
    incumbent: float,
    rng: np.random.Generator,
    *,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> AcquisitionSolution:
    """The next trial point of standard BO: solves `standard_acquisition` for the GP of
    the objective `fitted`, from `n_starts` Latin hypercube starts over the start bounds
    of `u`.

    Raises `NoSolutionError` if every start fails.
    """
    acquisition_problem = standard_acquisition(problem, fitted, incumbent)
    starts = lhs(*acquisition_problem.start_bounds, n_starts, rng)
    solution = solve_with_multistart(acquisition_problem.nlp, starts, solver)
    return AcquisitionSolution(solution.w, solution.f)


def hybrid_trial_point(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    incumbent: float,
    rng: np.random.Generator,
    *,
    approximation: SAA | SoftSAA,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> AcquisitionSolution:
    """The next trial point of hybrid BO: solves `hybrid_acquisition` for the surrogate
    model `fitted` (`fit_surrogate`), from `n_starts` Latin hypercube starts over the
    start bounds of its variables. The solution holds the scenario states there.

    Draw `scenarios` once, before the first iteration (`saa_scenarios`), and pass the
    same ones every time: nothing here checks that.

    Raises what `hybrid_acquisition` raises, and `NoSolutionError` if every start fails.
    """
    acquisition_problem = hybrid_acquisition(
        problem, fitted, scenarios, incumbent, approximation
    )
    starts = lhs(*acquisition_problem.start_bounds, n_starts, rng)
    solution = solve_with_multistart(acquisition_problem.nlp, starts, solver)
    u, x, y = _unpack_scenarios(acquisition_problem, solution.w)
    return AcquisitionSolution(u, solution.f, x, y)
