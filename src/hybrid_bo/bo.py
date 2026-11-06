"""Bayesian optimization: the acquisition functions, and standard BO's acquisition problem.

Standard BO's EI is exact: the GP gives the mean and standard deviation of f in closed
form (`expected_improvement`), so `standard_acquisition` minimizes it directly.

Hybrid BO has no closed form, because f depends on the GPs' prediction through the
equations, so a `HybridNLP` averages over scenarios: a sample-average approximation of an
acquisition function. `EI` and `SoftEI` are its objectives for BO, giving SAA-EI and
SAA-SoftEI. `SoftEI` reformulates the acquisition function; the
sample average is the same for both.
"""

from dataclasses import KW_ONLY, dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.gp import FittedGP
from hybrid_bo.problem import Problem, Symbolic, start_bounds
from hybrid_bo.solvers import NLP
from hybrid_bo.utils import log_softplus

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10


@dataclass(frozen=True)
class EI:
    """Expected improvement below the incumbent, as hybrid BO averages it over scenarios:
    SAA-EI (paper, Problems 8 and 9). `expected_improvement` is the closed form standard
    BO uses."""

    incumbent: float

    def __call__(self, f: Any, w: np.ndarray) -> Any:
        """-SAA-EI of f (S, 1) with weights w (S,): the weighted average of
        min(f_s - incumbent, 0)."""
        return Symbolic(w).T @ cas.fmin(f - self.incumbent, 0)


@dataclass(frozen=True)
class SoftEI:
    """EI with each scenario's improvement max(a, 0) replaced by the smooth maximum
    tau log(1 + exp(a / tau)) at the temperature tau, and the problem minimizing -log of
    the average, so it has no flat regions or kinks: SAA-SoftEI.

    The temperature is in units of the objective. Raises `ValueError` unless it is positive
    and finite.
    """

    incumbent: float
    _: KW_ONLY
    temperature: float

    def __post_init__(self) -> None:
        """Raises `ValueError` unless the temperature is positive and finite."""
        if not (np.isfinite(self.temperature) and self.temperature > 0):
            msg = f"SoftEI temperature must be finite and > 0, got {self.temperature}"
            raise ValueError(msg)

    def __call__(self, f: Any, w: np.ndarray) -> Any:
        """-log SAA-SoftEI of f (S, 1) with weights w (S,): -log of the weighted average
        of tau log(1 + exp(a_s / tau)) with a_s = incumbent - f_s, as a logsumexp over
        the scenarios."""
        tau = self.temperature
        log_smooth = np.log(tau) + log_softplus((self.incumbent - f) / tau)
        return -cas.logsumexp(Symbolic(np.log(w)) + log_smooth)


def expected_improvement(mean: Any, std: Any, incumbent: float) -> Any:
    """Expected improvement below the incumbent of a normal with this mean and std, in
    closed form, for CasADi expressions."""
    improvement = incumbent - mean
    gamma = improvement / (std + _EI_STD_EPSILON)
    cdf = 0.5 * (1 + cas.erf(gamma / np.sqrt(2)))
    pdf = cas.exp(-0.5 * gamma**2) / np.sqrt(2 * np.pi)
    return improvement * cdf + std * pdf


def standard_acquisition(
    problem: Problem,
    fitted: FittedGP,
    incumbent: float,
) -> NLP:
    """The acquisition problem of standard BO, over the decision variables within their
    bounds and started within their start bounds: `solve` gives the next trial point.

    The GP models u -> f directly, so EI is exact, from its mean and std.
    """
    u_vars = problem.system.u
    u = Symbolic.sym("u", len(u_vars))
    mean, std = fitted.predict(u.T)
    objective = -expected_improvement(mean, std, incumbent)
    return NLP(
        cas.Function("acquisition", [u], [objective]),
        np.array([var.lower for var in u_vars]),
        np.array([var.upper for var in u_vars]),
        start_bounds=start_bounds(u_vars),
    )
