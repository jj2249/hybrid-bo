"""Acquisition functions and the acquisition problems built from them.

`EI` and `LCB` name the acquisition function; each method computes it its own way.
Standard BO computes it analytically from the GP's mean and standard deviation
(`standard_acquisition`).
"""

from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.gp import GP, FittedGP, predict
from hybrid_bo.problem import Problem, Symbolic
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.solvers import NLP

# Added to the standard deviation in EI, so a zero std doesn't divide by zero
_EPSILON = 1e-10


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
    gamma = improvement / (std + _EPSILON)
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
