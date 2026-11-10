"""Objectives of the hybrid NLP: what `solve` minimizes, as a function of f.

`saa(f, w)` takes f under each scenario (S, 1) and the weights w (S,), for `hb.SAA`;
`exact(mean, std)` takes f's normal mean and std, for `hb.Exact`. Both return a CasADi
scalar to minimize.
"""

from dataclasses import KW_ONLY, dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.utils import Symbolic, log_softplus

# Added to the std in exact EI, so a zero std doesn't divide by zero
_EI_STD_EPSILON = 1e-10


@dataclass(frozen=True)
class Mean:
    """The expected f."""

    def saa(self, f: Any, w: np.ndarray) -> Any:
        """The weighted average of f."""
        return Symbolic(w).T @ f

    def exact(self, mean: Any, std: Any) -> Any:
        """The mean of f."""
        return mean


@dataclass(frozen=True)
class EI:
    """-EI: expected improvement of f below the incumbent, negated."""

    incumbent: float

    def saa(self, f: Any, w: np.ndarray) -> Any:
        """-SAA-EI: the weighted average of min(f - incumbent, 0)."""
        return Symbolic(w).T @ cas.fmin(f - self.incumbent, 0)

    def exact(self, mean: Any, std: Any) -> Any:
        """-EI of a normal f, in closed form."""
        improvement = self.incumbent - mean
        gamma = improvement / (std + _EI_STD_EPSILON)
        cdf = 0.5 * (1 + cas.erf(gamma / np.sqrt(2)))
        pdf = cas.exp(-0.5 * gamma**2) / np.sqrt(2 * np.pi)
        return -(improvement * cdf + std * pdf)


@dataclass(frozen=True)
class SoftEI:
    """-log SoftEI: EI with max(a, 0) smoothed to tau log(1 + exp(a / tau)), tau the
    temperature in units of f. Has no exact form.

    Raises `ValueError` unless the temperature is positive and finite.
    """

    incumbent: float
    _: KW_ONLY
    temperature: float

    def __post_init__(self) -> None:
        """Raises `ValueError` unless the temperature is positive and finite."""
        if not (np.isfinite(self.temperature) and self.temperature > 0):
            msg = f"SoftEI temperature must be finite and > 0, got {self.temperature}"
            raise ValueError(msg)

    def saa(self, f: Any, w: np.ndarray) -> Any:
        """-log of the weighted average of tau log(1 + exp((incumbent - f) / tau))."""
        tau = self.temperature
        log_smooth = np.log(tau) + log_softplus((self.incumbent - f) / tau)
        return -cas.logsumexp(Symbolic(np.log(w)) + log_smooth)

    def exact(self, mean: Any, std: Any) -> Any:
        """Raises `ValueError`: SoftEI has no exact form."""
        msg = "SoftEI has no exact form: solve it with hb.SAA"
        raise ValueError(msg)
