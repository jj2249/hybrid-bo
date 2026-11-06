"""Numerical tools that know nothing of systems or GPs: the symbol type, drawing points
from a box, affine scaling, and a stable log softplus."""

from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

# The symbol type of all library math: `cas.chol` has no MX overload, so SX only
Symbolic = cas.SX

# Below this t, log log(1 + exp(t)) is t within about 1e-13
LOG_SOFTPLUS_LINEAR_BELOW = -30.0


def lhs(
    lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube scaled to the box [lower, upper], shape (n, d).

    `lower` and `upper` have shape (d,) and must be finite. The draw is scipy's
    `qmc.LatinHypercube(d, rng=rng).random(n)`, made here because scipy is
    only a dev dependency: from the next spawned child of `rng`, which
    leaves its stream untouched, a point within each of the n strata, then a shuffle of
    the strata of each dimension.
    """
    d = len(lower)
    child = rng.spawn(1)[0]
    within_stratum = child.uniform(size=(n, d))
    strata = np.tile(np.arange(1, n + 1), (d, 1))
    for strata_of_dimension in strata:
        child.shuffle(strata_of_dimension)
    samples = (strata.T - within_stratum) / n
    return lower + (upper - lower) * samples


def uniform(
    lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points drawn uniformly from the box [lower, upper], shape (n, d).

    `lower` and `upper` have shape (d,) and must be finite.
    """
    samples = rng.uniform(size=(n, len(lower)))
    return lower + (upper - lower) * samples


@dataclass(frozen=True)
class Scaler:
    """An affine map per column: scaled = slope * value + intercept."""

    slope: np.ndarray  # (d,)
    intercept: np.ndarray  # (d,)

    @classmethod
    def minmax(cls, values: np.ndarray) -> "Scaler":
        """The scaler that maps each column of `values` (n, d) onto [0, 1].

        A constant column is shifted to 0 instead.
        """
        low, high = values.min(0), values.max(0)
        span = np.where(low == high, 1.0, high - low)
        return cls(1 / span, -low / span)

    @classmethod
    def standard(cls, values: np.ndarray) -> "Scaler":
        """The scaler that gives each column of `values` (n, d) zero mean and unit standard
        deviation (`ddof=0`). A constant column is only shifted."""
        mean, std = values.mean(0), values.std(0)
        std = np.where(std == 0.0, 1.0, std)
        return cls(1 / std, -mean / std)

    def scale(self, values: Any) -> Any:
        """Scales the rows of `values` (n, d), numeric or CasADi."""
        n = values.shape[0]
        return values * np.tile(self.slope, (n, 1)) + np.tile(self.intercept, (n, 1))

    def unscale(self, scaled: Any) -> Any:
        """Undoes `scale` on the rows of `scaled` (n, d), numeric or CasADi."""
        n = scaled.shape[0]
        return (scaled - np.tile(self.intercept, (n, 1))) / np.tile(self.slope, (n, 1))


def log_softplus(t: Any) -> Any:
    """log log(1 + exp(t)), elementwise for CasADi expressions, finite with a
    finite gradient for any finite t.

    Far below zero, log(1 + exp(t)) underflows, so its log is taken as t; above
    zero, it is t + log(1 + exp(-t)), so exp doesn't overflow.
    """
    below = cas.log(cas.log1p(cas.exp(t)))
    above = cas.log(t + cas.log1p(cas.exp(-t)))
    exact = cas.if_else(t > 0, above, below)
    return cas.if_else(t < LOG_SOFTPLUS_LINEAR_BELOW, t, exact)
