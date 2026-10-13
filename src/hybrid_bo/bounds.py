from dataclasses import dataclass

import numpy as np
import scipy
from numpy.typing import ArrayLike


@dataclass(frozen=True, init=False)
class Bounds:
    """Box bounds `lower <= v <= upper` for a vector `v` of length n.

    Accepts anything array-like; both arrays are stored as read-only 1-D float copies
    of shape (n,).
    """

    lower: np.ndarray
    upper: np.ndarray

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        if lower.ndim != 1 or upper.ndim != 1:
            msg = f"Bounds must be 1-D, got shapes {lower.shape} and {upper.shape}"
            raise ValueError(msg)
        if lower.shape != upper.shape:
            msg = f"lower and upper have different shapes: {lower.shape} and {upper.shape}"
            raise ValueError(msg)
        if not (lower <= upper).all():
            msg = f"lower must be <= upper, got {lower} and {upper}"
            raise ValueError(msg)

        lower.flags.writeable = False
        upper.flags.writeable = False
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def unbounded(cls, n: int) -> "Bounds":
        return cls(np.full(n, -np.inf), np.full(n, np.inf))


def lhs(bounds: Bounds, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points from a Latin hypercube scaled to `bounds`, shape (n, d)."""
    samples = scipy.stats.qmc.LatinHypercube(len(bounds.lower), rng=rng).random(n)
    return bounds.lower + (bounds.upper - bounds.lower) * samples


def uniform(bounds: Bounds, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points drawn uniformly from `bounds`, shape (n, d)."""
    samples = rng.uniform(size=(n, len(bounds.lower)))
    return bounds.lower + (bounds.upper - bounds.lower) * samples
