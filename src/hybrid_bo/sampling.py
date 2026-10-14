from collections.abc import Sequence

import numpy as np
import scipy

from hybrid_bo.problem import Var


def lhs(
    lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube scaled to the box [lower, upper], shape (n, d).

    `lower` and `upper` have shape (d,) and must be finite.
    """
    samples = scipy.stats.qmc.LatinHypercube(len(lower), rng=rng).random(n)
    return lower + (upper - lower) * samples


def uniform(
    lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points drawn uniformly from the box [lower, upper], shape (n, d).

    `lower` and `upper` have shape (d,) and must be finite.
    """
    samples = rng.uniform(size=(n, len(lower)))
    return lower + (upper - lower) * samples


def initial_points(
    variables: Sequence[Var], n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube over the variables' start bounds, shape (n, d)."""
    lower = np.array([var.start[0] for var in variables])
    upper = np.array([var.start[1] for var in variables])
    return lhs(lower, upper, n, rng)
