import numpy as np
import scipy


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
