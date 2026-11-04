from collections.abc import Sequence

import numpy as np

from hybrid_bo.problem import Var


def lhs(
    lower: np.ndarray, upper: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube scaled to the box [lower, upper], shape (n, d).

    `lower` and `upper` have shape (d,) and must be finite. The draw is scipy's
    `qmc.LatinHypercube(d, rng=rng).random(n)`, made here because importing
    `scipy.stats` takes about a second: from the next spawned child of `rng`, which
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


def _start_bounds(variables: Sequence[Var]) -> tuple[np.ndarray, np.ndarray]:
    """The lower and upper start bounds of the variables, each of shape (d,)."""
    return (
        np.array([var.start[0] for var in variables]),
        np.array([var.start[1] for var in variables]),
    )


def initial_points(
    variables: Sequence[Var], n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube over the variables' start bounds, shape (n, d)."""
    return lhs(*_start_bounds(variables), n, rng)


def _starting_points(
    variables: Sequence[Var],
    n_initial: int | None,
    points: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """The points a run starts from: `points`, or `n_initial` drawn with
    `initial_points`.

    Raises `ValueError` unless exactly one of them is given.
    """
    if (n_initial is None) == (points is None):
        msg = "Give exactly one of n_initial and initial_points"
        raise ValueError(msg)
    if points is not None:
        return np.asarray(points, dtype=float)
    assert n_initial is not None
    return initial_points(variables, n_initial, rng)
