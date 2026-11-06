"""The data of a set of points: what a surrogate is fitted to and the plots draw."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """The decision variables, states and objective of a set of points, one row each:
    what a surrogate is fitted to and what the plots draw.

    Rows are in the order the points were taken, which `incumbents` relies on. The
    first `n_initial` rows are the initial points.

    `x`, `y` and `f` are measured (the measured variables) and reconstructed (the
    rest): only what a real system gives.
    """

    u: np.ndarray  # (n, N)
    x: np.ndarray  # (n, M)
    y: np.ndarray  # (n, P)
    f: np.ndarray  # (n,)
    n_initial: int


def extend(dataset: Dataset, new: Dataset) -> Dataset:
    """A new dataset: the rows of `dataset` followed by those of `new`, with the initial
    points of `dataset`. A point is a one-row dataset, so this appends too.

    Raises numpy's `ValueError` if a field's columns don't match between the two.
    """
    return Dataset(
        u=np.concatenate([dataset.u, new.u]),
        x=np.concatenate([dataset.x, new.x]),
        y=np.concatenate([dataset.y, new.y]),
        f=np.concatenate([dataset.f, new.f]),
        n_initial=dataset.n_initial,
    )


def n_iterations(dataset: Dataset) -> int:
    """The number of points evaluated after the initial points: the iterations, or
    the design points for the baselines."""
    return len(dataset.u) - dataset.n_initial


def incumbents(dataset: Dataset) -> np.ndarray:
    """The best measured objective value after each evaluation, shape (n,)."""
    return np.minimum.accumulate(dataset.f)
