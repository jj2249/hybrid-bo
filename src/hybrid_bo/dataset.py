"""Evaluated points, as a dataset of rows in evaluation order."""

import warnings
from dataclasses import dataclass

import numpy as np

from hybrid_bo.errors import OracleOptimumWarning


@dataclass(frozen=True)
class Dataset:
    """Evaluated points, a row each, in evaluation order: the row number is the index.
    The first `n_initial` rows are the initial points.

    `x`, `y` and `f` are measured (the measured variables) and reconstructed (the
    rest). The oracle values are for benchmarking and are None when the measurement
    has no oracle.
    """

    u: np.ndarray  # (n, N)
    x: np.ndarray  # (n, M)
    y: np.ndarray  # (n, P)
    f: np.ndarray  # (n,)
    x_oracle: np.ndarray | None  # (n, M)
    y_oracle: np.ndarray | None  # (n, P)
    f_oracle: np.ndarray | None  # (n,)
    n_initial: int


def extend(dataset: Dataset, new: Dataset) -> Dataset:
    """A new dataset: the rows of `dataset` followed by those of `new`, with the initial
    points of `dataset`. A point is a one-row dataset, so this appends too. The oracle
    values are kept only when both have them."""

    def join(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
        """The rows of both, or None if either is None."""
        if first is None or second is None:
            return None
        return np.concatenate([first, second])

    return Dataset(
        u=np.concatenate([dataset.u, new.u]),
        x=np.concatenate([dataset.x, new.x]),
        y=np.concatenate([dataset.y, new.y]),
        f=np.concatenate([dataset.f, new.f]),
        x_oracle=join(dataset.x_oracle, new.x_oracle),
        y_oracle=join(dataset.y_oracle, new.y_oracle),
        f_oracle=join(dataset.f_oracle, new.f_oracle),
        n_initial=dataset.n_initial,
    )


def n_iterations(dataset: Dataset) -> int:
    """The number of points evaluated after the initial points: the iterations, or
    the design points for the baselines."""
    return len(dataset.u) - dataset.n_initial


def incumbents(dataset: Dataset) -> np.ndarray:
    """The best measured objective value after each evaluation, shape (n,)."""
    return np.minimum.accumulate(dataset.f)


def oracle_incumbents(dataset: Dataset) -> np.ndarray:
    """The benchmark metric: the oracle objective at the incumbent's point after each
    evaluation, shape (n,).

    The incumbent's point is the one the optimizer believes is best, so this is
    what acting on the run would give. Under noise it can be worse than the best
    oracle value so far. A later point that only ties the incumbent doesn't replace
    it. Raises `ValueError` when the dataset has no oracle values.
    """
    if dataset.f_oracle is None:
        msg = "the benchmark metric needs oracle values; the measurement had no oracle"
        raise ValueError(msg)

    # A point becomes the incumbent's point when it beats every earlier one
    best_before = np.minimum.accumulate(np.concatenate([[np.inf], dataset.f[:-1]]))
    improves = dataset.f < best_before
    incumbent_index = np.maximum.accumulate(
        np.where(improves, np.arange(len(dataset.f)), 0)
    )
    return dataset.f_oracle[incumbent_index]


def check_oracle_optimum(dataset: Dataset, optimum: float) -> bool:
    """Warns with `OracleOptimumWarning` if a point of `dataset` has an oracle objective
    below `optimum`, beyond the solvers' tolerance of a relative 1e-6, and returns
    whether it did. Without an oracle there is nothing to compare, so it doesn't."""
    if dataset.f_oracle is None or len(dataset.f_oracle) == 0:
        return False
    lowest = float(np.min(dataset.f_oracle))
    beats_optimum = lowest < optimum - 1e-6 * max(1.0, abs(optimum))
    if beats_optimum:
        msg = (
            f"An evaluated point has oracle objective {lowest!r}, below the stated "
            f"oracle optimum {optimum!r}, so the regret is wrong"
        )
        warnings.warn(msg, OracleOptimumWarning, stacklevel=2)
    return beats_optimum
