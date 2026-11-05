"""The evaluated points of one run."""

import warnings
from dataclasses import dataclass

import numpy as np

from hybrid_bo.errors import OracleOptimumWarning


@dataclass(frozen=True)
class History:
    """Every evaluated point of one run, in order; the first `n_initial` are the
    initial points.

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


def _concatenate(history: History, later: History) -> History:
    """`history` followed by the points of `later`, with the initial points of
    `history`. The oracle values are kept only when both have them."""

    def join(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
        """The rows of both, or None if either is None."""
        if first is None or second is None:
            return None
        return np.concatenate([first, second])

    return History(
        u=np.concatenate([history.u, later.u]),
        x=np.concatenate([history.x, later.x]),
        y=np.concatenate([history.y, later.y]),
        f=np.concatenate([history.f, later.f]),
        x_oracle=join(history.x_oracle, later.x_oracle),
        y_oracle=join(history.y_oracle, later.y_oracle),
        f_oracle=join(history.f_oracle, later.f_oracle),
        n_initial=history.n_initial,
    )


def _first(history: History, n: int) -> History:
    """The first `n` points of `history`, at least its initial points: the history as
    it stood then."""

    def rows(values: np.ndarray | None) -> np.ndarray | None:
        """The first `n` rows, or None without oracle values."""
        return None if values is None else values[:n]

    return History(
        u=history.u[:n],
        x=history.x[:n],
        y=history.y[:n],
        f=history.f[:n],
        x_oracle=rows(history.x_oracle),
        y_oracle=rows(history.y_oracle),
        f_oracle=rows(history.f_oracle),
        n_initial=history.n_initial,
    )


def n_iterations(history: History) -> int:
    """The number of points evaluated after the initial points: the iterations, or
    the design points for the baselines."""
    return len(history.u) - history.n_initial


def incumbents(history: History) -> np.ndarray:
    """The best measured objective value after each evaluation, shape (n,)."""
    return np.minimum.accumulate(history.f)


def oracle_incumbents(history: History) -> np.ndarray:
    """The benchmark metric: the oracle objective at the incumbent's point after each
    evaluation, shape (n,).

    The incumbent's point is the one the optimizer believes is best, so this is
    what acting on the run would give. Under noise it can be worse than the best
    oracle value so far. A later point that only ties the incumbent doesn't replace
    it. Raises `ValueError` when the history has no oracle values.
    """
    if history.f_oracle is None:
        msg = "the benchmark metric needs oracle values; the measurement had no oracle"
        raise ValueError(msg)

    # A point becomes the incumbent's point when it beats every earlier one
    best_before = np.minimum.accumulate(np.concatenate([[np.inf], history.f[:-1]]))
    improves = history.f < best_before
    incumbent_index = np.maximum.accumulate(
        np.where(improves, np.arange(len(history.f)), 0)
    )
    return history.f_oracle[incumbent_index]


def check_oracle_optimum(history: History, optimum: float) -> bool:
    """Warns with `OracleOptimumWarning` if a point of `history` has an oracle objective
    below `optimum`, beyond the solvers' tolerance of a relative 1e-6, and returns
    whether it did. Without an oracle there is nothing to compare, so it doesn't."""
    if history.f_oracle is None or len(history.f_oracle) == 0:
        return False
    lowest = float(np.min(history.f_oracle))
    beats_optimum = lowest < optimum - 1e-6 * max(1.0, abs(optimum))
    if beats_optimum:
        msg = (
            f"An evaluated point has oracle objective {lowest!r}, below the stated "
            f"oracle optimum {optimum!r}, so the regret is wrong"
        )
        warnings.warn(msg, OracleOptimumWarning, stacklevel=2)
    return beats_optimum
