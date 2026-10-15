"""The evaluated points of one run."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class History:
    """Every evaluated point of one run, in order; the first `n_initial` are the
    initial points.

    `x`, `y` and `f` are measured (the measured variables) and reconstructed (the
    rest). The oracle values are for benchmarking and are None when the experiment
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


def incumbents(history: History) -> np.ndarray:
    """The best measured objective value after each evaluation, shape (n,)."""
    return np.minimum.accumulate(history.f)


def oracle_incumbents(history: History) -> np.ndarray:
    """The oracle objective value at the incumbent's point after each evaluation,
    shape (n): the benchmark metric.

    The incumbent's point is the one the optimizer believes is best, so this is
    what acting on the run would give. Under noise it can be worse than the best
    oracle value so far. A later point that only ties the incumbent doesn't replace
    it. Raises `ValueError` when the history has no oracle values.
    """
    if history.f_oracle is None:
        msg = "the benchmark metric needs oracle values; the experiment had no oracle"
        raise ValueError(msg)

    # A point becomes the incumbent's point when it beats every earlier one
    best_before = np.minimum.accumulate(np.concatenate([[np.inf], history.f[:-1]]))
    improves = history.f < best_before
    incumbent_index = np.maximum.accumulate(
        np.where(improves, np.arange(len(history.f)), 0)
    )
    return history.f_oracle[incumbent_index]
