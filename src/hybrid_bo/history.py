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


def incumbents(history: History) -> np.ndarray:
    """The best measured objective value after each evaluation, shape (n,)."""
    return np.minimum.accumulate(history.f)
