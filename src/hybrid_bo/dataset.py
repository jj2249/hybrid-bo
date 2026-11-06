"""A set of evaluated points: what a surrogate is fitted to."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """The decision variables, states and objective of a set of points, one row each, in
    the order the points were taken, which `incumbents` relies on.

    `x` and `y` are measured (the measured variables) and reconstructed (the rest).
    """

    u: np.ndarray  # (n, N)
    x: np.ndarray  # (n, M)
    y: np.ndarray  # (n, P)
    f: np.ndarray  # (n,)


def extend(dataset: Dataset, new: Dataset) -> Dataset:
    """A new dataset: the rows of `dataset` followed by those of `new`. A point is a
    one-row dataset, so this appends too.

    Raises numpy's `ValueError` if a field's columns don't match between the two.
    """
    return Dataset(
        u=np.concatenate([dataset.u, new.u]),
        x=np.concatenate([dataset.x, new.x]),
        y=np.concatenate([dataset.y, new.y]),
        f=np.concatenate([dataset.f, new.f]),
    )


def incumbents(dataset: Dataset) -> np.ndarray:
    """The best measured objective value after each point, shape (n,)."""
    return np.minimum.accumulate(dataset.f)


def save_dataset(dataset: Dataset, path: str | Path) -> None:
    """Writes `dataset` to a new `.npz` file `path`.

    Raises `FileExistsError` if `path` exists.
    """
    with Path(path).open("xb") as file:
        np.savez(file, u=dataset.u, x=dataset.x, y=dataset.y, f=dataset.f)


def load_dataset(path: str | Path) -> Dataset:
    """The dataset `save_dataset` wrote to `path`, ready to fit or to `extend`."""
    with np.load(path) as saved:
        return Dataset(saved["u"], saved["x"], saved["y"], saved["f"])
