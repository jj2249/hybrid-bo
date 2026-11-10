"""A set of evaluated points: what a GP is fitted to."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as cas
import numpy as np

from hybrid_bo.errors import RootFindingError
from hybrid_bo.solvers import RootProblem
from hybrid_bo.utils import Symbolic
from hybrid_bo.variables import Var, lhs_points

if TYPE_CHECKING:
    from hybrid_bo.hybrid_problem import HybridNLP


@dataclass(frozen=True)
class Dataset:
    """The decision variables u (n, N), states x (n, M) and y (n, P), and objective f
    (n,) of n points in the order taken, and the `names` of the variables of u, x and y.

    `data[var]` is that variable's values (n,).
    """

    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def from_measurements(
        cls,
        nlp: "HybridNLP",
        u: np.ndarray,
        y: np.ndarray,
        *,
        rng: np.random.Generator,
        n_starts: int = 100,
        tol: float = 1e-10,
    ) -> "Dataset":
        """The dataset of u and the measured y, a point (N,), (P,) or a set (n, N),
        (n, P): x solved from g within its bounds from `n_starts` starts, and f.

        Raises `ValueError` for a wrong shape, and `RootFindingError` naming every row
        with no root or a value that isn't finite, with the other rows as
        `error.dataset`.
        """
        u, y = np.atleast_2d(u).astype(float), np.atleast_2d(y).astype(float)
        if u.shape[1:] != (len(nlp.u),) or y.shape != (len(u), len(nlp.y)):
            msg = (
                f"Measurements need u (n, {len(nlp.u)}) and y (n, {len(nlp.y)}), got "
                f"{u.shape} and {y.shape}"
            )
            raise ValueError(msg)

        # The root-finding problem: g for x, with u and y as parameters
        x_symbol = Symbolic.sym("x", len(nlp.x))
        u_symbol = Symbolic.sym("u", len(nlp.u))
        y_symbol = Symbolic.sym("y", len(nlp.y))
        root_problem = RootProblem(
            cas.Function(
                "known_equations",
                [x_symbol, cas.vertcat(u_symbol, y_symbol)],
                [nlp.g(x_symbol, y_symbol, u_symbol)],
            ),
            np.array([var.lower for var in nlp.x]),
            np.array([var.upper for var in nlp.x]),
        )

        # Solve it at each row, noting the rows that fail
        x, failed = [], []
        for row, p in enumerate(np.concatenate([u, y], axis=1)):
            starts = lhs_points(nlp.x, n_starts, rng) if nlp.x else np.zeros((0, 0))
            try:
                if not np.isfinite(p).all():
                    msg = f"Row {row} holds a value that isn't finite"
                    raise RootFindingError(msg)
                x.append(root_problem.solve(p, starts, tol))
            except RootFindingError:
                failed.append(row)

        u, y = np.delete(u, failed, axis=0), np.delete(y, failed, axis=0)
        x = np.reshape(x, (len(u), len(nlp.x)))
        f = np.array([nlp.f(*point).full().item() for point in zip(x, y, u)])
        names = tuple(var.name for var in (*nlp.u, *nlp.x, *nlp.y))
        dataset = cls(u, x, y, f, names)
        if failed:
            error = RootFindingError(
                f"No root, or a value that isn't finite, at rows {failed}"
            )
            error.dataset = dataset
            raise error
        return dataset

    def __getitem__(self, var: Var) -> np.ndarray:
        """The values of `var` (n,). Raises `KeyError` if it isn't in the dataset."""
        if var.name not in self.names:
            msg = f"{var.name!r} is not in the dataset; it has {', '.join(self.names)}"
            raise KeyError(msg)
        return np.column_stack([self.u, self.x, self.y])[:, self.names.index(var.name)]

    def extend(self, new: "Dataset") -> "Dataset":
        """The rows of this dataset followed by those of `new`.

        Raises `ValueError` if the two have different variables.
        """
        if new.names != self.names:
            msg = f"Can't extend a dataset of {self.names} with one of {new.names}"
            raise ValueError(msg)
        return Dataset(
            np.concatenate([self.u, new.u]),
            np.concatenate([self.x, new.x]),
            np.concatenate([self.y, new.y]),
            np.concatenate([self.f, new.f]),
            self.names,
        )

    @property
    def incumbents(self) -> np.ndarray:
        """The best f after each point, (n,)."""
        return np.minimum.accumulate(self.f)

    def save(self, path: str | Path) -> None:
        """Writes the dataset to a new `.npz` file. Raises `FileExistsError` if `path`
        exists."""
        with Path(path).open("xb") as file:
            np.savez(
                file, u=self.u, x=self.x, y=self.y, f=self.f, names=np.array(self.names)
            )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """The dataset `save` wrote to `path`."""
        with np.load(path) as saved:
            return cls(
                saved["u"],
                saved["x"],
                saved["y"],
                saved["f"],
                tuple(str(name) for name in saved["names"]),
            )
