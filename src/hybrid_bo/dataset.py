"""A set of evaluated points: what a GP is fitted to."""

from collections.abc import Sequence
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
    """The decision variables u (n, N), configuration c (n, K), states x (n, M) and y
    (n, P), and objective f (n,) of n points in the order taken, and the `names` of the
    variables of u, c, x and y.

    `data[var]` is that variable's values (n,).
    """

    u: np.ndarray
    c: np.ndarray
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def from_measurements(
        cls,
        nlp: "HybridNLP",
        u: np.ndarray,
        measured: dict[Var, np.ndarray],
        *,
        c: np.ndarray | None = None,
        rng: np.random.Generator,
        n_starts: int = 100,
        tol: float = 1e-10,
    ) -> "Dataset":
        """The dataset of u, a point (N,) or a set (n, N), at the configuration c, (K,)
        or (n, K) and None when there is none, and `measured`, the values of P of the
        states x and y, each a scalar or (n,): the other states solved from g within
        their bounds from `n_starts` starts, and f.

        Raises `ValueError` for a wrong shape, a c outside its bounds or a measured set
        that isn't P states, and `RootFindingError` naming every row with no root or a
        value that isn't finite, with the other rows as `error.dataset`.
        """
        u = np.atleast_2d(u).astype(float)
        measured = {
            var: np.atleast_1d(values).astype(float) for var, values in measured.items()
        }
        states = (*nlp.x, *nlp.y)
        if not set(measured) <= set(states) or len(measured) != len(nlp.y):
            msg = (
                f"Measure exactly {len(nlp.y)} of the states "
                f"{[var.name for var in states]}, as many as unknown states, got "
                f"{[var.name for var in measured]}"
            )
            raise ValueError(msg)
        shapes = [values.shape for values in measured.values()]
        if u.shape[1:] != (len(nlp.u),) or any(s != (len(u),) for s in shapes):
            msg = (
                f"Measurements need u (n, {len(nlp.u)}) and each measured state (n,), "
                f"got {u.shape} and {shapes}"
            )
            raise ValueError(msg)
        c = nlp.configuration(c, len(u))

        # The root-finding problem: g for the states not measured, with u, c and the
        # measured states as parameters
        unsolved = [var for var in states if var not in measured]
        symbols = {var: Symbolic.sym(var.name) for var in (*nlp.u, *nlp.c, *states)}

        def column(variables: Sequence[Var]) -> Symbolic:
            """The symbols of `variables`, stacked."""
            return cas.vertcat(*(symbols[var] for var in variables))

        root_problem = RootProblem(
            cas.Function(
                "known_equations",
                [column(unsolved), column([*nlp.u, *nlp.c, *measured])],
                [nlp.g(column(nlp.x), column(nlp.y), column(nlp.u), column(nlp.c))],
            ),
            np.array([var.lower for var in unsolved]),
            np.array([var.upper for var in unsolved]),
        )

        # Solve it at each row, noting the rows that fail
        parameters = np.column_stack([u, c, *measured.values()])
        solved, failed = [], []
        for row, p in enumerate(parameters):
            starts = (
                lhs_points(unsolved, n_starts, rng) if unsolved else np.zeros((0, 0))
            )
            try:
                if not np.isfinite(p).all():
                    msg = f"Row {row} holds a value that isn't finite"
                    raise RootFindingError(msg)
                solved.append(root_problem.solve(p, starts, tol))
            except RootFindingError:
                failed.append(row)

        # Every state's values, measured or solved, in the order of x and y
        parameters = np.delete(parameters, failed, axis=0)
        n, n_u, n_c = len(parameters), len(nlp.u), len(nlp.c)
        u, c = parameters[:, :n_u], parameters[:, n_u : n_u + n_c]
        columns = dict(zip(measured, parameters[:, n_u + n_c :].T, strict=True))
        columns |= dict(
            zip(unsolved, np.reshape(solved, (n, len(unsolved))).T, strict=True)
        )
        x, y = (
            np.reshape([columns[var] for var in block], (len(block), n)).T
            for block in (nlp.x, nlp.y)
        )
        f = np.array([nlp.f(*point).full().item() for point in zip(x, y, u, c)])
        names = tuple(var.name for var in (*nlp.u, *nlp.c, *nlp.x, *nlp.y))
        dataset = cls(u, c, x, y, f, names)
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
        columns = np.column_stack([self.u, self.c, self.x, self.y])
        return columns[:, self.names.index(var.name)]

    def extend(self, new: "Dataset") -> "Dataset":
        """The rows of this dataset followed by those of `new`.

        Raises `ValueError` if the two have different variables.
        """
        if new.names != self.names:
            msg = f"Can't extend a dataset of {self.names} with one of {new.names}"
            raise ValueError(msg)
        return Dataset(
            np.concatenate([self.u, new.u]),
            np.concatenate([self.c, new.c]),
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
                file,
                u=self.u,
                c=self.c,
                x=self.x,
                y=self.y,
                f=self.f,
                names=np.array(self.names),
            )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """The dataset `save` wrote to `path`, with no configuration if it has none
        (v0.1.0)."""
        with np.load(path) as saved:
            return cls(
                saved["u"],
                saved["c"] if "c" in saved else np.zeros((len(saved["u"]), 0)),
                saved["x"],
                saved["y"],
                saved["f"],
                tuple(str(name) for name in saved["names"]),
            )
