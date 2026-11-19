"""Named variables, points drawn over them, and equations over them traced by CasADi."""

import keyword
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.utils import Symbolic, lhs


@dataclass(frozen=True, init=False)
class Var:
    """A named scalar variable: its bounds, and the finite box `start` that starting
    points are drawn from, by default the bounds if they are finite, else None.

    Raises `ValueError` for a name that isn't an identifier, or lower > upper.
    """

    name: str
    lower: float
    upper: float
    start: tuple[float, float] | None

    def __init__(
        self,
        name: str,
        lower: float = -np.inf,
        upper: float = np.inf,
        start: tuple[float, float] | None = None,
    ) -> None:
        # g and f read the variable as v.<name>
        if not (
            name.isidentifier()
            and not keyword.iskeyword(name)
            and not name.startswith("_")
        ):
            msg = f"Var name {name!r} must be an identifier not starting with '_'"
            raise ValueError(msg)
        if not lower <= upper:
            msg = f"Var {name!r}: lower ({lower}) must be <= upper ({upper})"
            raise ValueError(msg)
        if start is None and np.isfinite(lower) and np.isfinite(upper):
            start = (lower, upper)

        # Frozen dataclass: set fields through object.__setattr__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lower", float(lower))
        object.__setattr__(self, "upper", float(upper))
        object.__setattr__(
            self, "start", None if start is None else (float(start[0]), float(start[1]))
        )


def start_bounds(variables: Sequence[Var]) -> tuple[np.ndarray, np.ndarray]:
    """The lower and upper start bounds of the variables, each (d,).

    Raises `ValueError` for a variable with no start.
    """
    starts = [var.start for var in variables]
    if None in starts:
        names = [var.name for var in variables if var.start is None]
        msg = f"{names} need a start, or finite bounds, to draw starting points over"
        raise ValueError(msg)
    lower, upper = np.array(starts, dtype=float).reshape(-1, 2).T
    return lower, upper


def lhs_points(
    variables: Sequence[Var], n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube over the variables' start bounds, (n, d)."""
    return lhs(*start_bounds(variables), n, rng)


class _Variables:
    """The `v` passed to the user's `g` and `f`: `v.<name>` is that variable's symbol."""

    def __init__(self, symbols: dict[str, Any]) -> None:
        object.__setattr__(self, "_symbols", symbols)

    def __getattr__(self, name: str) -> Any:
        """The symbol of the variable `name`; raises `AttributeError` listing the names."""
        symbols = object.__getattribute__(self, "_symbols")
        try:
            return symbols[name]
        except KeyError:
            msg = f"Unknown variable {name!r}; available: {', '.join(symbols)}"
            raise AttributeError(msg) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Raises `AttributeError`: the variables are read-only."""
        msg = "The variables passed to g and f are read-only"
        raise AttributeError(msg)


def to_casadi_function(
    name: str,
    equations: Callable[[Any], Any],
    x: Sequence[Var],
    y: Sequence[Var],
    u: Sequence[Var],
    c: Sequence[Var],
) -> cas.Function:
    """`equations` (`g` or `f`), called once on symbols, as
    `name(x, y, u, c) -> column`.

    Raises `TypeError` if `equations` doesn't return CasADi expressions.
    """
    symbols = {var.name: Symbolic.sym(var.name) for var in (*x, *y, *u, *c)}
    expressions = equations(_Variables(symbols))
    parts = expressions if isinstance(expressions, list | tuple) else [expressions]
    try:
        column = cas.vertcat(*parts)
    except NotImplementedError:  # CasADi's error for arguments it can't stack
        msg = f"{name} must return CasADi expressions built from v; got {expressions!r}"
        raise TypeError(msg) from None
    inputs = [
        cas.vertcat(*(symbols[var.name] for var in block)) for block in (x, y, u, c)
    ]
    return cas.Function(name, inputs, [column], ["x", "y", "u", "c"], [name])
