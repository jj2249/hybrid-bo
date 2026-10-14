"""The problem: a system described in the paper's notation, plus an objective.

Variables are named `Var`s. The known equations `g` and the objective `f` are callables
over those variables by name (`lambda v: [v.x + cas.exp(v.x) - v.y]`); each is called
once with CasADi symbols and turned into a `cas.Function` (ADR 0002).
"""

import keyword
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

# The one place the symbol type is chosen. Library math only uses CasADi operations,
# so it works for SX and MX alike.
Symbolic = cas.SX


@dataclass(frozen=True, init=False)
class Var:
    """A named scalar variable with bounds and finite start bounds.

    `start` is the box from which starting points are drawn (initial points, multistart
    starts, root-finding guesses). It defaults to the bounds when both are finite.
    """

    name: str
    lower: float
    upper: float
    start: tuple[float, float]

    def __init__(
        self,
        name: str,
        lower: float = -np.inf,
        upper: float = np.inf,
        start: tuple[float, float] | None = None,
    ) -> None:
        if not name.isidentifier() or keyword.iskeyword(name):
            msg = f"Var name {name!r} must be a Python identifier (it is used as v.{name})"
            raise ValueError(msg)
        lower, upper = float(lower), float(upper)
        if not lower <= upper:
            msg = f"Var {name!r}: lower ({lower}) must be <= upper ({upper})"
            raise ValueError(msg)
        if start is None:
            if not (np.isfinite(lower) and np.isfinite(upper)):
                msg = f"Var {name!r} has an infinite bound, so it needs start=(lower, upper)"
                raise ValueError(msg)
            start = (lower, upper)
        start_lower, start_upper = float(start[0]), float(start[1])
        if not (np.isfinite(start_lower) and np.isfinite(start_upper)):
            msg = f"Var {name!r}: start {start} must be finite"
            raise ValueError(msg)
        if not lower <= start_lower <= start_upper <= upper:
            msg = f"Var {name!r}: start {start} must be ordered and inside [{lower}, {upper}]"
            raise ValueError(msg)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "start", (start_lower, start_upper))


class _Variables:
    """Read-only access to one symbol per variable by name: `v.T`, `v.x1`."""

    def __init__(self, symbols: dict[str, Any]) -> None:
        object.__setattr__(self, "_symbols", symbols)

    def __getattr__(self, name: str) -> Any:
        symbols = object.__getattribute__(self, "_symbols")
        try:
            return symbols[name]
        except KeyError:
            msg = f"Unknown variable {name!r}; available: {', '.join(symbols)}"
            raise AttributeError(msg) from None

    def __setattr__(self, name: str, value: Any) -> None:
        msg = "The variables passed to g and f are read-only"
        raise AttributeError(msg)


def _trace(
    name: str,
    expression: Callable[[Any], Any],
    x: Sequence[Var],
    y: Sequence[Var],
    u: Sequence[Var],
) -> cas.Function:
    """Calls `expression` with fresh symbols and returns it as `name(x, y, u)`."""
    symbols = {var.name: Symbolic.sym(var.name) for var in (*x, *y, *u)}
    result = expression(_Variables(symbols))
    parts = result if isinstance(result, list | tuple) else [result]
    outputs = cas.vertcat(*parts)
    inputs = [cas.vertcat(*(symbols[var.name] for var in block)) for block in (x, y, u)]
    return cas.Function(name, inputs, [outputs], ["x", "y", "u"], [name])


def _check_drawn_from(
    chosen: Sequence[Var], allowed: Sequence[Var], label: str, where: str
) -> None:
    for var in chosen:
        if var not in allowed:
            msg = f"{label} contains {var.name!r}, which is not one of {where}"
            raise ValueError(msg)
    if len(set(chosen)) != len(chosen):
        msg = f"{label} contains a variable twice"
        raise ValueError(msg)


@dataclass(frozen=True, init=False)
class System:
    """The physical process (paper, Problem 4).

    - u: the N decision variables;
    - x: the M mechanistic states, determined by the known equations 0 = g(x, y, u);
    - y: the P unknown states, determined by the unknown equations y = h(x, u),
      whose form is not part of the system (ADR 0001);
    - h_inputs: the arguments of h, drawn from u and x;
    - measured: the states the sensors read, drawn from x and y. There are P of them,
      so the root-finding problem for the other states is square.

    `g` is a `cas.Function` with column-vector inputs `(x, y, u)` and output `(M, 1)`.
    """

    u: tuple[Var, ...]
    x: tuple[Var, ...]
    y: tuple[Var, ...]
    g: cas.Function
    h_inputs: tuple[Var, ...]
    measured: tuple[Var, ...]

    def __init__(
        self,
        u: Sequence[Var],
        x: Sequence[Var],
        y: Sequence[Var],
        g: Callable[[Any], Any],
        h_inputs: Sequence[Var],
        measured: Sequence[Var],
    ) -> None:
        u, x, y = tuple(u), tuple(x), tuple(y)
        for block, label in ((u, "u"), (x, "x"), (y, "y")):
            if not block:
                msg = f"The system needs at least one variable in {label}"
                raise ValueError(msg)
        names = [var.name for var in (*u, *x, *y)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"Variable names must be unique; repeated: {', '.join(duplicates)}"
            raise ValueError(msg)
        if len(y) > 1:
            msg = (
                f"Only one unknown state is supported for now (got {len(y)}: "
                f"{', '.join(var.name for var in y)})"
            )
            raise NotImplementedError(msg)

        h_inputs, measured = tuple(h_inputs), tuple(measured)
        _check_drawn_from(h_inputs, (*u, *x), "h_inputs", "u or x")
        _check_drawn_from(measured, (*x, *y), "measured", "x or y")
        if len(measured) != len(y):
            msg = (
                f"There must be as many measured variables as unknown states "
                f"({len(y)}), so the root-finding problem is square; got {len(measured)}"
            )
            raise ValueError(msg)

        g_function = _trace("g", g, x, y, u)
        if g_function.size_out(0) != (len(x), 1):
            msg = (
                f"g must give one equation per mechanistic state ({len(x)}), "
                f"got shape {g_function.size_out(0)}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "u", u)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "g", g_function)
        object.__setattr__(self, "h_inputs", h_inputs)
        object.__setattr__(self, "measured", measured)

    @property
    def n_u(self) -> int:
        return len(self.u)

    @property
    def n_x(self) -> int:
        return len(self.x)

    @property
    def n_y(self) -> int:
        return len(self.y)


@dataclass(frozen=True, init=False)
class Problem:
    """A system plus the objective to minimize.

    `f` is a `cas.Function` with column-vector inputs `(x, y, u)` and a scalar output.
    """

    system: System
    f: cas.Function

    def __init__(self, system: System, f: Callable[[Any], Any]) -> None:
        f_function = _trace("f", f, system.x, system.y, system.u)
        if f_function.size_out(0) != (1, 1):
            msg = f"f must be a scalar, got shape {f_function.size_out(0)}"
            raise ValueError(msg)

        object.__setattr__(self, "system", system)
        object.__setattr__(self, "f", f_function)
