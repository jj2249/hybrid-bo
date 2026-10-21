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

# The symbol type of all library math: SX only (see DESIGN.md, Decisions).
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
        # Name: g and f read the variable as v.<name>, so it must be a valid
        # attribute name. Names starting with "_" would shadow _Variables internals.
        is_valid_name = (
            isinstance(name, str)
            and name.isidentifier()
            and not keyword.iskeyword(name)
            and not name.startswith("_")
        )
        if not is_valid_name:
            msg = (
                f"Var name {name!r} must be a Python identifier that doesn't start "
                f"with '_' (it is used as v.{name})"
            )
            raise ValueError(msg)

        # Bounds: may be infinite. The comparison is also False for NaN.
        lower, upper = float(lower), float(upper)
        if not lower <= upper:
            msg = f"Var {name!r}: lower ({lower}) must be <= upper ({upper})"
            raise ValueError(msg)

        # Start bounds: default to the bounds, which is only possible when both are
        # finite (points can't be drawn from an infinite box).
        if start is None:
            if not (np.isfinite(lower) and np.isfinite(upper)):
                msg = (
                    f"Var {name!r} has an infinite bound, so it needs start bounds: "
                    f"start=(lower, upper)"
                )
                raise ValueError(msg)
            start = (lower, upper)
        start_lower, start_upper = float(start[0]), float(start[1])
        if not (np.isfinite(start_lower) and np.isfinite(start_upper)):
            msg = f"Var {name!r}: start bounds {start} must be finite"
            raise ValueError(msg)
        if not lower <= start_lower <= start_upper <= upper:
            msg = (
                f"Var {name!r}: start bounds {start} must be ordered and inside "
                f"the bounds [{lower}, {upper}]"
            )
            raise ValueError(msg)

        # The dataclass is frozen, so fields are set through object.__setattr__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "start", (start_lower, start_upper))


class _Variables:
    """The `v` passed to the user's `g` and `f`: one CasADi symbol per variable.

    `v.T` returns the symbol for the variable named "T". An unknown name raises an
    `AttributeError` that lists the available names, and assigning to `v` is blocked.
    The symbols live in `_symbols`, which is why `Var` names can't start with `_`.
    """

    def __init__(self, symbols: dict[str, Any]) -> None:
        # Bypasses __setattr__ below, which blocks every assignment
        object.__setattr__(self, "_symbols", symbols)

    # Only called when normal lookup fails, i.e. for v.<variable name>
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
    """Turns a user callable (`g` or `f`) into a `cas.Function` named `name`.

    1. Creates one symbol per variable in x, y and u.
    2. Calls `expression(v)` once with those symbols.
    3. Stacks the result (one expression, or a list of them) into a column.
    4. Returns `name(x, y, u) -> column`, with x, y, u as column vectors in
       declaration order (the paper's argument order).

    Raises `TypeError` if the callable returns anything that isn't CasADi
    expressions, e.g. `None` or a nested list. Checking the output's size is left
    to the caller, which knows what size it expects.
    """
    symbols = {var.name: Symbolic.sym(var.name) for var in (*x, *y, *u)}
    result = expression(_Variables(symbols))

    # Accept one expression or a list of them; stack into a column either way
    parts = result if isinstance(result, list | tuple) else [result]
    try:
        outputs = cas.vertcat(*parts)
    except NotImplementedError:  # CasADi's error for arguments it can't stack
        msg = (
            f"{name} must return a CasADi expression or a list of scalar expressions "
            f"built from v; got {result!r}"
        )
        raise TypeError(msg) from None

    # One column-vector input per block, in declaration order
    inputs = [cas.vertcat(*(symbols[var.name] for var in block)) for block in (x, y, u)]
    return cas.Function(name, inputs, [outputs], ["x", "y", "u"], [name])


def _as_text(
    function: cas.Function, x: Sequence[Var], y: Sequence[Var], u: Sequence[Var]
) -> str:
    """The expression of a function built by `_trace`, as text in the variable names
    (e.g. `((x+exp(x))-y)`), or `none` for a function of no equations (g with M = 0).
    """
    if function.numel_out(0) == 0:
        return "none"
    symbols = [
        cas.vertcat(*(Symbolic.sym(var.name) for var in block)) for block in (x, y, u)
    ]
    return str(function(*symbols))


def _check_unique(names: Sequence[str], label: str) -> None:
    """Raises `ValueError` naming every name that appears more than once.

    `label` says which collection was checked, for the message.
    """
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        msg = f"{label} must not repeat a variable; repeated: {', '.join(repeated)}"
        raise ValueError(msg)


def _check_drawn_from(
    chosen: Sequence[Var], allowed: Sequence[Var], label: str, where: str
) -> None:
    """Checks that `chosen` (e.g. `h_inputs`) only uses variables from `allowed`.

    Raises `ValueError` if a variable isn't in `allowed` or appears twice. `label`
    names the argument and `where` describes `allowed` (e.g. "u or x"), for the
    message.
    """
    for var in chosen:
        if var not in allowed:
            msg = f"{label} contains {var.name!r}, which is not one of {where}"
            raise ValueError(msg)
    _check_unique([var.name for var in chosen], label)


@dataclass(frozen=True, init=False)
class System:
    """The physical process (paper, Problem 4).

    - u: the N decision variables;
    - x: the M mechanistic states, determined by the known equations 0 = g(x, y, u).
      Both are optional and go together: without them (M = 0) hybrid BO is BO of a
      composite function f(y, u) with y = h(u) (Astudillo & Frazier, arXiv
      1906.01537);
    - y: the P unknown states, determined by the unknown equations y = h(x, u),
      whose form is not part of the system (ADR 0001);
    - h_inputs: the arguments of h, drawn from u and x;
    - measured: the states the sensors read, drawn from x and y. There are P of them,
      so the root-finding problem for the other states is square.

    `g` is a `cas.Function` with column-vector inputs `(x, y, u)` and output `(M, 1)`,
    with no equations (0, 1) when M = 0. The arguments are keyword-only. Raises
    `ValueError` for an invalid argument, including mechanistic states without `g`
    or `g` without mechanistic states.
    """

    u: tuple[Var, ...]
    x: tuple[Var, ...]
    y: tuple[Var, ...]
    g: cas.Function
    h_inputs: tuple[Var, ...]
    measured: tuple[Var, ...]

    def __init__(
        self,
        *,
        u: Sequence[Var],
        x: Sequence[Var] = (),
        y: Sequence[Var],
        g: Callable[[Any], Any] | None = None,
        h_inputs: Sequence[Var],
        measured: Sequence[Var],
    ) -> None:
        # The variables: u and y are non-empty, x and g go together, and every name
        # is unique
        u, x, y = tuple(u), tuple(x), tuple(y)
        for block, label in ((u, "u"), (y, "y")):
            if not block:
                msg = f"The system needs at least one variable in {label}"
                raise ValueError(msg)
        if x and g is None:
            msg = "The mechanistic states need their known equations: pass g"
            raise ValueError(msg)
        if g is not None and not x:
            msg = (
                "g is given but there are no mechanistic states: pass x, or neither "
                "for a system without known equations"
            )
            raise ValueError(msg)
        _check_unique([var.name for var in (*u, *x, *y)], "u, x and y together")

        # The structure: what h depends on, and what the sensors read
        h_inputs, measured = tuple(h_inputs), tuple(measured)
        _check_drawn_from(h_inputs, (*u, *x), "h_inputs", "u or x")
        _check_drawn_from(measured, (*x, *y), "measured", "x or y")
        if len(measured) != len(y):
            msg = (
                f"There must be as many measured variables as unknown states "
                f"({len(y)}), so the root-finding problem is square; got {len(measured)}"
            )
            raise ValueError(msg)

        # The known equations: one per mechanistic state, none without them
        g_function = _trace("g", g or (lambda v: []), x, y, u)
        if g_function.size_out(0) != (len(x), 1):
            msg = (
                f"g must give one equation per mechanistic state ({len(x)}) as a "
                f"column, got shape {g_function.size_out(0)}"
            )
            raise ValueError(msg)

        # Frozen dataclass: set fields through object.__setattr__
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


def _h_inputs(system: System, u: Any, x: Any) -> list[Any]:
    """The arguments of h at each row of u (n, N) and x (n, M), numeric or CasADi, as
    a list of columns in the order of `system.h_inputs`."""
    columns = {var: u[:, i] for i, var in enumerate(system.u)} | {
        var: x[:, j] for j, var in enumerate(system.x)
    }
    return [columns[var] for var in system.h_inputs]


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

        # Frozen dataclass: set fields through object.__setattr__
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "f", f_function)
