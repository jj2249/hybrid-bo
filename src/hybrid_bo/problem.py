"""The problem: a system described in the paper's notation, plus an objective.

Variables are named `Var`s. The known equations `g` and the objective `f` are callables
over those variables by name (`lambda v: [v.x + cas.exp(v.x) - v.y]`); each is called
once with CasADi symbols and turned into a `cas.Function`.
"""

import keyword
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.utils import Symbolic, lhs, uniform


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


def start_bounds(variables: Sequence[Var]) -> tuple[np.ndarray, np.ndarray]:
    """The lower and upper start bounds of the variables, each of shape (d,)."""
    return (
        np.array([var.start[0] for var in variables]),
        np.array([var.start[1] for var in variables]),
    )


def lhs_points(
    variables: Sequence[Var], n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points from a Latin hypercube over the variables' start bounds, shape (n, d)."""
    return lhs(*start_bounds(variables), n, rng)


def uniform_points(
    variables: Sequence[Var], n: int, rng: np.random.Generator
) -> np.ndarray:
    """n points drawn uniformly from the variables' start bounds, shape (n, d)."""
    return uniform(*start_bounds(variables), n, rng)


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


def _to_casadi_function(
    name: str,
    equations: Callable[[Any], Any],
    x: Sequence[Var],
    y: Sequence[Var],
    u: Sequence[Var],
) -> cas.Function:
    """Turns the user's `equations` (`g` or `f`) into `name(x, y, u) -> column`, a
    `cas.Function` that the solvers can evaluate and differentiate exactly.

    `equations` is called once, with a CasADi symbol in place of each variable. Arithmetic
    on symbols records the expression instead of computing a number, so the result is the
    equations themselves. The inputs x, y and u are columns in declaration order, the
    paper's argument order.

    Raises `TypeError` if `equations` returns anything that isn't CasADi expressions, e.g.
    `None` or a nested list. Checking the output's size is left to the caller, which knows
    what size it expects.
    """
    # Call the equations once, on symbols
    symbols = {var.name: Symbolic.sym(var.name) for var in (*x, *y, *u)}
    expressions = equations(_Variables(symbols))

    # They may give one expression or a list: stack them into a column either way
    parts = expressions if isinstance(expressions, list | tuple) else [expressions]
    try:
        column = cas.vertcat(*parts)
    except NotImplementedError:  # CasADi's error for arguments it can't stack
        msg = (
            f"{name} must return a CasADi expression or a list of scalar expressions "
            f"built from v; got {expressions!r}"
        )
        raise TypeError(msg) from None

    # The function of x, y and u, each a column of its variables' symbols
    x_column, y_column, u_column = (
        cas.vertcat(*(symbols[var.name] for var in block)) for block in (x, y, u)
    )
    return cas.Function(
        name, [x_column, y_column, u_column], [column], ["x", "y", "u"], [name]
    )


@dataclass(frozen=True, init=False)
class System:
    """The physical process (paper, Problem 4).

    - u: the N decision variables;
    - x: the M mechanistic states, determined by the known equations 0 = g(x, y, u).
      x and g are optional and go together: without them (M = 0) hybrid BO is BO of
      a composite function f(y, u) with y = h(u) (Astudillo & Frazier, arXiv
      1906.01537);
    - y: the P unknown states, determined by the unknown equations y = h(x, u),
      whose form is not part of the system;
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
        # The variables: u and y are non-empty, and x and g go together
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

        # The structure: every name is unique, and h and the sensors use variables of the
        # system
        h_inputs, measured = tuple(h_inputs), tuple(measured)
        for chosen, allowed, label, where in (
            ((*u, *x, *y), (*u, *x, *y), "u, x and y together", "u, x or y"),
            (h_inputs, (*u, *x), "h_inputs", "u or x"),
            (measured, (*x, *y), "measured", "x or y"),
        ):
            for var in chosen:
                if var not in allowed:
                    msg = f"{label} contains {var.name!r}, which is not one of {where}"
                    raise ValueError(msg)
            names = [var.name for var in chosen]
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                msg = (
                    f"{label} must not repeat a variable; repeated: "
                    f"{', '.join(repeated)}"
                )
                raise ValueError(msg)
        if len(measured) != len(y):
            msg = (
                f"There must be as many measured variables as unknown states "
                f"({len(y)}), so the root-finding problem is square; got {len(measured)}"
            )
            raise ValueError(msg)

        # The known equations: one per mechanistic state, none without them
        g_function = _to_casadi_function("g", g or (lambda v: []), x, y, u)
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

    @property
    def states(self) -> tuple[Var, ...]:
        """The states in the order used for arrays of states: x, then y."""
        return (*self.x, *self.y)

    @property
    def unmeasured(self) -> tuple[Var, ...]:
        """The states the root-finding problem solves for, in state order."""
        return tuple(var for var in self.states if var not in self.measured)

    def h_arguments(self, u: Any, x: Any) -> list[Any]:
        """The arguments of h at each row of u (n, N) and x (n, M), numeric or CasADi, as
        a list of columns in the order of `h_inputs`."""
        columns = {var: u[:, i] for i, var in enumerate(self.u)} | {
            var: x[:, j] for j, var in enumerate(self.x)
        }
        return [columns[var] for var in self.h_inputs]


@dataclass(frozen=True, init=False)
class Problem:
    """A system plus the objective to minimize.

    `f` is a `cas.Function` with column-vector inputs `(x, y, u)` and a scalar output.
    """

    system: System
    f: cas.Function

    def __init__(self, system: System, f: Callable[[Any], Any]) -> None:
        f_function = _to_casadi_function("f", f, system.x, system.y, system.u)
        if f_function.size_out(0) != (1, 1):
            msg = f"f must be a scalar, got shape {f_function.size_out(0)}"
            raise ValueError(msg)

        # Frozen dataclass: set fields through object.__setattr__
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "f", f_function)
