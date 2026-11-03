"""An S2MPJ problem as the generator's rules see it: `read` loads it from the checkout
and traces it once with CasADi symbols into a `Source`, a small record of CasADi
functions, arrays and names. Its one tie to S2MPJ is `evaluate`, S2MPJ's own float
evaluation, which the rules call but never look inside. Only this module touches
S2MPJ's code.

S2MPJ (Gratton & Toint, arXiv 2407.07812) holds the CUTEst problems in Python, each in
group-partially-separable form: a group is a linear part plus a weighted sum of
elements, minus a constant, through a group function and divided by a scale. The
trace keeps each constraint's element sum apart, so the rules can hide one.
"""

import enum
import importlib.util
import re
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as cas
import numpy as np
from fetch_s2mpj import CHECKOUT


class Reason(enum.StrEnum):
    """Why a problem is outside the rules (#181 has how each would be lifted)."""

    INEQUALITIES = "inequality constraints"
    NO_DEGREES_OF_FREEDOM = "no degrees of freedom"
    ONLY_LINEAR_CONSTRAINTS = "only linear constraints"
    NO_SOLUTION = "no numeric SOLTN"
    IPOPT_FAILED = "IPOPT found no solution"
    IPOPT_DISAGREES = "IPOPT disagrees with SOLTN"
    HIDDEN_TERM_IS_CONSTANT = "hidden constraint is its hidden term plus a constant"
    ILL_CONDITIONED = "ill-conditioned split"
    UNTRACEABLE = "S2MPJ code can't be traced"


class UnsupportedProblemError(Exception):
    """The S2MPJ problem is outside the rules, for `reason`."""

    def __init__(self, name: str, reason: Reason, detail: str) -> None:
        """`name` is unsupported for `reason`, with `detail` saying how."""
        super().__init__(f"{name} is unsupported: {reason}: {detail}")
        self.name = name
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, kw_only=True)
class Source:
    """An S2MPJ problem, min f(v) s.t. c(v) = 0 over S2MPJ's n variables v, traced.

    - `variable_names` (n) and `constraint_names` (m): S2MPJ's;
    - `lower`, `upper` (n,): S2MPJ's bounds, possibly infinite, and `start` (n,):
      S2MPJ's start point;
    - `n_inequalities`: how many of the m constraints are inequalities;
    - `nonlinear` (m): whether each constraint is nonlinear, and
      `element_variables` (m): the variables of each constraint's elements, in order;
    - `stated_optima`: the distinct values of the file's `# LO SOLTN` comments;
    - `f(v)`: the objective; `c(v, s)` (m,): the constraints with their element sums
      s (m,) given; `element_sums(v)` (m,): the element sums, so c(v) is
      `c(v, element_sums(v))`. All take columns;
    - `evaluate(v)`: f and c (m,) at the point v (n,), by S2MPJ's own float code, to
      check the trace against.
    """

    name: str
    variable_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    start: np.ndarray
    n_inequalities: int
    nonlinear: tuple[bool, ...]
    element_variables: tuple[tuple[int, ...], ...]
    stated_optima: tuple[str, ...]
    f: cas.Function
    c: cas.Function
    element_sums: cas.Function
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]]

    @property
    def n(self) -> int:
        """The number of variables."""
        return len(self.variable_names)

    @property
    def m(self) -> int:
        """The number of constraints."""
        return len(self.constraint_names)


def classified_sizes(name: str) -> tuple[str, str]:
    """n and m as the problem file's CUTEst classification states them (`3` and `1`
    in `C-CSQR2-AN-3-1`), `V` for a size the user can choose.

    Raises `ValueError` if the problem has no classification.
    """
    text = problem_file(name).read_text()
    match = re.search(r'classification = "[^"]*-(\w+)-(\w+)"', text)
    if match is None:
        msg = f"{name} has no classification"
        raise ValueError(msg)
    return match.group(1), match.group(2)


def problem_names() -> list[str]:
    """The names of every problem in the checkout, sorted."""
    return sorted(path.stem for path in (CHECKOUT / "python_problems").glob("*.py"))


def problem_file(name: str) -> Path:
    """The checkout's file of the problem `name`.

    Raises `FileNotFoundError` without the checkout, `ValueError` if S2MPJ has no such
    problem.
    """
    if not (CHECKOUT / "s2mpjlib.py").exists():
        msg = f"no S2MPJ checkout at {CHECKOUT}: run pixi run -e benchmarks fetch-s2mpj"
        raise FileNotFoundError(msg)
    path = CHECKOUT / "python_problems" / f"{name}.py"
    if not re.fullmatch(r"\w+", name) or not path.exists():
        msg = f"S2MPJ has no problem {name!r} (no {path})"
        raise ValueError(msg)
    return path


def sizes(name: str) -> tuple[int, int, int, bool]:
    """n, m, the number of inequalities and whether there is an objective, of the
    problem `name` as S2MPJ builds it, without tracing it.

    Raises what `read` raises, except for a failure on symbols.
    """
    data = _load(name, problem_file(name))
    has_objective = len(getattr(data, "objgrps", [])) > 0 or hasattr(data, "H")
    return (
        int(data.n),
        len(_constraint_groups(data)),
        _n_inequalities(data),
        has_objective,
    )


def read(name: str) -> Source:
    """The S2MPJ problem `name` (`HS27`) from the checkout, traced.

    Raises `FileNotFoundError` without the checkout, `ValueError` if S2MPJ has no such
    problem, and `UnsupportedProblemError` (`Reason.UNTRACEABLE`) if its code fails to
    load or fails on symbols.
    """
    path = problem_file(name)
    data = _load(name, path)
    traced = _traced_class(name, path)

    # The structure the rules read
    groups = _constraint_groups(data)
    m = len(groups)
    linear = {int(i) for i in getattr(data, "lincons", [])}
    element_variables = tuple(
        tuple(sorted({int(i) for e in _elements(data, g) for i in data.elvar[e]}))
        for g in groups
    )
    text = path.read_text()
    stated = sorted(set(re.findall(r"^#\s*LO SOLTN\s+(\S+)\s*$", text, re.MULTILINE)))

    # f, c with the element sums given, and the element sums, traced once
    v = cas.SX.sym("v", data.n)
    s = cas.SX.sym("s", m)
    column = _column([v[i] for i in range(data.n)])
    try:
        f = cas.SX(_objective(data, traced, column, v))
        c = cas.vertcat(
            *(
                _group(data, traced, int(group), column, s[i])
                for i, group in enumerate(groups)
            )
        )
        sums = cas.vertcat(
            *(_element_sum(data, traced, int(group), column) for group in groups)
        )
    except Exception as error:  # noqa: BLE001  (any failure of S2MPJ's code on symbols)
        detail = f"{type(error).__name__}: {error}"
        raise UnsupportedProblemError(name, Reason.UNTRACEABLE, detail) from None

    def evaluate(point: np.ndarray) -> tuple[float, np.ndarray]:
        """f and c at `point` by S2MPJ's float code, which writes into its input."""
        f_value = float(np.ravel(data.fx(point.copy()))[0])
        c_value = np.ravel(data.cx(point.copy())) if m else np.zeros(0)
        return f_value, c_value.astype(float)

    return Source(
        name=name,
        variable_names=tuple(str(x) for x in data.xnames),
        constraint_names=tuple(str(x) for x in getattr(data, "cnames", [])),
        lower=data.xlower.ravel().astype(float),
        upper=data.xupper.ravel().astype(float),
        start=data.x0.ravel().astype(float),
        n_inequalities=_n_inequalities(data),
        nonlinear=tuple(i not in linear for i in range(m)),
        element_variables=element_variables,
        stated_optima=tuple(stated),
        f=cas.Function("f", [v], [f]),
        c=cas.Function("c", [v, s], [c]),
        element_sums=cas.Function("element_sums", [v], [cas.SX(sums)]),
        evaluate=evaluate,
    )


# Loading


def _load(name: str, path: Path) -> Any:
    """The problem as S2MPJ builds it, with its global element and group parameters
    set (`getglobs`), as `fx` sets them.

    Raises `UnsupportedProblemError` (`Reason.UNTRACEABLE`) if its code fails.
    """
    # The problem files import s2mpjlib as a top-level module, so the checkout goes
    # on sys.path, once, for the process: S2MPJ offers no other way to load them
    if str(CHECKOUT) not in sys.path:
        sys.path.insert(0, str(CHECKOUT))
    spec = importlib.util.spec_from_file_location(f"s2mpj_{name}", path)
    if spec is None or spec.loader is None:
        msg = f"can't load {path}"
        raise ValueError(msg)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = getattr(module, name)()
        data.getglobs()  # sets attributes on S2MPJ's object, as `fx` would first
    except Exception as error:  # noqa: BLE001  (any failure of S2MPJ's code)
        detail = f"doesn't load: {type(error).__name__}: {error}"
        raise UnsupportedProblemError(name, Reason.UNTRACEABLE, detail) from None
    return data


def _constraint_groups(data: Any) -> list[int]:
    """The groups of the constraints, in order; none for an unconstrained problem,
    which S2MPJ builds without `congrps`."""
    return [int(group) for group in getattr(data, "congrps", [])]


def _n_inequalities(data: Any) -> int:
    """The number of inequality constraints."""
    return int(getattr(data, "nle", 0) + getattr(data, "nge", 0))


def _traced_class(name: str, path: Path) -> type:
    """The problem's class, loaded from its source with S2MPJ's two obstacles to
    symbols patched: `np.zeros(` makes object arrays, which hold symbols (a float
    array stores NaN), and `to_scalar` passes symbols through (`float` of one is NaN).
    Neither failure would raise, so the rules check the trace against `evaluate`.
    """
    source = path.read_text().replace("np.zeros(", "_zeros(")
    module = types.ModuleType(f"s2mpj_{name}_traced")
    module.__dict__["_zeros"] = _object_zeros
    exec(compile(source, f"{path} (traced)", "exec"), module.__dict__)  # noqa: S102
    module.__dict__["to_scalar"] = _symbolic_scalar
    return getattr(module, name)


def _object_zeros(shape: Any, *_args: Any, **_kwargs: Any) -> np.ndarray:
    """`np.zeros(shape)` as an object array, which can hold CasADi symbols."""
    return np.zeros(shape, dtype=object)


def _symbolic_scalar(value: Any) -> Any:
    """S2MPJ's `to_scalar` (the first entry of an array, as a float), keeping a
    symbol a symbol."""
    if isinstance(value, np.ndarray):
        value = value.flatten()[0]
    return value if isinstance(value, cas.SX) else float(value)


# Tracing


def _objective(data: Any, traced: type, column: np.ndarray, v: cas.SX) -> Any:
    """f at the variables v (`column` is v as S2MPJ takes it): the objective groups
    and the quadratic term, if any."""
    value = sum((_group(data, traced, int(g), column) for g in data.objgrps), 0.0)
    if hasattr(data, "H"):
        value = value + 0.5 * v.T @ cas.DM(data.H.toarray()) @ v
    return value


def _group(
    data: Any, traced: type, group: int, column: np.ndarray, element_sum: Any = None
) -> Any:
    """Group `group` at `column`, as S2MPJ evaluates its value (`evalgrsum`), with
    `element_sum` in place of its element sum if given."""
    value = 0.0
    if hasattr(data, "gconst") and group < len(data.gconst):
        value = -float(np.asarray(data.gconst[group]).item())
    if hasattr(data, "A") and group < data.A.shape[0]:
        row = data.A.getrow(group)
        for i, weight in zip(row.indices, row.data, strict=True):
            value = value + weight * column[i, 0]
    if element_sum is None:
        element_sum = _element_sum(data, traced, group, column)
    value = value + element_sum

    # The group function, then the scale
    group_type = _entry(data, "grftype", group)
    if group_type is not None and group_type != "TRIVIAL":
        value = _traced_call(data, traced, group_type, value, group)
    scale = _entry(data, "gscale", group)
    has_scale = scale is not None and abs(scale) > 1e-15
    return value / scale if has_scale else value


def _element_sum(data: Any, traced: type, group: int, column: np.ndarray) -> Any:
    """The weighted sum of the elements of group `group` at `column`, through the
    traced element functions; 0 for a group without elements."""
    weights = _entry(data, "grelw", group)
    total = 0.0
    for position, element in enumerate(_elements(data, group)):
        element_column = column[np.array(data.elvar[element], dtype=int)]
        value = _traced_call(
            data, traced, data.elftype[element], element_column, element
        )
        total = total + (value if weights is None else weights[position] * value)
    return total


def _traced_call(data: Any, traced: type, function: str, *args: Any) -> Any:
    """The value of the traced element or group function `function` at `args`, as
    S2MPJ calls it. NumPy functions of symbols in S2MPJ's code warn that CasADi will
    change their result type; the rules check the values, so that is silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return getattr(traced, function)(data, 1, *args)


def _entry(data: Any, field: str, group: int) -> Any:
    """The group's entry of S2MPJ's per-group list `field`, None if it has none."""
    values = getattr(data, field, None)
    return None if values is None or group >= len(values) else values[group]


def _elements(data: Any, group: int) -> list[int]:
    """The indices of the group's elements."""
    elements = _entry(data, "grelt", group)
    return [] if elements is None else [int(element) for element in elements]


def _column(symbols: list[Any]) -> np.ndarray:
    """The symbols as an (n, 1) object array, as S2MPJ's functions take the
    variables."""
    column = np.empty((len(symbols), 1), dtype=object)
    for i, symbol in enumerate(symbols):
        column[i, 0] = symbol
    return column
