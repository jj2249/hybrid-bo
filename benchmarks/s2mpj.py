"""A problem suite of benchmark experiments on S2MPJ problems, one per problem name,
built by rules from the problem's own code (ADR 0012). No S2MPJ problem is copied.

S2MPJ (Gratton & Toint, arXiv 2407.07812) holds the CUTEst problems in Python, each
in group-partially-separable form: a group is a linear part plus a weighted sum of
elements, minus a constant, through a group function and divided by a scale. The
rules turn an equality-constrained problem min f(z) s.t. c(z) = 0 into a hybrid one:

- the unknown equation (P = 1): the element sum N_j of one nonlinear constraint j,
  picked at random with a seed from the problem name, is the unknown state y; the
  oracle is y = N_j(z), its variables are the `h_inputs`;
- the known equations: every constraint, with N_j replaced by y in constraint j;
- the mechanistic states: the m variables a column-pivoted QR of dg/dz (y held) at
  the solution z* picks, ties going to variables outside N_j, as the root-finding
  problem solves g for them with y measured; the decision variables are the other
  d = n - m;
- the box: S2MPJ's finite bounds, and for each infinite one the edge of the hull of
  S2MPJ's start z0 and IPOPT's z*, widened by max(1, half the hull's width);
- the oracle optimum: the problem file's `# LO SOLTN`, checked by IPOPT.

Every build checks the traced f and c against S2MPJ's float evaluation, and a
problem outside these rules raises `UnsupportedProblemError` naming the reason
(#181 tracks lifting them). Run it, after `fetch_s2mpj.py` (the pixi task runs it):

    pixi run benchmark s2mpj HS27 [--n-runs 3 --plot ...]
    pixi run benchmark s2mpj HS27 resume STUDY

`pixi run benchmark s2mpj HS27 --help` lists the flags.
"""

import argparse
import enum
import functools
import hashlib
import importlib.util
import re
import sys
import types
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as cas
import numpy as np
import scipy.linalg
from fetch_s2mpj import CHECKOUT, COMMIT

import hybrid_bo as hb
from hybrid_bo.errors import NoSolutionError
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import NLP, Ipopt, Solution, multistart

# Agreement of the traced functions with S2MPJ's float evaluation, and the number of
# random points in the box they are compared at
TRACE_RTOL = 1e-8
TRACE_ATOL = 1e-8
N_TRACE_POINTS = 20

# Agreement of IPOPT's optimum with the stated SOLTN (HS106's 7049.248 against
# 7049.331 disagrees)
OPTIMUM_RTOL = 1e-5
OPTIMUM_ATOL = 1e-6

# The largest condition number of the states' Jacobian block at z*
MAX_CONDITION = 1e8

# The decimals of z* kept for the box, so the box doesn't carry IPOPT's last digits
BOX_DECIMALS = 6

# SoftSAA's smallest temperature, for f* near 0
MIN_TEMPERATURE = 1e-3

# Points sampled over the box for the range of the hidden term (y's start bounds)
N_RANGE_POINTS = 256


class Reason(enum.StrEnum):
    """Why a problem is outside the rules (#181 has how each would be lifted)."""

    INEQUALITIES = "inequality constraints"
    NO_DEGREES_OF_FREEDOM = "no degrees of freedom"
    ONLY_LINEAR_CONSTRAINTS = "only linear constraints"
    NO_SOLUTION = "no numeric SOLTN"
    IPOPT_DISAGREES = "IPOPT disagrees with SOLTN"
    ILL_CONDITIONED = "ill-conditioned split"
    UNTRACEABLE = "S2MPJ code can't be traced"


class UnsupportedProblemError(Exception):
    """The S2MPJ problem is outside the rules, for `reason`."""

    def __init__(self, name: str, reason: Reason, detail: str) -> None:
        """`name` is unsupported for `reason`, with `detail` saying how."""
        super().__init__(f"{name} is unsupported: {reason}: {detail}")
        self.name = name
        self.reason = reason


@dataclass(frozen=True)
class S2MPJProblem:
    """An S2MPJ problem from the checkout: `data`, the problem as S2MPJ builds it
    (its structure and float evaluation, `fx` and `cx`), and `traced`, its class
    loaded again from a patched source, whose element and group functions take
    CasADi symbols (`_traced_class`)."""

    name: str
    data: Any
    traced: type


@dataclass(frozen=True, kw_only=True)
class Split:
    """The hybrid problem the rules make of an S2MPJ problem, by variable index into
    S2MPJ's z: the decision variables and the mechanistic states (each in z's order),
    the hidden constraint (its index among the constraints and its name), the
    `h_inputs`, the box (n,), the start bounds of y, S2MPJ's start z0, IPOPT's z*,
    and the oracle optimum SOLTN with IPOPT's value."""

    decisions: tuple[int, ...]
    states: tuple[int, ...]
    hidden: int
    hidden_name: str
    h_inputs: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray
    y_start: tuple[float, float]
    start: np.ndarray
    solution: np.ndarray
    optimum: float
    ipopt_optimum: float


@dataclass(frozen=True, kw_only=True)
class BenchmarkExperiment:
    """The benchmark experiment of one S2MPJ problem: what `hb.cli.main` takes."""

    problem: hb.Problem
    measure: hb.Measure
    configurations: dict[str, hb.Configuration]
    study_settings: hb.StudySettings
    optimum: float
    notes: dict[str, str]


# Loading


def load(name: str) -> S2MPJProblem:
    """The S2MPJ problem `name` (`HS27`) from the checkout.

    Raises `FileNotFoundError` without the checkout, `ValueError` if S2MPJ has no
    such problem.
    """
    # The problem's file, and s2mpjlib importable, as the problem files import it
    if not (CHECKOUT / "s2mpjlib.py").exists():
        msg = f"no S2MPJ checkout at {CHECKOUT}: run pixi run -e benchmarks fetch-s2mpj"
        raise FileNotFoundError(msg)
    path = _problem_file(name)
    if str(CHECKOUT) not in sys.path:
        sys.path.insert(0, str(CHECKOUT))

    # The problem as S2MPJ builds it, then its class again for tracing
    spec = importlib.util.spec_from_file_location(f"s2mpj_{name}", path)
    if spec is None or spec.loader is None:
        msg = f"can't load {path}"
        raise ValueError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = getattr(module, name)()
    data.getglobs()  # the global element and group parameters, as `fx` sets them
    return S2MPJProblem(name=name, data=data, traced=_traced_class(name, path))


def _problem_file(name: str) -> Path:
    """The checkout's file of the problem `name`.

    Raises `ValueError` if there is none.
    """
    path = CHECKOUT / "python_problems" / f"{name}.py"
    if not re.fullmatch(r"\w+", name) or not path.exists():
        msg = f"S2MPJ has no problem {name!r} (no {path})"
        raise ValueError(msg)
    return path


def _traced_class(name: str, path: Path) -> type:
    """The problem's class, loaded from its source with S2MPJ's two obstacles to
    symbols patched: `np.zeros(` makes object arrays, which hold symbols (a float
    array stores NaN), and `to_scalar` passes symbols through (`float` of one is NaN).
    Neither failure would raise, so `check_trace` compares with the float evaluation.
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


# The problem's functions, traced


def _group(problem: S2MPJProblem, group: int, z: np.ndarray, hidden: Any = None) -> Any:
    """Group `group` at z (an (n, 1) object array of symbols), as S2MPJ evaluates
    its value (`evalgrsum`), with its element sum replaced by `hidden` if given."""
    data = problem.data
    value = 0.0
    if hasattr(data, "gconst") and group < len(data.gconst):
        value = -float(np.asarray(data.gconst[group]).item())
    if hasattr(data, "A") and group < data.A.shape[0]:
        row = data.A.getrow(group)
        for column, weight in zip(row.indices, row.data, strict=True):
            value = value + weight * z[column, 0]
    value = value + (_element_sum(problem, group, z) if hidden is None else hidden)

    # The group function, then the scale
    group_type = _entry(data, "grftype", group)
    if group_type is not None and group_type != "TRIVIAL":
        value = _traced_call(problem, group_type, value, group)
    scale = _entry(data, "gscale", group)
    has_scale = scale is not None and abs(scale) > 1e-15
    return value / scale if has_scale else value


def _element_sum(problem: S2MPJProblem, group: int, z: np.ndarray) -> Any:
    """The weighted sum of the elements of group `group` at z, through the traced
    element functions."""
    data = problem.data
    weights = _entry(data, "grelw", group)
    total = 0.0
    for position, element in enumerate(_elements(data, group)):
        element_z = z[_element_variables(data, element)]
        value = _traced_call(problem, data.elftype[element], element_z, element)
        total = total + (value if weights is None else weights[position] * value)
    return total


def _traced_call(problem: S2MPJProblem, function: str, *args: Any) -> Any:
    """The value of the traced element or group function `function` at `args`, as
    S2MPJ calls it. NumPy functions of symbols in S2MPJ's code warn that CasADi will
    change their result type; `check_trace` checks the values, so that is silenced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return getattr(problem.traced, function)(problem.data, 1, *args)


def _entry(data: Any, field: str, group: int) -> Any:
    """The group's entry of S2MPJ's per-group list `field`, None if it has none."""
    values = getattr(data, field, None)
    return None if values is None or group >= len(values) else values[group]


def _elements(data: Any, group: int) -> list[int]:
    """The indices of the group's elements."""
    elements = _entry(data, "grelt", group)
    return [] if elements is None else [int(element) for element in elements]


def _element_variables(data: Any, element: int) -> np.ndarray:
    """The indices into z of the element's variables."""
    return np.array(data.elvar[element], dtype=int)


def _column(symbols: Sequence[Any]) -> np.ndarray:
    """The symbols as an (n, 1) object array, as S2MPJ's functions take z."""
    z = np.empty((len(symbols), 1), dtype=object)
    for i, symbol in enumerate(symbols):
        z[i, 0] = symbol
    return z


def objective(problem: S2MPJProblem, symbols: Sequence[Any]) -> Any:
    """f at z, the symbols of S2MPJ's variables in order: the objective groups and
    the quadratic term, if any."""
    data = problem.data
    z = _column(symbols)
    value = sum((_group(problem, int(group), z) for group in data.objgrps), 0.0)
    if hasattr(data, "H"):
        column = cas.vertcat(*symbols)
        value = value + 0.5 * column.T @ cas.DM(data.H.toarray()) @ column
    return value


def constraints(
    problem: S2MPJProblem,
    symbols: Sequence[Any],
    hidden: int | None = None,
    y: Any = None,
) -> list[Any]:
    """c at z, the symbols of S2MPJ's variables in order, one entry per constraint,
    with the element sum of constraint `hidden` replaced by y if given."""
    z = _column(symbols)
    return [
        _group(problem, int(group), z, y if i == hidden else None)
        for i, group in enumerate(problem.data.congrps)
    ]


def hidden_term(problem: S2MPJProblem, symbols: Sequence[Any], hidden: int) -> Any:
    """The element sum N_j of constraint `hidden` at z."""
    group = int(problem.data.congrps[hidden])
    return _element_sum(problem, group, _column(symbols))


def traced(problem: S2MPJProblem) -> tuple[cas.Function, cas.Function]:
    """f and c as CasADi functions of the column z.

    Raises `UnsupportedProblemError` (`Reason.UNTRACEABLE`) if S2MPJ's code raises
    on symbols.
    """
    z = cas.SX.sym("z", problem.data.n)
    symbols = [z[i] for i in range(problem.data.n)]
    try:
        f = cas.SX(objective(problem, symbols))
        c = cas.vertcat(*constraints(problem, symbols))
    except Exception as error:  # noqa: BLE001  (any failure of S2MPJ's code on symbols)
        detail = f"{type(error).__name__}: {error}"
        raise UnsupportedProblemError(
            problem.name, Reason.UNTRACEABLE, detail
        ) from None
    return cas.Function("f", [z], [f]), cas.Function("c", [z], [c])


def check_trace(
    problem: S2MPJProblem, f: cas.Function, c: cas.Function, points: np.ndarray
) -> None:
    """Checks f and c against S2MPJ's float evaluation (`fx`, `cx`) at each row of
    `points`.

    Raises `UnsupportedProblemError` (`Reason.UNTRACEABLE`) at the first point where
    they differ.
    """
    data = problem.data
    for point in points:
        traced_values = np.concatenate(
            [f(point).full().ravel(), c(point).full().ravel()]
        )
        float_values = np.concatenate(
            [np.ravel(data.fx(point.copy())), np.ravel(data.cx(point.copy()))]
        )
        agrees = np.allclose(
            traced_values,
            float_values,
            rtol=TRACE_RTOL,
            atol=TRACE_ATOL,
            equal_nan=True,
        )
        if not agrees:
            detail = (
                f"at z = {point.tolist()} the traced (f, c) is "
                f"{traced_values.tolist()}, S2MPJ's {float_values.tolist()}"
            )
            raise UnsupportedProblemError(problem.name, Reason.UNTRACEABLE, detail)


# The rules


def stated_optimum(name: str) -> float:
    """f* as the problem's file states it, in its `# LO SOLTN` comment.

    Raises `UnsupportedProblemError` (`Reason.NO_SOLUTION`) if there is none, more
    than one, or it isn't a number (`???`); a size-specific `SOLTN(n)` doesn't count.
    """
    text = _problem_file(name).read_text()
    values = set(re.findall(r"^#\s*LO SOLTN\s+(\S+)\s*$", text, flags=re.MULTILINE))
    if len(values) != 1:
        detail = f"{len(values)} values stated" if values else "none stated"
        raise UnsupportedProblemError(name, Reason.NO_SOLUTION, detail)
    (value,) = values
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise UnsupportedProblemError(name, Reason.NO_SOLUTION, value) from None


def hidden_constraint_seed(name: str) -> int:
    """The seed of the hidden constraint's pick: from sha256 of the problem name, the
    same in every process (Python's `hash` of a string isn't)."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def pick_hidden(name: str, candidates: Sequence[int]) -> int:
    """The hidden constraint among `candidates`, picked at random, reproducibly."""
    rng = np.random.default_rng(hidden_constraint_seed(name))
    return candidates[int(rng.integers(len(candidates)))]


def hidden_candidates(data: Any) -> list[int]:
    """The constraints, by index, that can be hidden: the nonlinear ones with
    elements."""
    linear = {int(i) for i in data.lincons}
    return [
        i
        for i, group in enumerate(data.congrps)
        if i not in linear and _elements(data, int(group))
    ]


def hidden_variables(data: Any, hidden: int) -> tuple[int, ...]:
    """The indices into z of the hidden constraint's element variables, in order."""
    group = int(data.congrps[hidden])
    variables = {
        int(i) for element in _elements(data, group) for i in data.elvar[element]
    }
    return tuple(sorted(variables))


def choose_states(jacobian: np.ndarray, hidden: Sequence[int]) -> tuple[int, ...]:
    """The m states a column-pivoted QR of the (m, n) Jacobian picks, in z's order.
    A tie goes to the variable outside `hidden`: LAPACK pivots on the first of equal
    columns, so those come first."""
    m, n = jacobian.shape
    order = [i for i in range(n) if i not in hidden] + list(hidden)
    _, _, pivots = scipy.linalg.qr(jacobian[:, order], pivoting=True)
    return tuple(sorted(order[i] for i in pivots[:m]))


def box(
    lower: np.ndarray, upper: np.ndarray, start: np.ndarray, solution: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The box: each finite bound kept, each infinite one the edge of the hull of the
    start and the solution (rounded to `BOX_DECIMALS`), widened by max(1, half the
    hull's width)."""
    solution = np.round(solution, BOX_DECIMALS)
    low, high = np.minimum(start, solution), np.maximum(start, solution)
    margin = np.maximum(1.0, (high - low) / 2)
    return (
        np.where(np.isfinite(lower), lower, low - margin),
        np.where(np.isfinite(upper), upper, high + margin),
    )


def solve(problem: S2MPJProblem, f: cas.Function, c: cas.Function) -> Solution:
    """IPOPT's solution of the full, known problem from S2MPJ's start: min f(z) s.t.
    c(z) = 0 within S2MPJ's bounds.

    Raises `NoSolutionError` if IPOPT fails.
    """
    data = problem.data
    nlp = NLP(f=f, lower=data.xlower.ravel(), upper=data.xupper.ravel(), g=c)
    return multistart(nlp, data.x0.reshape(1, -1), Ipopt())


def split(problem: S2MPJProblem) -> Split:
    """The rules applied to the problem: its hybrid problem, by variable index.

    Raises `UnsupportedProblemError` naming the reason if the problem is outside
    them.
    """
    data, name = problem.data, problem.name
    unsupported = functools.partial(UnsupportedProblemError, name)

    # In scope: equality constraints only, a degree of freedom, a constraint to hide
    # and a stated optimum
    if data.nle + data.nge > 0:
        raise unsupported(Reason.INEQUALITIES, f"{data.nle + data.nge} of them")
    if data.n <= data.m:
        raise unsupported(Reason.NO_DEGREES_OF_FREEDOM, f"n = {data.n}, m = {data.m}")
    candidates = hidden_candidates(data)
    if not candidates:
        raise unsupported(Reason.ONLY_LINEAR_CONSTRAINTS, "no element sum to hide")
    optimum = stated_optimum(name)

    # The trace at S2MPJ's start, then IPOPT's solution from there, agreeing with
    # the stated optimum
    f, c = traced(problem)
    start = data.x0.ravel().astype(float)
    check_trace(problem, f, c, start[np.newaxis])
    try:
        solution = solve(problem, f, c)
    except NoSolutionError:
        raise unsupported(Reason.IPOPT_DISAGREES, "IPOPT found no solution") from None
    agrees = abs(solution.f - optimum) <= OPTIMUM_ATOL + OPTIMUM_RTOL * abs(optimum)
    if not agrees:
        detail = f"IPOPT found {solution.f!r}, SOLTN is {optimum!r}"
        raise unsupported(Reason.IPOPT_DISAGREES, detail)

    # The box, and the trace checked over it
    rng = np.random.default_rng(hidden_constraint_seed(name))
    lower, upper = box(data.xlower.ravel(), data.xupper.ravel(), start, solution.w)
    points = lhs(lower, upper, N_TRACE_POINTS, rng)
    check_trace(problem, f, c, np.vstack([solution.w, points]))

    # The hidden constraint, then the states by QR of dg/dz at z* with y held, as
    # the root-finding problem solves g for them; well conditioned in c too, which
    # the oracle simulation solves
    hidden = pick_hidden(name, candidates)
    h_inputs = hidden_variables(data, hidden)
    held = _held_jacobian(problem, hidden, solution.w)
    states = choose_states(held, h_inputs)
    z = cas.SX.sym("z", data.n)
    jacobian = cas.Function("J", [z], [cas.jacobian(c(z), z)])(solution.w).full()
    condition = max(
        float(np.linalg.cond(held[:, states])),
        float(np.linalg.cond(jacobian[:, states])),
    )
    if not np.any(held[hidden]):
        detail = (
            f"{data.cnames[hidden]} is its element sum plus a constant, so hiding it "
            "leaves y a constant of g"
        )
        raise unsupported(Reason.ILL_CONDITIONED, detail)
    if not condition <= MAX_CONDITION:
        detail = f"the states' Jacobian block has condition number {condition:.3g}"
        raise unsupported(Reason.ILL_CONDITIONED, detail)

    # y's start bounds: the hidden term's range over the box
    term = _hidden_term_function(problem, hidden)
    values = term(lhs(lower, upper, N_RANGE_POINTS, rng).T).full().ravel()
    y_low, y_high = float(np.min(values)), float(np.max(values))
    if y_low == y_high:
        y_low, y_high = y_low - 1.0, y_high + 1.0
    return Split(
        decisions=tuple(i for i in range(data.n) if i not in states),
        states=states,
        hidden=hidden,
        hidden_name=str(data.cnames[hidden]),
        h_inputs=h_inputs,
        lower=lower,
        upper=upper,
        y_start=(y_low, y_high),
        start=start,
        solution=solution.w,
        optimum=optimum,
        ipopt_optimum=solution.f,
    )


def _held_jacobian(problem: S2MPJProblem, hidden: int, z_value: np.ndarray) -> Any:
    """The Jacobian of the known equations in z at z_value, y held at N_j(z_value)."""
    z = cas.SX.sym("z", problem.data.n)
    y = cas.SX.sym("y")
    symbols = [z[i] for i in range(problem.data.n)]
    g = cas.vertcat(*constraints(problem, symbols, hidden, y))
    jacobian = cas.Function("J", [z, y], [cas.jacobian(g, z)])
    y_value = _hidden_term_function(problem, hidden)(z_value)
    return jacobian(z_value, y_value).full()


def _hidden_term_function(problem: S2MPJProblem, hidden: int) -> cas.Function:
    """N_j as a CasADi function of the column z."""
    z = cas.SX.sym("z", problem.data.n)
    term = hidden_term(problem, [z[i] for i in range(problem.data.n)], hidden)
    return cas.Function("N", [z], [cas.SX(term)])


# The experiment


def variable_names(data: Any) -> list[str]:
    """The names of the hybrid problem's variables: S2MPJ's, when each can name a
    `Var` and none is y, else z1, ..., zn."""
    names = [str(name) for name in data.xnames]
    is_usable = all(
        name.isidentifier() and not name.startswith("_") and name != "y"
        for name in names
    ) and len(set(names)) == len(names)
    return names if is_usable else [f"z{i + 1}" for i in range(len(names))]


def hybrid_problem(
    problem: S2MPJProblem, chosen: Split
) -> tuple[hb.Problem, Callable[[Any], list[Any]]]:
    """The hybrid problem the split makes, and its oracle y = N_j(z)."""
    data = problem.data
    names = variable_names(data)

    # The variables: the decision variables in the box, the states with S2MPJ's
    # bounds and the box as start bounds, y with the hidden term's range
    def var(i: int, bounded: bool) -> hb.Var:
        box_bounds = (float(chosen.lower[i]), float(chosen.upper[i]))
        if bounded:
            return hb.Var(names[i], *box_bounds)
        bounds = (float(data.xlower[i, 0]), float(data.xupper[i, 0]))
        return hb.Var(names[i], *bounds, start=box_bounds)

    u = [var(i, bounded=True) for i in chosen.decisions]
    x = [var(i, bounded=False) for i in chosen.states]
    y = hb.Var("y", start=chosen.y_start)

    def z(v: Any) -> list[Any]:
        """S2MPJ's z, in order, from the namespace of symbols."""
        return [getattr(v, name) for name in names]

    def oracle(v: Any) -> list[Any]:
        """The unknown equation: y is the hidden constraint's element sum."""
        return [hidden_term(problem, z(v), chosen.hidden)]

    system = hb.System(
        u=u,
        x=x,
        y=[y],
        g=lambda v: constraints(problem, z(v), chosen.hidden, v.y),
        h_inputs=[(u + x)[_position(chosen, i)] for i in chosen.h_inputs],
        measured=[y],
    )
    return hb.Problem(system, f=lambda v: objective(problem, z(v))), oracle


def _position(chosen: Split, variable: int) -> int:
    """The position of variable `variable` among u followed by x."""
    return (chosen.decisions + chosen.states).index(variable)


def benchmark_experiment(name: str) -> BenchmarkExperiment:
    """The benchmark experiment of the S2MPJ problem `name`, by the chosen.

    Raises what `load` and `split` raise.
    """
    problem = load(name)
    chosen = split(problem)
    hybrid, oracle = hybrid_problem(problem, chosen)
    measure = hb.oracle_measure(hybrid.system, oracle, noise_std=0.0)

    # The prior GP of every BO method: as the other benchmark experiments have it,
    # with a tiny fixed noise as a nugget, since the oracle simulation is noise-free
    outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
    lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
    gp = hb.gp.GP(
        mean=hb.gp.Zero(),
        kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
        noise_std=1e-6,
    )

    # Methods, with d decision variables: 25d scenarios, and SoftSAA's temperature
    # 1e-3 |f*|, at least MIN_TEMPERATURE when f* is near 0
    d = len(chosen.decisions)
    n_scenarios = 25 * d
    temperature = max(1e-3 * abs(chosen.optimum), MIN_TEMPERATURE)
    methods = {
        "hybrid_bo": functools.partial(
            hb.hybrid_bo,
            settings=hb.HybridBOSettings(
                gp=gp,
                acquisition=hb.EI(),
                approximation=hb.SAA(n_scenarios=n_scenarios),
            ),
        ),
        "hybrid_bo_soft": functools.partial(
            hb.hybrid_bo,
            settings=hb.HybridBOSettings(
                gp=gp,
                acquisition=hb.EI(),
                approximation=hb.SoftSAA(
                    temperature=temperature, n_scenarios=n_scenarios
                ),
            ),
        ),
        "standard_bo": functools.partial(
            hb.standard_bo, settings=hb.StandardBOSettings(gp=gp, acquisition=hb.EI())
        ),
        "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    }

    # Configurations: every method with 10d evaluations, 2d + 1 of them initial
    configurations = {
        label: hb.Configuration(
            method=method, n_evaluations=10 * d, n_initial=2 * d + 1
        )
        for label, method in methods.items()
    }
    names = variable_names(problem.data)
    notes = {
        "problem": name,
        "s2mpj_commit": COMMIT,
        "hidden_constraint": chosen.hidden_name,
        "states": ", ".join(names[i] for i in chosen.states),
        "box": ", ".join(
            f"{names[i]} in [{float(chosen.lower[i])!r}, {float(chosen.upper[i])!r}]"
            for i in range(len(names))
        ),
        "optimum": f"{chosen.optimum!r} (SOLTN; IPOPT {chosen.ipopt_optimum!r})",
    }
    return BenchmarkExperiment(
        problem=hybrid,
        measure=measure,
        configurations=configurations,
        study_settings=hb.StudySettings(n_runs=10, seed=None),
        optimum=chosen.optimum,
        notes=notes,
    )


# Run or plot from the command line: the problem name first, then hb.cli's command
# and flags. The oracle measurement is cheap, so the iteration figures draw its true
# curves
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python s2mpj.py",
        usage="%(prog)s PROBLEM [command] [flags]",
        description="Run a hybrid BO benchmark on an S2MPJ problem. "
        "`%(prog)s PROBLEM --help` lists the commands and flags.",
    )
    parser.add_argument("problem", help="the S2MPJ problem's name, e.g. HS27")
    problem_name = parser.parse_args(sys.argv[1:2]).problem
    try:
        experiment = benchmark_experiment(problem_name)
    except (UnsupportedProblemError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    hb.cli.main(
        experiment.problem,
        experiment.measure,
        experiment.configurations,
        experiment.study_settings,
        optimum=experiment.optimum,
        true_curves=True,
        arguments=[problem_name],
        notes=experiment.notes,
    )
