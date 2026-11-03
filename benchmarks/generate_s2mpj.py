"""Generates a benchmark experiment from each S2MPJ problem the rules support, as an
ordinary experiment file in `benchmarks/s2mpj/` (ADR 0012), and `SUITE.md` there, the
table of every in-scope problem: generated, or refused with the reason.

`benchmarks/README.md` states the rules that split a problem into a hybrid one, and
works HS27 through; `split` applies them. Each file is `s2mpj_experiment.py.template`
filled in, and is checked against S2MPJ's float evaluation before it is written.

The generated files are never edited by hand. A change to the rules, or to how
experiments are written, goes into this module or the template, raises
`GENERATOR_VERSION`, and every file is regenerated. Regenerate named problems, or
every in-scope one:

    pixi run -e benchmarks generate-s2mpj [NAMES...]
"""

import argparse
import builtins
import functools
import hashlib
import importlib.util
import keyword
import shutil
import string
import subprocess
import sys
import tempfile
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import casadi as cas
import numpy as np
import scipy.linalg
from fetch_s2mpj import COMMIT
from s2mpj_source import (
    Reason,
    Source,
    UnsupportedProblemError,
    classified_sizes,
    problem_names,
    read,
    sizes,
)

from hybrid_bo.errors import NoSolutionError
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import NLP, Ipopt, multistart

# Raise it with any change to what the generator writes, so a file's header says
# which rules made it
GENERATOR_VERSION = 1

# Where the generated experiments and SUITE.md go
OUTPUT_DIR = Path(__file__).parent / "s2mpj"

# Every generated experiment, with `$name` fields: a change to how the benchmarks are
# written goes here (or into the rules), then every file is regenerated
TEMPLATE = string.Template(
    (Path(__file__).parent / "s2mpj_experiment.py.template").read_text()
)

# The suite: problems with equality constraints only and at most this many variables
MAX_VARIABLES = 10

# Agreement of the trace, and of a generated file, with S2MPJ's float evaluation, and
# the number of random points within the bounds they are compared at
TRACE_RTOL = 1e-8
TRACE_ATOL = 1e-8
N_TRACE_POINTS = 20

# Agreement of IPOPT's optimum with the stated SOLTN (HS106's 7049.248 against
# 7049.331 disagrees)
OPTIMUM_RTOL = 1e-5
OPTIMUM_ATOL = 1e-6

# The largest condition number of the states' Jacobian block at v*
MAX_CONDITION = 1e8

# The decimals of v* kept, so the bounds don't carry IPOPT's last digits
SOLUTION_DECIMALS = 6

# SoftSAA's smallest temperature, for a small but nonzero f*
MIN_TEMPERATURE = 1e-6

# Points sampled within the bounds for the hidden term's range (y's start bounds),
# which is rounded out to this many decimals
N_RANGE_POINTS = 256
RANGE_DECIMALS = 6

# Names a generated file defines, which a variable's name must not shadow
_FILE_NAMES = frozenset(
    {"functools", "cas", "hb", "y", "g", "h", "f", "system", "problem", "measure"}
    | {"optimum", "outputscale", "lengthscale", "gp", "methods", "configurations"}
    | {"study_settings", "notes", "label", "method", "v"}
)


@dataclass(frozen=True, kw_only=True)
class Split:
    """The hybrid problem the rules make of an S2MPJ problem, by index into S2MPJ's
    variables: the decision variables and the mechanistic states (each in S2MPJ's
    order), the hidden constraint, the `h_inputs`, the bounds the rules give every
    variable (n,), and y's start bounds."""

    decisions: tuple[int, ...]
    states: tuple[int, ...]
    hidden: int
    h_inputs: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray
    y_start: tuple[float, float]


@dataclass(frozen=True, kw_only=True)
class Origin:
    """Where a split comes from, for the generated file's header: the S2MPJ commit,
    S2MPJ's start, v* (rounded), the oracle optimum as the problem file states it
    (its literal and value), and IPOPT's."""

    commit: str
    start: np.ndarray
    solution: np.ndarray
    stated_optimum: str
    optimum: float
    ipopt_optimum: float


@dataclass(frozen=True, kw_only=True)
class Outcome:
    """What the rules made of one problem: n and m (None if it didn't load), and the
    hidden constraint's name and the generated experiment's checked text, or why it
    was refused."""

    name: str
    n: int | None
    m: int | None
    hidden: str | None = None
    text: str | None = None
    error: UnsupportedProblemError | None = None


# The rules


def split(source: Source) -> tuple[Split, Origin]:
    """The rules applied to the problem: its hybrid problem, and where it comes from.
    The rules draw their random numbers from `rng_of(source.name)`, not from an `rng`
    the caller passes, as the split must be the same in every process.

    Raises `UnsupportedProblemError` naming the reason if the problem is outside
    them.
    """
    candidates, stated, optimum = _scope(source)
    rng = rng_of(source.name)
    hidden = _hide(source, candidates, rng)
    solution, ipopt_optimum = _solve(source, stated, optimum)
    lower, upper = bounds(source.lower, source.upper, source.start, solution)
    points = lhs(lower, upper, N_TRACE_POINTS, rng)
    check_trace(source, np.vstack([source.start, solution, points]))
    states = _states(source, hidden, solution)
    chosen = Split(
        decisions=tuple(i for i in range(source.n) if i not in states),
        states=states,
        hidden=hidden,
        h_inputs=source.element_variables[hidden],
        lower=lower,
        upper=upper,
        y_start=_y_start(source, hidden, lower, upper, rng),
    )
    origin = Origin(
        commit=COMMIT,
        start=source.start,
        solution=solution,
        stated_optimum=stated,
        optimum=optimum,
        ipopt_optimum=ipopt_optimum,
    )
    return chosen, origin


def _scope(source: Source) -> tuple[list[int], str, float]:
    """The constraints that can be hidden (nonlinear, with elements) and the stated
    optimum, as its literal and value, of a problem in scope: equality constraints
    only, a degree of freedom, a constraint to hide and one numeric SOLTN.

    Raises `UnsupportedProblemError` otherwise.
    """
    unsupported = functools.partial(UnsupportedProblemError, source.name)
    if source.n_inequalities:
        raise unsupported(Reason.INEQUALITIES, f"{source.n_inequalities} of them")
    if source.n <= source.m:
        raise unsupported(
            Reason.NO_DEGREES_OF_FREEDOM, f"n = {source.n}, m = {source.m}"
        )
    candidates = [
        j
        for j in range(source.m)
        if source.nonlinear[j] and source.element_variables[j]
    ]
    if not candidates:
        raise unsupported(Reason.ONLY_LINEAR_CONSTRAINTS, "no element sum to hide")
    if len(source.stated_optima) != 1:
        detail = f"{len(source.stated_optima)} stated" if source.stated_optima else ""
        raise unsupported(Reason.NO_SOLUTION, detail or "none stated")
    (stated,) = source.stated_optima
    try:
        optimum = float(stated.replace("D", "E").replace("d", "e"))
    except ValueError:
        raise unsupported(Reason.NO_SOLUTION, stated) from None
    return candidates, stated, optimum


def _hide(source: Source, candidates: Sequence[int], rng: np.random.Generator) -> int:
    """The hidden constraint, one of `candidates` at random: the first draw of the
    rules' random numbers, so it doesn't depend on IPOPT.

    Raises `UnsupportedProblemError` (`Reason.HIDDEN_TERM_IS_CONSTANT`) if its known
    equation keeps no variable: y would be a constant of g that no state determines.
    """
    hidden = candidates[int(rng.integers(len(candidates)))]
    v, y = cas.SX.sym("v", source.n), cas.SX.sym("y")
    if not cas.depends_on(known_equations(source, v, y, hidden)[hidden], v):
        detail = f"{source.constraint_names[hidden]} leaves g a function of y alone"
        raise UnsupportedProblemError(
            source.name, Reason.HIDDEN_TERM_IS_CONSTANT, detail
        )
    return hidden


def _solve(source: Source, stated: str, optimum: float) -> tuple[np.ndarray, float]:
    """v*, rounded to `SOLUTION_DECIMALS`, and f there: IPOPT's solution of the full,
    known problem from S2MPJ's start.

    Raises `UnsupportedProblemError` if IPOPT fails (`Reason.IPOPT_FAILED`) or its
    optimum isn't the stated one (`Reason.IPOPT_DISAGREES`).
    """
    unsupported = functools.partial(UnsupportedProblemError, source.name)
    v = cas.SX.sym("v", source.n)
    c = cas.Function("c", [v], [source.c(v, source.element_sums(v))])
    nlp = NLP(f=source.f, lower=source.lower, upper=source.upper, g=c)
    try:
        solved = multistart(nlp, source.start.reshape(1, -1), Ipopt())
    except NoSolutionError:
        raise unsupported(Reason.IPOPT_FAILED, "from S2MPJ's start") from None
    agrees = abs(solved.f - optimum) <= OPTIMUM_ATOL + OPTIMUM_RTOL * abs(optimum)
    if not agrees:
        detail = f"IPOPT found {solved.f:.7g}, SOLTN is {stated}"
        raise unsupported(Reason.IPOPT_DISAGREES, detail)
    return np.round(solved.w, SOLUTION_DECIMALS) + 0.0, solved.f  # + 0.0: no -0.0


def _states(source: Source, hidden: int, solution: np.ndarray) -> tuple[int, ...]:
    """The mechanistic states: by QR of dg/dv at v* with y held, as the root-finding
    problem solves g for them, and well conditioned in c too, which the oracle
    simulation solves. Evaluated at the rounded v*, so IPOPT's last digits don't
    choose them.

    Raises `UnsupportedProblemError` (`Reason.ILL_CONDITIONED`) if either block's
    condition number is above `MAX_CONDITION`.
    """
    v, y = cas.SX.sym("v", source.n), cas.SX.sym("y")
    g = known_equations(source, v, y, hidden)
    c = source.c(v, source.element_sums(v))
    held_term = float(source.element_sums(solution)[hidden])
    held = cas.Function("held", [v, y], [cas.jacobian(g, v)])(solution, held_term)
    full = cas.Function("full", [v], [cas.jacobian(c, v)])(solution)
    states = choose_states(held.full(), source.element_variables[hidden])
    condition = max(
        float(np.linalg.cond(held.full()[:, states])),
        float(np.linalg.cond(full.full()[:, states])),
    )
    is_well_conditioned = condition <= MAX_CONDITION  # False for NaN too
    if not is_well_conditioned:
        detail = f"the states' Jacobian block has condition number {condition:.3g}"
        raise UnsupportedProblemError(source.name, Reason.ILL_CONDITIONED, detail)
    return states


def _y_start(
    source: Source,
    hidden: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """y's start bounds: the hidden term's range over Latin hypercube points within
    the bounds, rounded out, and widened by 1 each way if it is a single value."""
    term = source.element_sums(lhs(lower, upper, N_RANGE_POINTS, rng).T)
    values = term.full()[hidden]
    low, high = _round_out(float(np.min(values)), float(np.max(values)))
    return (low - 1.0, high + 1.0) if low == high else (low, high)


def rng_of(name: str) -> np.random.Generator:
    """The rules' random numbers for the problem `name`, seeded from sha256 of the
    name, so every process draws the same (Python's `hash` of a string doesn't)."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def known_equations(source: Source, v: cas.SX, y: cas.SX, hidden: int) -> cas.SX:
    """g at the variables v and y: S2MPJ's constraints, with the hidden constraint's
    hidden term replaced by y."""
    sums = source.element_sums(v)
    given = cas.vertcat(*(y if j == hidden else sums[j] for j in range(source.m)))
    return source.c(v, given)


def choose_states(jacobian: np.ndarray, hidden: Sequence[int]) -> tuple[int, ...]:
    """The m states a column-pivoted QR of the (m, n) Jacobian picks, in S2MPJ's
    order. A tie goes to the variable outside `hidden`: LAPACK pivots on the first of
    equal columns, so those come first."""
    m, n = jacobian.shape
    order = [i for i in range(n) if i not in hidden] + list(hidden)
    _, _, pivots = scipy.linalg.qr(jacobian[:, order], pivoting=True)
    return tuple(sorted(order[i] for i in pivots[:m]))


def bounds(
    lower: np.ndarray, upper: np.ndarray, start: np.ndarray, solution: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each finite bound kept, each infinite one the edge of the hull of the start and
    the solution, widened by max(1, half the hull's width)."""
    low, high = np.minimum(start, solution), np.maximum(start, solution)
    margin = np.maximum(1.0, (high - low) / 2)
    return (
        np.where(np.isfinite(lower), lower, low - margin),
        np.where(np.isfinite(upper), upper, high + margin),
    )


def check_trace(source: Source, points: np.ndarray) -> None:
    """Checks the traced f and c against S2MPJ's float evaluation at each row of
    `points`.

    Raises `UnsupportedProblemError` (`Reason.UNTRACEABLE`) at the first point where
    they differ.
    """
    for point in points:
        sums = source.element_sums(point)
        traced = np.concatenate(
            [source.f(point).full().ravel(), source.c(point, sums).full().ravel()]
        )
        _check_agrees(source, point, traced, "the trace")


def temperature(optimum: float) -> float:
    """SoftSAA's temperature for the stated optimum f*: 1e-3 if it is exactly 0 (it
    is the file's literal, not IPOPT's value), otherwise 1e-3 |f*| and at least
    `MIN_TEMPERATURE`, to 6 significant digits."""
    if optimum == 0.0:
        return 1e-3
    return float(f"{max(1e-3 * abs(optimum), MIN_TEMPERATURE):.6g}")


def _round_out(low: float, high: float) -> tuple[float, float]:
    """[low, high] widened to `RANGE_DECIMALS` decimals."""
    scale = 10.0**RANGE_DECIMALS
    return float(np.floor(low * scale) / scale), float(np.ceil(high * scale) / scale)


def _check_agrees(
    source: Source, point: np.ndarray, values: np.ndarray, what: str
) -> None:
    """Checks f and c from `what` at `point` (`values`, f first) against S2MPJ's float
    evaluation.

    Raises `UnsupportedProblemError` (`Reason.UNTRACEABLE`) if they differ.
    """
    f_value, c_value = source.evaluate(point)
    expected = np.concatenate([[f_value], c_value])
    agrees = np.allclose(
        values, expected, rtol=TRACE_RTOL, atol=TRACE_ATOL, equal_nan=True
    )
    if not agrees:
        detail = (
            f"at {np.array2string(point, precision=6)} {what} gives (f, c) = "
            f"{np.array2string(values, precision=6)}, S2MPJ "
            f"{np.array2string(expected, precision=6)}"
        )
        raise UnsupportedProblemError(source.name, Reason.UNTRACEABLE, detail)


# Python from CasADi expressions

# The precedence of what an expression prints as, from loosest to tightest
_SUM, _PRODUCT, _UNARY, _POWER, _ATOM = range(5)

_BINARY = {
    cas.OP_ADD: ("+", _SUM),
    cas.OP_SUB: ("-", _SUM),
    cas.OP_MUL: ("*", _PRODUCT),
    cas.OP_DIV: ("/", _PRODUCT),
    cas.OP_POW: ("**", _POWER),
    cas.OP_CONSTPOW: ("**", _POWER),
}

_FUNCTIONS = {
    cas.OP_SQRT: "sqrt",
    cas.OP_EXP: "exp",
    cas.OP_LOG: "log",
    cas.OP_SIN: "sin",
    cas.OP_COS: "cos",
    cas.OP_TAN: "tan",
    cas.OP_ASIN: "asin",
    cas.OP_ACOS: "acos",
    cas.OP_ATAN: "atan",
    cas.OP_SINH: "sinh",
    cas.OP_COSH: "cosh",
    cas.OP_TANH: "tanh",
    cas.OP_ASINH: "asinh",
    cas.OP_ACOSH: "acosh",
    cas.OP_ATANH: "atanh",
    cas.OP_FABS: "fabs",
    cas.OP_SIGN: "sign",
    cas.OP_FLOOR: "floor",
    cas.OP_CEIL: "ceil",
    cas.OP_ERF: "erf",
    cas.OP_ATAN2: "atan2",
    cas.OP_FMIN: "fmin",
    cas.OP_FMAX: "fmax",
}


def to_python(expression: cas.SX) -> str:
    """The scalar CasADi expression as Python over `v` (`v.X1 + 1 + v.y`), its
    symbols read as `v.<name>` and its functions from `cas`. Evaluated with the same
    symbols, the text gives the same expression, so it computes the same values.

    Raises `ValueError` for an operation it can't write, or a constant that isn't
    finite.
    """
    return _printed(expression)[0]


def _printed(e: cas.SX) -> tuple[str, int]:
    """The expression's text and its precedence."""
    # Symbols and constants
    if e.is_symbolic():
        return f"v.{e.name()}", _ATOM
    if e.is_constant():
        value = float(e)
        return _literal(value), _UNARY if value < 0 else _ATOM

    # Binary operators, with the parentheses that keep the expression's tree
    op = e.op()
    if op in _BINARY or op == cas.OP_SQ:
        symbol, own = _BINARY.get(op, ("**", _POWER))
        left, left_precedence = _printed(e.dep(0))
        right, right_precedence = (
            ("2", _ATOM) if op == cas.OP_SQ else _printed(e.dep(1))
        )
        if own == _POWER:
            left = _wrap(left, left_precedence <= _POWER)
            right = _wrap(right, right_precedence < _POWER)
        else:
            left = _wrap(left, left_precedence < own)
            right = _wrap(right, right_precedence <= own)
        return f"{left} {symbol} {right}", own

    # Unary operators and functions
    if op == cas.OP_NEG:
        operand, precedence = _printed(e.dep(0))
        return f"-{_wrap(operand, precedence <= _UNARY)}", _UNARY
    if op == cas.OP_TWICE:
        operand, precedence = _printed(e.dep(0))
        return f"2 * {_wrap(operand, precedence <= _PRODUCT)}", _PRODUCT
    if op == cas.OP_INV:
        operand, precedence = _printed(e.dep(0))
        return f"1 / {_wrap(operand, precedence <= _PRODUCT)}", _PRODUCT
    if op in _FUNCTIONS:
        arguments = ", ".join(_printed(e.dep(i))[0] for i in range(e.n_dep()))
        return f"cas.{_FUNCTIONS[op]}({arguments})", _ATOM
    msg = f"no Python for the CasADi operation {op} in {e}"
    raise ValueError(msg)


def _wrap(text: str, needed: bool) -> str:
    """`text` in parentheses if `needed`."""
    return f"({text})" if needed else text


def _literal(value: float) -> str:
    """A finite number as Python: an integer without `.0`, otherwise its repr, which
    reads back as the same float.

    Raises `ValueError` if it isn't finite.
    """
    value = float(value)  # a NumPy float's repr names its type
    if not np.isfinite(value):
        msg = f"no Python literal for {value}"
        raise ValueError(msg)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


# The generated experiment


def variable_names(source: Source) -> tuple[str, ...]:
    """S2MPJ's names of the variables, which the generated file uses for its `Var`s
    and its Python variables.

    Raises `ValueError` if one can't name both.
    """
    for name in source.variable_names:
        is_usable = (
            name.isidentifier()
            and not keyword.iskeyword(name)
            and not name.startswith("_")
            and name not in _FILE_NAMES
            and not hasattr(builtins, name)
        )
        if not is_usable:
            msg = f"S2MPJ's variable name {name!r} can't name a variable in the file"
            raise ValueError(msg)
    return source.variable_names


def experiment_text(source: Source, chosen: Split, origin: Origin) -> str:
    """The generated experiment file of the problem: `TEMPLATE` filled in, before
    formatting.

    Raises `ValueError` if a name or an expression can't be written as Python.
    """
    names = variable_names(source)
    d = len(chosen.decisions)

    # The equations in the variables' names
    v = cas.vertcat(*(cas.SX.sym(name) for name in names))
    g = known_equations(source, v, cas.SX.sym("y"), chosen.hidden)
    g_lines = "\n".join(
        f"        {to_python(g[j])},  # {source.constraint_names[j]}"
        for j in range(source.m)
    )
    h = to_python(source.element_sums(v)[chosen.hidden])
    f = to_python(source.f(v)[0])

    # The variables: decision variables within the bounds, mechanistic states with
    # S2MPJ's bounds and the rules' bounds as start bounds
    def var(i: int) -> str:
        """The `hb.Var` of variable i, as the rules bound it."""
        lower, upper = _literal(chosen.lower[i]), _literal(chosen.upper[i])
        if i in chosen.decisions:
            arguments = [f"lower={lower}", f"upper={upper}"]
        else:
            s2mpj_bounds = (("lower", source.lower[i]), ("upper", source.upper[i]))
            arguments = [
                f"{side}={_literal(bound)}"
                for side, bound in s2mpj_bounds
                if np.isfinite(bound)
            ] + [f"start=({lower}, {upper})"]
        return f'{names[i]} = hb.Var("{names[i]}", {", ".join(arguments)})'

    tau = temperature(origin.optimum)
    uses_casadi = "cas." in g_lines + h + f
    return TEMPLATE.substitute(
        name=source.name,
        module=source.name.lower(),
        version=GENERATOR_VERSION,
        commit=origin.commit,
        constraint=source.constraint_names[chosen.hidden],
        decimals=SOLUTION_DECIMALS,
        points="\n".join(
            f"  {names[i]}: start {_literal(origin.start[i])}, "
            f"v* {_literal(origin.solution[i])}"
            for i in range(source.n)
        ),
        stated_optimum=origin.stated_optimum,
        ipopt_optimum=f"{origin.ipopt_optimum:.6g}",
        min_temperature=_literal(MIN_TEMPERATURE),
        temperature=_literal(tau),
        casadi_import="import casadi as cas\n\n" if uses_casadi else "",
        variables="\n".join(var(i) for i in (*chosen.decisions, *chosen.states)),
        y_start=f"{_literal(chosen.y_start[0])}, {_literal(chosen.y_start[1])}",
        g_lines=g_lines,
        h=h,
        f=f,
        u=", ".join(names[i] for i in chosen.decisions),
        x=", ".join(names[i] for i in chosen.states),
        h_inputs=", ".join(names[i] for i in chosen.h_inputs),
        optimum=_literal(origin.optimum),
        n_scenarios=25 * d,
        n_evaluations=10 * d,
        n_initial=2 * d + 1,
    )


def formatted(text: str) -> str:
    """Python source as `ruff format` writes it, with the repository's settings.

    Raises `FileNotFoundError` without ruff, `subprocess.CalledProcessError` if it
    fails.
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        msg = "ruff isn't installed: run the generator in the benchmarks environment"
        raise FileNotFoundError(msg)
    result = subprocess.run(
        [ruff, "format", "--stdin-filename", str(OUTPUT_DIR / "generated.py"), "-"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.stdout


def load_experiment(path: Path) -> types.ModuleType:
    """The generated experiment at `path`, imported as a module (not run)."""
    spec = importlib.util.spec_from_file_location(f"generated_{path.stem}", path)
    if spec is None or spec.loader is None:
        msg = f"can't load {path}"
        raise ValueError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_experiment(experiment: types.ModuleType, source: Source) -> None:
    """Checks the generated experiment's f, g and h against S2MPJ's float evaluation
    at random points within its bounds (the decision variables' bounds and the
    mechanistic states' start bounds): f, and g at y = h(x, u), which is c.

    Raises `UnsupportedProblemError` (`Reason.UNTRACEABLE`) where they differ.
    """
    system = experiment.problem.system
    index = {name: i for i, name in enumerate(variable_names(source))}
    u_columns = [index[var.name] for var in system.u]
    x_columns = [index[var.name] for var in system.x]

    # Points within the bounds, in S2MPJ's order
    lower, upper = np.zeros(source.n), np.zeros(source.n)
    for var in system.u:
        lower[index[var.name]], upper[index[var.name]] = var.lower, var.upper
    for var in system.x:
        lower[index[var.name]], upper[index[var.name]] = var.start
    points = lhs(lower, upper, N_TRACE_POINTS, rng_of(source.name))

    # f and g there, with y from h
    for point in points:
        u, x = point[u_columns], point[x_columns]
        y = experiment.measure.oracle(x, 0.0, u)
        values = np.concatenate(
            [
                experiment.problem.f(x, y, u).full().ravel(),
                system.g(x, y, u).full().ravel(),
            ]
        )
        _check_agrees(source, point, values, "the generated file")


def outcome(name: str) -> Outcome:
    """What the rules make of the problem `name`: its generated experiment's text,
    checked against S2MPJ, or why it is refused.

    Raises `FileNotFoundError` without the checkout and `ValueError` if S2MPJ has
    no such problem.
    """
    # The problem, and its split
    try:
        source = read(name)
    except UnsupportedProblemError as error:
        return Outcome(name=name, n=None, m=None, error=error)
    try:
        chosen, origin = split(source)
        text = formatted(experiment_text(source, chosen, origin))
    except UnsupportedProblemError as error:
        return Outcome(name=name, n=source.n, m=source.m, error=error)
    except ValueError as error:  # a name or an expression it can't write
        error = UnsupportedProblemError(name, Reason.UNTRACEABLE, str(error))
        return Outcome(name=name, n=source.n, m=source.m, error=error)

    # The text checked as a module from a temporary folder
    with tempfile.TemporaryDirectory() as folder:
        checked = Path(folder) / f"{name.lower()}.py"
        checked.write_text(text, encoding="utf-8", newline="\n")
        try:
            check_experiment(load_experiment(checked), source)
        except UnsupportedProblemError as error:
            return Outcome(name=name, n=source.n, m=source.m, error=error)
    hidden = source.constraint_names[chosen.hidden]
    return Outcome(name=name, n=source.n, m=source.m, hidden=hidden, text=text)


def generate(name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Writes the generated experiment of the S2MPJ problem `name` to `output_dir`
    and returns its path.

    Raises what `outcome` raises, and its `UnsupportedProblemError` if the problem
    is refused, writing nothing.
    """
    result = outcome(name)
    if result.error is not None:
        raise result.error
    return _write(result, output_dir)


def _write(result: Outcome, output_dir: Path) -> Path:
    """Writes the generated experiment of a problem the rules accept, and returns its
    path."""
    path = output_dir / f"{result.name.lower()}.py"
    path.write_text(str(result.text), encoding="utf-8", newline="\n")
    return path


# The suite


def in_scope() -> list[str]:
    """The S2MPJ problems in the suite's scope: an objective, equality constraints
    only (at least one) and at most `MAX_VARIABLES` variables, as S2MPJ builds them
    by default. A problem whose classification says so but that doesn't load is
    kept, refused by the rules."""
    names = []
    for name in problem_names():
        # The classification first, which is cheap: fixed sizes, few variables and
        # some constraints
        try:
            n, m = classified_sizes(name)
        except ValueError:
            continue
        is_classified_in_scope = (
            n.isdigit() and int(n) <= MAX_VARIABLES and m.isdigit() and int(m) > 0
        )
        if not is_classified_in_scope:
            continue

        # Then the problem as S2MPJ builds it
        try:
            n_built, m_built, n_inequalities, has_objective = sizes(name)
        except UnsupportedProblemError:
            names.append(name)
            continue
        is_in_scope = (
            n_built <= MAX_VARIABLES
            and m_built
            and not n_inequalities
            and has_objective
        )
        if is_in_scope:
            names.append(name)
    return names


def suite_table(outcomes: Sequence[Outcome]) -> str:
    """SUITE.md: every in-scope problem, generated or refused, with the reason."""
    # One row per problem, with `|` escaped in the details, then the page
    rows = []
    for result in outcomes:
        n = "?" if result.n is None else str(result.n)
        m = "?" if result.m is None else str(result.m)
        if result.error is None:
            module = result.name.lower()
            what = f"generated: [{module}.py]({module}.py)"
            detail = f"hides {result.hidden}"
        else:
            what = f"refused: {result.error.reason}"
            detail = result.error.detail.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {result.name} | {n} | {m} | {what} | {detail} |")
    n_generated = sum(result.error is None for result in outcomes)
    return (
        f"""# The S2MPJ problem suite

Every S2MPJ problem with an objective, equality constraints only and at most
{MAX_VARIABLES} variables, at S2MPJ commit `{COMMIT}`.
{n_generated} of the {len(outcomes)} are generated as benchmark experiments here; the
rules in `benchmarks/generate_s2mpj.py` (version {GENERATOR_VERSION}) refuse the rest,
with the reason. #181 tracks lifting the reasons. Don't edit this file: regenerate it
with `pixi run -e benchmarks generate-s2mpj`.

| problem | n | m | outcome | detail |
|---|---|---|---|---|
"""
        + "\n".join(rows)
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generates the named problems, or every in-scope one, and SUITE.md; returns the
    exit status, 1 if a named problem was refused. Generating every one also removes
    the files of problems the rules now refuse, so the folder matches SUITE.md."""
    parser = argparse.ArgumentParser(
        prog="python generate_s2mpj.py",
        description="Generate benchmark experiments from S2MPJ problems, and "
        "SUITE.md, the table of every in-scope problem.",
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="S2MPJ problem names, in any case (default: every in-scope one)",
    )
    requested = parser.parse_args(argv).names

    # The names as S2MPJ spells them, as the files are lowercase
    spelled = {name.lower(): name for name in problem_names()}
    names = [spelled.get(name.lower(), name) for name in requested]

    # What the rules make of every in-scope problem, for SUITE.md, and of any named
    # one outside the scope
    in_scope_outcomes = [outcome(name) for name in in_scope()]
    outcomes = {result.name: result for result in in_scope_outcomes}
    status = 0
    for name in names:
        try:
            outcomes.setdefault(name, outcome(name))
        except ValueError as error:
            print(error, file=sys.stderr)
            status = 1
    to_write = [
        result
        for name, result in outcomes.items()
        if (name in names or not names) and result.error is None
    ]

    # The files of problems the rules no longer generate go, when generating all
    if not names:
        kept = {f"{result.name.lower()}.py" for result in to_write}
        for path in OUTPUT_DIR.glob("*.py"):
            if path.name not in kept:
                path.unlink()
                print(f"Removed {path}")

    # The experiments, each checked already, and the refused named problems
    for result in to_write:
        print(f"Generated {_write(result, OUTPUT_DIR)}")
    for name in names:
        error = outcomes[name].error if name in outcomes else None
        if error is not None:
            print(error, file=sys.stderr)
            status = 1
    suite = OUTPUT_DIR / "SUITE.md"
    suite.write_text(suite_table(in_scope_outcomes), encoding="utf-8", newline="\n")
    print(f"Wrote {suite}")
    return status


if __name__ == "__main__":
    sys.exit(main())
