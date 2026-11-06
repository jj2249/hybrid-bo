"""Nonlinear programs, the multistart solver and the root finder.

An `NLP` describes the problem, `Ipopt` the solver settings, and `solve_with_multistart`
builds the CasADi `Opti` problem once, solves it from several starts and keeps the best
solution. `_find_root` is every fsolve, of the root-finding problem and of the oracle
simulation. The caller draws the starts (with `lhs`), so the draw order stays visible
where the generator is used.
"""

import logging
import signal
import threading
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FrameType
from typing import Any

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.errors import NoSolutionError, RootFindingError
from hybrid_bo.problem import Var

_logger = logging.getLogger(__name__)

# Objectives closer than this (relative, or absolute below 1) tie in
# `solve_with_multistart`
_TIE_TOLERANCE = 1e-13


@dataclass(frozen=True)
class NLP:
    """min f(w) subject to g(w) = 0 and lower <= w <= upper.

    `f` and `g` are CasADi functions (`cas.Function`) of the column w; the bounds have
    shape (n,) and may be infinite. A `g` of no rows is no constraint.
    """

    f: cas.Function
    lower: np.ndarray
    upper: np.ndarray
    g: cas.Function | None = None


@dataclass(frozen=True)
class Ipopt:
    """IPOPT through CasADi: options for the CasADi plugin and for IPOPT itself."""

    plugin_options: dict[str, Any] = field(default_factory=lambda: {"print_time": 0})
    solver_options: dict[str, Any] = field(default_factory=lambda: {"print_level": 0})


@dataclass(frozen=True)
class Solution:
    """A solution of an NLP: the variables w (n,) and the objective there."""

    w: np.ndarray
    f: float


def solve_with_multistart(nlp: NLP, starts: np.ndarray, solver: Ipopt) -> Solution:
    """Builds `nlp` as a CasADi `Opti` problem, solves it with IPOPT from each row of
    `starts` (n_starts, n), and returns the solution with the lowest objective.
    Objectives that differ only by rounding tie, and the earlier start wins, so the
    result doesn't depend on the platform's arithmetic.

    A start that fails is expected, so it is skipped. Starts that ran IPOPT to its
    iteration limit are counted in one warning. Raises `NoSolutionError` if every start fails, and
    `KeyboardInterrupt` on Ctrl-C, even while IPOPT runs.
    """
    # Build the problem once, with the bounds as constraints (as the pre-refactor
    # optimizer had them, so IPOPT takes the steps the regression baselines record)
    opti = cas.Opti()
    w = opti.variable(len(nlp.lower))
    opti.minimize(nlp.f(w))
    # CasADi refuses an empty constraint, so a g of no rows is left out
    has_equations = nlp.g is not None and nlp.g.numel_out(0) > 0
    if has_equations:
        opti.subject_to(nlp.g(w) == 0)
    opti.subject_to(w >= nlp.lower[:, np.newaxis])
    opti.subject_to(w <= nlp.upper[:, np.newaxis])

    # Solve it from each start; a NaN objective never wins, nor does a tie with the best
    best, f_best, n_solved, n_at_limit = None, np.inf, 0, 0
    for i, start in enumerate(starts):
        opti.set_initial(w, start)
        opti.solver("ipopt", solver.plugin_options, solver.solver_options)
        try:
            with _interrupts_noticed() as interrupts:
                solution = opti.solve()
        except RuntimeError as error:
            # CasADi catches a Ctrl-C inside IPOPT and fails the solve instead
            if interrupts:
                raise KeyboardInterrupt from error
            at_limit = opti.stats()["return_status"] == "Maximum_Iterations_Exceeded"
            if at_limit:
                n_at_limit += 1
            continue
        n_solved += 1
        f = float(solution.value(opti.f))
        ties_best = best is not None and f >= f_best - _TIE_TOLERANCE * max(
            1.0, abs(f_best)
        )
        if f < f_best and not ties_best:
            best, f_best = np.atleast_1d(solution.value(w)).astype(float), f
    if n_at_limit:
        _logger.warning(
            "%s: %d of %d starts stopped at IPOPT's iteration limit",
            nlp.f.name(),
            n_at_limit,
            len(starts),
        )
    _logger.info(
        "%s: %d of %d starts solved, best %.6g",
        nlp.f.name(),
        n_solved,
        len(starts),
        f_best,
    )

    if best is None:
        msg = f"Every one of the {len(starts)} starts failed"
        raise NoSolutionError(msg)
    return Solution(best, f_best)


@contextmanager
def _interrupts_noticed() -> Generator[list[int]]:
    """Yields a list that gets an entry for each Ctrl-C (SIGINT) during the block,
    which is still handled as before. Only the main thread handles signals, and only
    a Python handler can be wrapped, so otherwise the list stays empty."""
    previous = signal.getsignal(signal.SIGINT)
    interrupts = []
    is_main_thread = threading.current_thread() is threading.main_thread()
    if not (is_main_thread and callable(previous)):
        yield interrupts
        return

    def noticing(signum: int, frame: FrameType | None) -> None:
        """Records the interrupt, then handles it as the previous handler did."""
        interrupts.append(signum)
        previous(signum, frame)

    signal.signal(signal.SIGINT, noticing)
    try:
        yield interrupts
    finally:
        signal.signal(signal.SIGINT, previous)


def _find_root(
    w: Any, residuals: Any, variables: Sequence[Var], starts: np.ndarray, tol: float
) -> np.ndarray:
    """A root of `residuals`, a column expression in the column of symbols `w`.

    Runs fsolve with the CasADi Jacobian from each row of `starts` in turn and returns
    the first root, shape (d,), whose residual norm is below `tol` and that lies within
    the bounds of `variables` (one per entry of `w`). A start that fails is expected,
    so fsolve's convergence warnings are silenced. With no variables (no unmeasured
    states, as without known equations) the root is empty and nothing is solved.

    Raises `RootFindingError` if no start gives such a root.
    """
    if not variables:
        return np.zeros(0)

    # Numeric residual and Jacobian, and the box a root must lie in
    residual = cas.Function("residual", [w], [residuals])
    jacobian = cas.Function("jacobian", [w], [cas.jacobian(residuals, w)])
    lower = np.array([var.lower for var in variables])
    upper = np.array([var.upper for var in variables])

    # The first start that converges to a root within the bounds wins
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            root = scipy.optimize.fsolve(
                lambda w: residual(w[:, np.newaxis]).full().ravel(),
                start,
                fprime=lambda w: jacobian(w[:, np.newaxis]).full(),
            )
        is_root = np.linalg.norm(residual(root[:, np.newaxis]).full()) < tol
        if is_root and np.all(lower <= root) and np.all(root <= upper):
            return root

    names = ", ".join(var.name for var in variables)
    msg = f"No root for {names} within the bounds from {len(starts)} starts"
    raise RootFindingError(msg)
