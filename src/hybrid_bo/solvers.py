"""Nonlinear programs and the multistart solver.

An `NLP` describes the problem, `Ipopt` the solver settings, and `multistart` solves the
problem from several starts and keeps the best solution. The caller draws the starts
(with `lhs`), so the draw order stays visible where the generator is used.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.errors import NoSolutionError

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLP:
    """min f(w) subject to g(w) = 0 and lower <= w <= upper.

    `f` and `g` are CasADi functions (`cas.Function`) of the column w; the bounds have
    shape (n,) and may be infinite.
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


def multistart(nlp: NLP, starts: np.ndarray, solver: Ipopt) -> Solution:
    """Solves `nlp` from each row of `starts` (n_starts, n) and returns the solution
    with the lowest objective.

    A start that fails is expected, so it is logged (at debug level, without a
    traceback) and skipped; so are the number that failed and the time taken, under
    the name of `nlp.f`. Starts that ran IPOPT to its iteration limit are also
    counted in one warning. Raises `NoSolutionError` if every start fails.
    """
    # The problem, with the bounds as constraints (as the pre-refactor optimizer had
    # them, so IPOPT takes the steps the regression baselines record)
    opti = cas.Opti()
    w = opti.variable(len(nlp.lower))
    opti.minimize(nlp.f(w))
    if nlp.g is not None:
        opti.subject_to(nlp.g(w) == 0)
    opti.subject_to(w >= nlp.lower[:, np.newaxis])
    opti.subject_to(w <= nlp.upper[:, np.newaxis])

    # Solve from each start; a NaN objective never wins
    started = time.perf_counter()
    best, f_best, n_failed, n_at_limit = None, np.inf, 0, 0
    for i, start in enumerate(starts):
        opti.set_initial(w, start)
        opti.solver("ipopt", solver.plugin_options, solver.solver_options)
        try:
            solution = opti.solve()
        except RuntimeError as error:
            _logger.debug("Start %d of %d failed: %s", i + 1, len(starts), error)
            n_failed += 1
            at_limit = opti.stats()["return_status"] == "Maximum_Iterations_Exceeded"
            if at_limit:
                n_at_limit += 1
            continue
        f = float(solution.value(opti.f))
        if f < f_best:
            best, f_best = np.atleast_1d(solution.value(w)).astype(float), f
    _logger.debug(
        "%s: %d of %d starts failed, %.2fs",
        nlp.f.name(),
        n_failed,
        len(starts),
        time.perf_counter() - started,
    )
    if n_at_limit:
        _logger.warning(
            "%s: %d of %d starts stopped at IPOPT's iteration limit",
            nlp.f.name(),
            n_at_limit,
            len(starts),
        )

    if best is None:
        msg = f"Every one of the {len(starts)} starts failed"
        raise NoSolutionError(msg)
    return Solution(best, f_best)
