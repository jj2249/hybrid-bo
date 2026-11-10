"""The solvers: multistart IPOPT for an `NLP`, and Newton's method for a `RootProblem`."""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.errors import NoSolutionError, RootFindingError
from hybrid_bo.utils import lhs

_logger = logging.getLogger(__name__)

# Objectives closer than this (relative, or absolute below 1) tie in `NLP.solve_from`
_TIE_TOLERANCE = 1e-13

# Newton's method converges in a few iterations or not at all, and a start that doesn't
# would otherwise run CasADi's default of 1000
_NEWTON_MAX_ITERATIONS = 100


@dataclass(frozen=True)
class Ipopt:
    """Options for CasADi's IPOPT plugin and for IPOPT itself."""

    plugin_options: dict[str, Any] = field(default_factory=lambda: {"print_time": 0})
    solver_options: dict[str, Any] = field(default_factory=lambda: {"print_level": 0})


@dataclass(frozen=True)
class NLP:
    """min f(w) subject to g(w) = 0 and lower <= w <= upper, bounds (n,), with
    multistart starts drawn from `start_bounds`, by default the bounds."""

    f: cas.Function
    lower: np.ndarray
    upper: np.ndarray
    g: cas.Function | None = None
    start_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def solve(
        self,
        rng: np.random.Generator,
        *,
        n_starts: int = 10,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
    ) -> tuple[np.ndarray, float]:
        """The best minimizer w (n,) and its objective from `n_starts` Latin hypercube
        starts. Raises `NoSolutionError` if every start fails."""
        starts = lhs(*(self.start_bounds or (self.lower, self.upper)), n_starts, rng)
        return self.solve_from(starts, solver)

    def solve_from(
        self,
        starts: np.ndarray,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
    ) -> tuple[np.ndarray, float]:
        """The best minimizer w (n,) and its objective from the rows of `starts`
        (n_starts, n); on a tie the earlier start wins. Raises `NoSolutionError` if every
        start fails."""
        # Build the problem once, with the bounds as constraints, so IPOPT takes the steps
        # the regression baselines record
        opti = cas.Opti()
        w = opti.variable(len(self.lower))
        opti.minimize(self.f(w))
        # CasADi refuses an empty constraint, so a g of no rows is left out
        has_equations = self.g is not None and self.g.numel_out(0) > 0
        if has_equations:
            opti.subject_to(self.g(w) == 0)
        opti.subject_to(w >= self.lower[:, np.newaxis])
        opti.subject_to(w <= self.upper[:, np.newaxis])

        # Solve it from each start; a NaN objective never wins, nor does a tie with the
        # best
        best, f_best, n_solved, n_at_limit = None, np.inf, 0, 0
        for start in starts:
            opti.set_initial(w, start)
            opti.solver("ipopt", solver.plugin_options, solver.solver_options)
            try:
                solution = opti.solve()
            except RuntimeError:
                status = opti.stats()["return_status"]
                if status == "Maximum_Iterations_Exceeded":
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
                self.f.name(),
                n_at_limit,
                len(starts),
            )
        _logger.info(
            "%s: %d of %d starts solved, best %.6g",
            self.f.name(),
            n_solved,
            len(starts),
            f_best,
        )

        if best is None:
            msg = f"Every one of the {len(starts)} starts failed"
            raise NoSolutionError(msg)
        return best, f_best


@dataclass(frozen=True)
class RootProblem:
    """residual(w, p) = 0 for w within lower <= w <= upper, bounds (n,), at the
    parameters p."""

    residual: cas.Function
    lower: np.ndarray
    upper: np.ndarray

    @cached_property
    def _newton(self) -> cas.Function:
        """Newton's method on the residual: (start, p) -> w. Built once, on first use."""
        # `solve` checks every result itself, and a start that diverges is expected
        options = {
            "error_on_fail": False,
            "max_iter": _NEWTON_MAX_ITERATIONS,
            "show_eval_warnings": False,
        }
        return cas.rootfinder("newton", "newton", self.residual, options)

    def solve(self, p: np.ndarray, starts: np.ndarray, tol: float) -> np.ndarray:
        """The first root w (n,) within the bounds and below `tol`, at the parameters p
        (k,), by Newton's method from the rows of `starts` (n_starts, n).

        Raises `RootFindingError` if no start gives one or the equations are singular.
        """
        if len(self.lower) == 0:
            return np.zeros(0)
        try:
            newton = self._newton
        except RuntimeError as error:
            msg = "The equations can't determine the variables: they are singular"
            raise RootFindingError(msg) from error

        # The first start that converges to a root within the bounds wins
        for start in starts:
            root = newton(start, p).full().ravel()
            is_root = np.linalg.norm(self.residual(root, p).full()) < tol
            if is_root and np.all(self.lower <= root) and np.all(root <= self.upper):
                return root

        msg = f"No root within the bounds from {len(starts)} starts"
        raise RootFindingError(msg)
