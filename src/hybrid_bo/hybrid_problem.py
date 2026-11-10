"""The hybrid NLP: min f(x, y, u) subject to 0 = g(x, y, u), with y from fitted GPs,
solved by a sample average over scenarios (`SAA`) or in closed form (`Exact`)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.gp import FittedGP
from hybrid_bo.obj import EI, Mean, SoftEI
from hybrid_bo.solvers import NLP, Ipopt
from hybrid_bo.utils import Symbolic
from hybrid_bo.variables import Var, start_bounds, to_casadi_function


@dataclass(frozen=True, init=False)
class SAA:
    """The sample average over the draws z (S, P), a row per scenario and a column per
    unknown state, with weights w (S,), by default 1/S.

    Raises `ValueError` unless w has S entries that are >= 0 and sum to 1.
    """

    z: np.ndarray
    w: np.ndarray

    def __init__(self, z: np.ndarray, w: np.ndarray | None = None) -> None:
        z = np.asarray(z, dtype=float)
        w = np.full(len(z), 1 / len(z)) if w is None else np.asarray(w, dtype=float)
        if z.ndim != 2 or w.shape != (len(z),):
            msg = f"SAA needs z (S, P) and w (S,), got {z.shape} and {w.shape}"
            raise ValueError(msg)
        if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            msg = f"SAA weights must be >= 0 and sum to 1, got {w}"
            raise ValueError(msg)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "w", w)


@dataclass(frozen=True)
class Exact:
    """The objective in closed form from the GP's mean and std. Valid only when f is
    one unknown state and there are no mechanistic states: standard BO."""


@dataclass(frozen=True)
class Solution:
    """The minimizer u (N,), the states there x (S, M) and y (S, P), a row per scenario
    (S = 1 for `Exact`), and the minimized value, with the method, surrogate and
    objective that `solve` minimized it by."""

    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    value: float
    method: SAA | Exact
    surrogate: tuple[FittedGP, ...] = field(repr=False)
    objective: Mean | EI | SoftEI


@dataclass(frozen=True, init=False)
class HybridNLP:
    """What you know: the decision variables u, the mechanistic states x with their
    known equations g, the unknown states y, and the objective f.

    `g` and `f` are callables of `v`, read as `v.<name>`, traced once by CasADi. Raises
    `ValueError` for a repeated name, a g without one equation per state of x, or a
    non-scalar f.
    """

    u: tuple[Var, ...]
    x: tuple[Var, ...]
    y: tuple[Var, ...]
    g: cas.Function = field(repr=False)
    f: cas.Function = field(repr=False)

    def __init__(
        self,
        *,
        u: Sequence[Var],
        x: Sequence[Var] = (),
        y: Sequence[Var],
        g: Callable[[Any], Any] | None = None,
        f: Callable[[Any], Any],
    ) -> None:
        u, x, y = tuple(u), tuple(x), tuple(y)
        names = [var.name for var in (*u, *x, *y)]
        if len(set(names)) != len(names):
            msg = f"Variable names must be unique, got {names}"
            raise ValueError(msg)
        g_function = to_casadi_function("g", g or (lambda v: []), x, y, u)
        if g_function.size_out(0) != (len(x), 1):
            msg = f"g must give one equation per state of x ({len(x)}), got {g_function.size_out(0)}"
            raise ValueError(msg)
        f_function = to_casadi_function("f", f, x, y, u)
        if f_function.size_out(0) != (1, 1):
            msg = f"f must be a scalar, got shape {f_function.size_out(0)}"
            raise ValueError(msg)

        # Frozen dataclass: set fields through object.__setattr__
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "g", g_function)
        object.__setattr__(self, "f", f_function)

    def solve(
        self,
        method: SAA | Exact,
        rng: np.random.Generator,
        *,
        surrogate: Sequence[FittedGP],
        objective: Mean | EI | SoftEI = Mean(),  # noqa: B008  (frozen)
        n_starts: int = 10,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
        unknown_states_as_variables: bool = True,
    ) -> Solution:
        """The `Solution` minimizing `objective` with y from `surrogate`, one fitted GP
        per unknown state in any order, by `method`, from `n_starts` starts.

        With `unknown_states_as_variables`, SAA's y are variables tied to the GPs.
        Raises `ValueError` for a surrogate or method that doesn't fit the problem, and
        `NoSolutionError` if every start fails.
        """
        nlp, scenarios = self._nlp(
            method, surrogate, objective, unknown_states_as_variables
        )
        best, value = nlp.solve(rng, n_starts=n_starts, solver=solver)
        x, y, _ = (part.full() for part in scenarios(best))
        u = best[: len(self.u)]
        return Solution(u, x, y, value, method, tuple(surrogate), objective)

    def _nlp(
        self,
        method: SAA | Exact,
        surrogate: Sequence[FittedGP],
        objective: Mean | EI | SoftEI,
        unknown_states_as_variables: bool,
    ) -> tuple[NLP, cas.Function]:
        """The `NLP` that `solve` minimizes, over u and then, if they are variables,
        the scenario states, and `scenarios`: its decisions -> x (S, M), y (S, P) and
        f (S, 1).

        Raises `ValueError` for a surrogate or method that doesn't fit the problem.
        """
        gps = {gp.output: gp for gp in surrogate}
        if len(surrogate) != len(self.y) or set(gps) != set(self.y):
            msg = (
                f"The surrogate needs one GP per unknown state "
                f"{[var.name for var in self.y]}, got GPs of "
                f"{[gp.output.name for gp in surrogate]}"
            )
            raise ValueError(msg)
        if any(var not in (*self.u, *self.x) for gp in surrogate for var in gp.inputs):
            msg = "A GP's inputs must be decision variables or mechanistic states"
            raise ValueError(msg)
        n_u, n_x, n_y = len(self.u), len(self.x), len(self.y)
        u = Symbolic.sym("u", n_u)

        if isinstance(method, Exact):
            # f is one unknown state, whose GP gives its mean and std at u
            y = Symbolic.sym("y", n_y)
            f = self.f(Symbolic(0, 1), y, u)
            is_f = [bool(cas.is_equal(f, y[j])) for j in range(n_y)]
            if self.x or not any(is_f):
                msg = "Exact needs f to be one unknown state, and no states in x"
                raise ValueError(msg)
            columns = dict(zip(self.u, cas.vertsplit(u), strict=True))
            mean, std = zip(
                *(
                    gps[var].predict(
                        cas.horzcat(*(columns[v] for v in gps[var].inputs))
                    )
                    for var in self.y
                ),
                strict=True,
            )
            j = is_f.index(True)
            nlp = NLP(
                cas.Function("objective", [u], [objective.exact(mean[j], std[j])]),
                np.array([var.lower for var in self.u]),
                np.array([var.upper for var in self.u]),
                start_bounds=start_bounds(self.u),
            )
            outputs = [Symbolic(1, 0), cas.horzcat(*mean), mean[j]]
            return nlp, cas.Function("scenarios", [u], outputs)

        # SAA: each scenario s has its own states x_s and y_s, with g(x_s, y_s, u) = 0
        z, w = method.z, method.w
        n_scenarios = len(w)
        if z.shape[1] != n_y:
            msg = (
                f"SAA needs one column of z per unknown state ({n_y}), got {z.shape[1]}"
            )
            raise ValueError(msg)
        x = Symbolic.sym("x", n_scenarios, n_x)

        # Unknown state j of scenario s is its GP's prediction at the scenario's inputs,
        # mean_j + std_j * z_sj
        columns = {
            var: cas.repmat(u[i], n_scenarios, 1) for i, var in enumerate(self.u)
        }
        columns |= {var: x[:, j] for j, var in enumerate(self.x)}
        mean, std = zip(
            *(
                gps[var].predict(cas.horzcat(*(columns[v] for v in gps[var].inputs)))
                for var in self.y
            ),
            strict=True,
        )
        prediction = cas.horzcat(*mean) + cas.horzcat(*std) * Symbolic(z)
        y = (
            Symbolic.sym("y", n_scenarios, n_y)
            if unknown_states_as_variables
            else prediction
        )

        # Each scenario: its known equations, its tie to the GP, and its f
        constraints, f_parts = [], []
        for s in range(n_scenarios):
            x_s, y_s = x[s, :].T, y[s, :].T
            constraints.append(self.g(x_s, y_s, u))
            if unknown_states_as_variables:
                constraints.append((y - prediction)[s, :].T)
            f_parts.append(self.f(x_s, y_s, u))
        f = cas.vertcat(*f_parts)

        # The decision vector: u, then each state's S scenario copies
        blocks = [x, y] if unknown_states_as_variables else [x]
        decisions = cas.vertcat(u, *(cas.vec(block) for block in blocks))
        variables = [*self.u] + [
            var
            for block in ([self.x, self.y] if unknown_states_as_variables else [self.x])
            for var in block
            for _ in range(n_scenarios)
        ]
        nlp = NLP(
            cas.Function("objective", [decisions], [objective.saa(f, w)]),
            np.array([var.lower for var in variables]),
            np.array([var.upper for var in variables]),
            cas.Function("g", [decisions], [cas.vertcat(*constraints)]),
            start_bounds(variables),
        )
        return nlp, cas.Function("scenarios", [decisions], [x, y, f])
