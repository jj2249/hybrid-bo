"""Figures of a solved `HybridNLP`, drawn on the axes you give them."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.errors import RootFindingError
from hybrid_bo.hybrid_problem import SAA, HybridNLP, Solution
from hybrid_bo.obj import Mean
from hybrid_bo.solvers import RootProblem
from hybrid_bo.utils import Symbolic

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Each grid point's scenario states solve the constraints to this
_GRID_TOLERANCE = 1e-8


def solution(nlp: HybridNLP, solution: Solution, axes: "Sequence[Axes]") -> None:
    """Draws what `solution` minimized over the one decision variable: the surrogate,
    f under each scenario with their weighted mean, and the objective unless it is
    `Mean`, one panel each, with the minimizer marked.

    Raises `ValueError` unless u has one variable and there is one axes per panel.
    """
    _draw(nlp, solution, axes, label="minimizer")


def iteration(
    nlp: HybridNLP, solution: Solution, axes: "Sequence[Axes]", data: Dataset
) -> None:
    """Draws one BO iteration as `solution` does, with the points measured so far
    `data` and their incumbent on the f panel, and the trial point marked.

    Raises `ValueError` unless u has one variable and there is one axes per panel.
    """
    _draw(nlp, solution, axes, label="trial point")
    axes[1].scatter(data.u[:, 0], data.f, color="k", s=12, label="measured")
    axes[1].axhline(data.incumbents[-1], color="k", linestyle="--", label="incumbent")
    axes[1].legend(fontsize="small")


def _draw(
    nlp: HybridNLP, solution: Solution, axes: "Sequence[Axes]", *, label: str
) -> None:
    """Draws the panels of `solution` over a grid of u on the solver's branch.

    Raises `ValueError` unless u has one variable and there is one axes per panel.
    """
    n_panels = 2 if isinstance(solution.objective, Mean) else 3
    if len(nlp.u) != 1 or len(axes) != n_panels:
        msg = f"This figure draws one u on {n_panels} axes, got {len(nlp.u)} u and {len(axes)} axes"
        raise ValueError(msg)
    problem, scenarios = nlp._nlp(
        solution.method, solution.surrogate, solution.objective, True
    )

    # With u fixed, SAA's constraints are a square system in the scenario states
    (var,) = nlp.u
    stored = np.zeros(0)
    root_problem = None
    if isinstance(solution.method, SAA):
        stored = np.concatenate([solution.x.ravel("F"), solution.y.ravel("F")])
        states, u_fixed = Symbolic.sym("states", len(stored)), Symbolic.sym("u", 1)
        assert problem.g is not None
        root_problem = RootProblem(
            cas.Function(
                "constraints",
                [states, u_fixed],
                [problem.g(cas.vertcat(u_fixed, states))],
            ),
            problem.lower[1:],
            problem.upper[1:],
        )

    # Outwards from the solution, each grid point from the nearest one solved
    u = np.linspace(*var.start, 201)
    n_scenarios, n_y = solution.y.shape
    objective = np.full(len(u), np.nan)
    y = np.full((len(u), n_scenarios, n_y), np.nan)
    f = np.full((len(u), n_scenarios), np.nan)
    solved_u, solved_states = [solution.u[0]], [stored]
    for i in np.argsort(np.abs(u - solution.u[0])):
        start = solved_states[np.argmin(np.abs(np.array(solved_u) - u[i]))]
        try:
            root = (
                root_problem.solve(u[i : i + 1], start[np.newaxis], _GRID_TOLERANCE)
                if root_problem
                else stored
            )
        except RootFindingError:
            continue
        w = np.concatenate([u[i : i + 1], root])
        objective[i] = float(problem.f(w))
        _, y_i, f_i = scenarios(w)
        y[i], f[i] = y_i.full(), f_i.full().ravel()
        solved_u.append(u[i])
        solved_states.append(root)

    # The surrogate: a GP of u alone as its band, else its scenarios on the branch
    for fitted in solution.surrogate:
        j = nlp.y.index(fitted.output)
        if fitted.inputs == (var,):
            mean, std = fitted.predict(u[:, np.newaxis])
            axes[0].plot(u, mean, color=f"C{j}", label=fitted.output.name)
            axes[0].fill_between(
                u, mean - 2 * std, mean + 2 * std, color=f"C{j}", alpha=0.2
            )
            axes[0].scatter(
                fitted.x_scaler.unscale(fitted.X),
                fitted.y_scaler.unscale(fitted.Y[:, np.newaxis]),
                color=f"C{j}",
                s=12,
            )
        else:
            axes[0].plot(u, y[:, :, j], color=f"C{j}", alpha=0.3)
    axes[0].set_ylabel("surrogate")

    # f under each scenario and their weighted mean, then the objective
    w = solution.method.w if isinstance(solution.method, SAA) else np.ones(1)
    if n_scenarios > 1:
        axes[1].plot(u, f, color="C0", alpha=0.2, linewidth=0.5)
    axes[1].plot(u, f @ w, color="C0", label="E[f]")
    axes[1].set_ylabel("f")
    if n_panels == 3:
        axes[2].plot(u, objective, color="C0")
        axes[2].set_ylabel(f"objective ({type(solution.objective).__name__})")
    axes[-1].set_xlabel(var.name)
    for ax in axes:
        ax.axvline(solution.u[0], color="k", linestyle=":", label=label)
        ax.legend(fontsize="small")
