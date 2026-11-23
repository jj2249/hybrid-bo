"""Figures of a solved `HybridNLP`, drawn on the axes you give them."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.errors import RootFindingError
from hybrid_bo.hybrid_problem import SAA, HybridNLP, Solution
from hybrid_bo.obj import EI, Mean
from hybrid_bo.solvers import RootProblem
from hybrid_bo.utils import Symbolic
from hybrid_bo.variables import start_bounds

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Each grid point's scenario states solve the constraints to this
_GRID_TOLERANCE = 1e-8
# Grid points along each u: a line of them for one u, a square for two
_GRID_POINTS = {1: 201, 2: 41}


def solution(nlp: HybridNLP, solution: Solution, axes: "Sequence[Axes]") -> None:
    """Draws what `solution` minimized over one or two decision variables, one panel
    each, with the minimizer marked. Over one u: the surrogate, f's weighted mean over
    the scenarios with a band of ±2 standard deviations, and the acquisition in units of
    f unless the objective is `Mean` (EI is -objective, SoftLogEI's is exp(-objective)).
    Over two u, as filled contours: the mean and standard deviation of each GP's output,
    then of f, then the acquisition.

    Raises `ValueError` unless u has one or two variables and there is one axes per
    panel.
    """
    _draw(nlp, solution, axes, label="minimizer")


def iteration(
    nlp: HybridNLP, solution: Solution, axes: "Sequence[Axes]", data: Dataset
) -> None:
    """Draws one BO iteration as `solution` does, with the trial point marked and the
    points measured so far `data`: over one u, with their incumbent on the f panel;
    over two u, on every panel, with the incumbent circled.

    Raises `ValueError` unless u has one or two variables and there is one axes per
    panel.
    """
    _draw(nlp, solution, axes, label="trial point")
    if len(nlp.u) == 1:
        axes[1].scatter(data.u[:, 0], data.f, color="k", s=12, label="measured")
        axes[1].axhline(
            data.incumbents[-1], color="k", linestyle="--", label="incumbent"
        )
        axes[1].legend(fontsize="small")
        return
    best = np.argmin(data.f)
    for ax in axes:
        ax.scatter(
            data.u[:, 0],
            data.u[:, 1],
            color="k",
            edgecolors="w",
            s=30,
            label="measured",
        )
        ax.scatter(
            *data.u[best],
            facecolors="none",
            edgecolors="C3",
            linewidths=2,
            s=150,
            label="incumbent",
        )
    axes[0].legend(fontsize="small")


def acquisition(nlp: HybridNLP, solution: Solution, ax: "Axes") -> None:
    """Draws the acquisition of `solution` in units of f on the one axes `ax`, as a line
    over one u or filled contours over two, with the trial point marked.

    Raises `TypeError` for a `Mean` objective, and `ValueError` unless u has one or two
    variables.
    """
    if isinstance(solution.objective, Mean):
        msg = "A Mean objective has no acquisition to draw"
        raise TypeError(msg)
    u, objective, *_ = _evaluate(nlp, solution)
    _draw_acquisition(nlp, solution, ax, u, objective, label="trial point")


def _draw(
    nlp: HybridNLP, solution: Solution, axes: "Sequence[Axes]", *, label: str
) -> None:
    """Draws the panels of `solution` over a grid of u on the solver's branch.

    Raises `ValueError` unless u has one or two variables and there is one axes per
    panel.
    """
    n_acquisition = 0 if isinstance(solution.objective, Mean) else 1
    n_panels = {1: 2, 2: 2 * len(solution.surrogate) + 2}.get(len(nlp.u), 0)
    if not n_panels or len(axes) != n_panels + n_acquisition:
        msg = (
            f"This figure draws one u on {2 + n_acquisition} axes, or two u on "
            f"{2 * len(solution.surrogate) + 2 + n_acquisition}, "
            f"got {len(nlp.u)} u and {len(axes)} axes"
        )
        raise ValueError(msg)
    u, objective, y_mean, y_std, f_mean, f_std = _evaluate(nlp, solution)

    if len(nlp.u) == 2:
        # The mean and std of each GP's output, then of f, then the acquisition
        panels = []
        for fitted in solution.surrogate:
            j = nlp.y.index(fitted.output)
            if fitted.inputs == tuple(nlp.u):
                mean, std = fitted.predict(u)
            else:
                mean, std = y_mean[:, j], y_std[:, j]
            panels += [
                (mean, f"E[{fitted.output.name}]"),
                (std, f"σ[{fitted.output.name}]"),
            ]
        panels += [(f_mean, "E[f]"), (f_std, "σ[f]")]
        for ax, (values, title) in zip(axes, panels, strict=False):
            _contour(nlp, solution, ax, u, values, title, label=label)
        if n_acquisition:
            _draw_acquisition(nlp, solution, axes[-1], u, objective, label=label)
        return

    # The surrogate: a GP of u alone as its band, else its scenarios' band on the branch
    (var,) = nlp.u
    for fitted in solution.surrogate:
        j = nlp.y.index(fitted.output)
        if fitted.inputs == (var,):
            mean, std = fitted.predict(u)
            axes[0].scatter(
                fitted.x_scaler.unscale(fitted.X),
                fitted.y_scaler.unscale(fitted.Y[:, np.newaxis]),
                color=f"C{j}",
                s=12,
            )
        else:
            mean, std = y_mean[:, j], y_std[:, j]
        axes[0].plot(u[:, 0], mean, color=f"C{j}", label=fitted.output.name)
        axes[0].fill_between(
            u[:, 0], mean - 2 * std, mean + 2 * std, color=f"C{j}", alpha=0.2
        )
    axes[0].set_ylabel("surrogate")

    # f's mean and band over the scenarios, then the acquisition
    axes[1].plot(u[:, 0], f_mean, color="C0", label="E[f]")
    axes[1].fill_between(
        u[:, 0], f_mean - 2 * f_std, f_mean + 2 * f_std, color="C0", alpha=0.2
    )
    axes[1].set_ylabel("f")
    if n_acquisition:
        _draw_acquisition(nlp, solution, axes[2], u, objective, label=label)
    for ax in axes[:2]:
        ax.axvline(solution.u[0], color="k", linestyle=":", label=label)
        ax.legend(fontsize="small")
    axes[-1].set_xlabel(var.name)


def _evaluate(
    nlp: HybridNLP, solution: Solution
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A grid u (n_points, d) over the start bounds of one or two u, and on it the
    objective (n_points,), the mean and std over the scenarios of y (n_points, P), and
    of f (n_points,), on the solver's branch: NaN where the constraints have no root.

    Raises `ValueError` unless u has one or two variables.
    """
    if len(nlp.u) not in _GRID_POINTS:
        msg = f"These figures draw one or two u, got {len(nlp.u)}"
        raise ValueError(msg)
    problem, scenarios = nlp._nlp(
        solution.method, solution.surrogate, solution.objective, True, solution.c
    )

    # With u fixed, SAA's constraints are a square system in the scenario states
    stored = np.zeros(0)
    root_problem = None
    if isinstance(solution.method, SAA):
        stored = np.concatenate([solution.x.ravel("F"), solution.y.ravel("F")])
        states = Symbolic.sym("states", len(stored))
        u_fixed = Symbolic.sym("u", len(nlp.u))
        assert problem.g is not None
        root_problem = RootProblem(
            cas.Function(
                "constraints",
                [states, u_fixed],
                [problem.g(cas.vertcat(u_fixed, states))],
            ),
            problem.lower[len(nlp.u) :],
            problem.upper[len(nlp.u) :],
        )

    # Outwards from the solution, each grid point from the nearest one solved
    lower, upper = start_bounds(nlp.u)
    lines = [
        np.linspace(lo, up, _GRID_POINTS[len(nlp.u)]) for lo, up in zip(lower, upper)
    ]
    u = np.stack([line.ravel() for line in np.meshgrid(*lines)], axis=1)
    scale = upper - lower
    n_scenarios, n_y = solution.y.shape
    objective = np.full(len(u), np.nan)
    y = np.full((len(u), n_scenarios, n_y), np.nan)
    f = np.full((len(u), n_scenarios), np.nan)
    solved_u, solved_states = [solution.u], [stored]
    for i in np.argsort(np.linalg.norm((u - solution.u) / scale, axis=1)):
        distance = np.linalg.norm((np.array(solved_u) - u[i]) / scale, axis=1)
        start = solved_states[np.argmin(distance)]
        try:
            root = (
                root_problem.solve(u[i], start[np.newaxis], _GRID_TOLERANCE)
                if root_problem
                else stored
            )
        except RootFindingError:
            continue
        w = np.concatenate([u[i], root])
        objective[i] = float(problem.f(w))
        _, y_i, f_i = scenarios(w)
        y[i], f[i] = y_i.full(), f_i.full().ravel()
        solved_u.append(u[i])
        solved_states.append(root)

    # Mean and standard deviation over the scenarios, by their weights
    w = solution.method.w if isinstance(solution.method, SAA) else np.ones(1)
    y_mean = np.einsum("isj,s->ij", y, w)
    y_std = np.sqrt(np.einsum("isj,s->ij", (y - y_mean[:, np.newaxis]) ** 2, w))
    f_mean = f @ w
    f_std = np.sqrt((f - f_mean[:, np.newaxis]) ** 2 @ w)
    return u, objective, y_mean, y_std, f_mean, f_std


def _draw_acquisition(
    nlp: HybridNLP,
    solution: Solution,
    ax: "Axes",
    u: np.ndarray,
    objective: np.ndarray,
    *,
    label: str,
) -> None:
    """Draws the acquisition in units of f from the `objective` on the grid u, with
    the solution marked as `label`."""
    # EI is -objective, SoftLogEI's is exp(-objective)
    if isinstance(solution.objective, EI):
        values, name = -objective, "EI"
    else:
        values, name = np.exp(-objective), "SoftEI"
    if len(nlp.u) == 2:
        _contour(nlp, solution, ax, u, values, name, label=label)
        return
    ax.plot(u[:, 0], values, color="C0")
    ax.set_ylabel(name)
    ax.set_xlabel(nlp.u[0].name)
    ax.axvline(solution.u[0], color="k", linestyle=":", label=label)
    ax.legend(fontsize="small")


def _contour(
    nlp: HybridNLP,
    solution: Solution,
    ax: "Axes",
    u: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    label: str,
) -> None:
    """Draws `values` on the square grid u of two u as filled contours, light to dark,
    with contour lines, a color bar, gray where the constraints have no root, and the
    solution marked as `label`."""
    shape = (_GRID_POINTS[2], _GRID_POINTS[2])
    grid = u[:, 0].reshape(shape), u[:, 1].reshape(shape), values.reshape(shape)
    filled = ax.contourf(*grid, levels=10, cmap="Blues")
    ax.contour(*grid, levels=filled.levels, colors="k", linewidths=0.5, alpha=0.3)
    ax.set_facecolor("0.8")
    figure = ax.get_figure()
    assert figure is not None
    figure.colorbar(filled, ax=ax)
    ax.scatter(
        *solution.u,
        color="C1",
        edgecolors="k",
        marker="*",
        s=200,
        zorder=3,
        label=label,
    )
    ax.set_title(title)
    ax.set_xlabel(nlp.u[0].name)
    ax.set_ylabel(nlp.u[1].name)
