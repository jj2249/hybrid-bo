"""Static figures, each drawn on the axes you give it. Needs matplotlib.

The library's look is the matplotlib style `default.mplstyle`: call
`plt.style.use("hybrid_bo.default")` before making your axes.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.errors import RootFindingError
from hybrid_bo.gp import FittedGP
from hybrid_bo.hybrid_model import DeterministicEquivalent
from hybrid_bo.problem import Symbolic
from hybrid_bo.solvers import NLP, RootProblem, Solution, find_root
from hybrid_bo.utils import unscale

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# `acquisition_on_grid` solves the equations at each grid point to this
_GRID_TOLERANCE = 1e-8


def convergence(curves: Mapping[str, Sequence[np.ndarray]], ax: "Axes") -> "Axes":
    """Draws each label's incumbent curves (`hb.incumbents`) over the evaluations: the
    median, with the interquartile range shaded. A shorter curve counts up to its end."""
    for label, group in curves.items():
        # Curves as rows, NaN past a curve's last evaluation
        rows = np.full((len(group), max(len(curve) for curve in group)), np.nan)
        for row, curve in zip(rows, group, strict=True):
            row[: len(curve)] = curve
        lower, median, upper = np.nanpercentile(rows, [25, 50, 75], axis=0)
        n = np.arange(1, rows.shape[1] + 1)
        (drawn,) = ax.plot(n, median, label=label)
        ax.fill_between(n, lower, upper, color=drawn.get_color(), alpha=0.2)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("incumbent")
    ax.legend()
    return ax


def regret(
    curves: Mapping[str, Sequence[np.ndarray]], optimum: float, ax: "Axes"
) -> "Axes":
    """Draws the regret from `optimum` of each label's incumbent curves, as
    `convergence` does, on a log y-axis."""
    regrets = {
        label: [curve - optimum for curve in group] for label, group in curves.items()
    }
    convergence(regrets, ax)
    ax.set_ylabel("regret")
    ax.set_yscale("log")
    return ax


def surrogate(fitted: FittedGP, lower: float, upper: float, ax: "Axes") -> "Axes":
    """Draws a fitted GP of one input between `lower` and `upper`: its mean, its band of
    two standard deviations and its training points, in measured units."""
    x = np.linspace(lower, upper, 201)
    mean, std = fitted.predict(x[:, np.newaxis])
    (drawn,) = ax.plot(x, mean, label="GP mean")
    ax.fill_between(
        x, mean - 2 * std, mean + 2 * std, color=drawn.get_color(), alpha=0.2
    )
    ax.scatter(
        unscale(fitted.x_scaler, fitted.X),
        unscale(fitted.y_scaler, fitted.Y[:, np.newaxis]),
        color="k",
        label="points",
    )
    ax.legend()
    return ax


def acquisition_on_grid(
    problem: NLP,
    solution: Solution,
    u: np.ndarray,
) -> tuple[np.ndarray, tuple[Dataset, ...]]:
    """The acquisition problem over a grid of points nobody measured, to draw what the
    solver minimized: the acquisition value at each row of u (n, N) and, for hybrid BO, one
    `Dataset` per scenario, of the states and f under that scenario at each row of u.

    A `Dataset` of the system can't give this: it holds only the points that were
    measured, and the real system's states, not the GPs' prediction of them.

    Standard BO's acquisition is analytic, so it has no scenarios. With u fixed, hybrid
    BO's constraints are a square system in the scenario states, which `find_root` solves point
    by point outwards from `solution`, each from the nearest point solved, to stay on the
    branch the optimizer chose. A point fails if the residual norm is above
    `_GRID_TOLERANCE` or a state is outside its bounds, and is NaN.
    """
    if not isinstance(problem, DeterministicEquivalent):
        return -problem.f.map(len(u))(u.T).full().ravel(), ()

    # The constraints as a root problem: the states are unknowns and u a parameter
    n_u = problem.n_u
    assert problem.g is not None
    assert solution.x is not None and solution.y is not None
    stored = problem.pack(solution)[n_u:]
    states, u_fixed = Symbolic.sym("v", len(stored)), Symbolic.sym("u", n_u)
    equations = cas.Function(
        "scenario_states", [states, u_fixed], [problem.g(cas.vertcat(u_fixed, states))]
    )
    root_problem = RootProblem(equations, problem.lower[n_u:], problem.upper[n_u:])

    # Outwards from the stored solution, each point from the nearest one solved
    solved_u, solved_states = [solution.u], [stored]
    values = np.full(len(u), np.nan)
    f = np.full((len(u), problem.n_scenarios), np.nan)
    x = np.full((len(u), *solution.x.shape), np.nan)
    y = np.full((len(u), *solution.y.shape), np.nan)
    for i in np.argsort(np.linalg.norm(u - solution.u, axis=1)):
        nearest = np.argmin(np.linalg.norm(np.array(solved_u) - u[i], axis=1))
        start = solved_states[nearest][np.newaxis]
        try:
            root = find_root(root_problem, u[i], start, _GRID_TOLERANCE)
        except RootFindingError:
            continue
        w = np.concatenate([u[i], root])
        values[i] = float(problem.f(w))
        f_i, y_i = problem.scenario_values(w)
        f[i], y[i], x[i] = f_i.full().ravel(), y_i.full(), problem.unpack(w)[1]
        solved_u.append(u[i])
        solved_states.append(root)

    scenarios = range(problem.n_scenarios)
    return problem.acquisition.from_objective(values), tuple(
        Dataset(u, x[:, s], y[:, s], f[:, s]) for s in scenarios
    )


def iteration(
    problem: NLP,
    solution: Solution,
    fitted: FittedGP,
    dataset: Dataset,
    axes: "Sequence[Axes]",
) -> None:
    """Draws one iteration of one decision variable on `axes`: the state before the trial
    point `solution`, over the start bounds of u, as `acquisition_on_grid` gives it, so the
    panels show the function the solver minimized. `dataset` holds the points before the
    trial point.

    Standard BO has two panels, its GP of f and the acquisition. Hybrid BO has three:
    `fitted`, the GP of an unknown state of u alone, then f under the scenarios, their
    weighted mean and std, and the acquisition. A point that wasn't solved is a gap.

    Raises `ValueError` if there isn't one decision variable, or one axes per panel.
    """
    # The checks
    panels = list(axes)
    is_hybrid = isinstance(problem, DeterministicEquivalent)
    if len(solution.u) != 1:
        msg = f"The panels draw one decision variable, got {len(solution.u)}"
        raise ValueError(msg)
    if len(panels) != (3 if is_hybrid else 2):
        msg = f"Standard BO draws 2 panels and hybrid BO 3, got {len(panels)} axes"
        raise ValueError(msg)

    # The GP, in its panel's color, and the acquisition over the start bounds of u
    assert problem.start_bounds is not None
    lower, upper = (float(bound[0]) for bound in problem.start_bounds)
    surrogate(fitted, lower, upper, panels[0])
    color = panels[0].get_lines()[0].get_color()
    u = np.linspace(lower, upper, 201)
    acquisition, scenarios = acquisition_on_grid(problem, solution, u[:, np.newaxis])

    # f under the scenarios, with the points so far
    if is_hybrid:
        f = np.column_stack([scenario.f for scenario in scenarios])
        mean = f @ problem.weights
        std = np.sqrt((f - mean[:, np.newaxis]) ** 2 @ problem.weights)
        panels[1].plot(u, mean, color=color, label="mean over scenarios")
        panels[1].fill_between(u, mean - std, mean + std, color=color, alpha=0.2)
        panels[1].scatter(dataset.u[:, 0], dataset.f, color="k", label="points")
        panels[1].set_ylabel("f")

    # The acquisition, and the trial point on every panel
    panels[-1].plot(u, acquisition, color=color, label="acquisition")
    panels[-1].set(xlabel="u", ylabel="acquisition")
    for ax in panels:
        ax.axvline(solution.u[0], color=color, linestyle=":", label="trial point")
        ax.legend(fontsize="small")
