"""Static figures, each drawn on the axes you give it. Needs matplotlib.

The library's look is the matplotlib style `default.mplstyle`: call
`plt.style.use("hybrid_bo.default")` before making your axes.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from hybrid_bo.acquisition import replay
from hybrid_bo.dataset import Dataset
from hybrid_bo.gp import FittedGP, predict
from hybrid_bo.hybrid_model import DeterministicEquivalent, Solution
from hybrid_bo.utils import unscale

if TYPE_CHECKING:
    from matplotlib.axes import Axes


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
    mean, std = predict(fitted, x[:, np.newaxis])
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


def iteration(
    problem: DeterministicEquivalent,
    solution: Solution,
    fitted: FittedGP,
    dataset: Dataset,
    axes: "Sequence[Axes]",
) -> None:
    """Draws one iteration of one decision variable on `axes`: the state before the trial
    point `solution`, over the start bounds of u, as `replay` gives it, so the panels show
    the function the solver minimized. `dataset` holds the points before the trial point.

    Standard BO has two panels, its GP of f and EI. Hybrid BO has three: `fitted`, the GP of an unknown state of u alone, then f under the scenarios,
    their weighted mean and std, and EI. A point the replay didn't solve is a gap.

    Raises `ValueError` if there isn't one decision variable, or one axes per panel.
    """
    # The checks
    panels = list(axes)
    is_hybrid = problem.n_scenarios > 0
    if problem.n_u != 1:
        msg = f"The panels draw one decision variable, got {problem.n_u}"
        raise ValueError(msg)
    if len(panels) != (3 if is_hybrid else 2):
        msg = f"Standard BO draws 2 panels and hybrid BO 3, got {len(panels)} axes"
        raise ValueError(msg)

    # The GP, in its panel's color, and the replay over the start bounds of u
    lower, upper = (float(bound[0]) for bound in problem.start_bounds)
    surrogate(fitted, lower, upper, panels[0])
    color = panels[0].get_lines()[0].get_color()
    u = np.linspace(lower, upper, 201)
    replayed = replay(problem, solution, u[:, np.newaxis])

    # f under the scenarios, with the points so far
    if is_hybrid:
        assert replayed.f is not None
        mean = replayed.f @ problem.weights
        std = np.sqrt((replayed.f - mean[:, np.newaxis]) ** 2 @ problem.weights)
        panels[1].plot(u, mean, color=color, label="mean over scenarios")
        panels[1].fill_between(u, mean - std, mean + std, color=color, alpha=0.2)
        panels[1].scatter(dataset.u[:, 0], dataset.f, color="k", label="points")
        panels[1].set_ylabel("f")

    # EI, and the trial point on every panel
    panels[-1].plot(u, replayed.ei, color=color, label="EI")
    panels[-1].set(xlabel="u", ylabel="EI")
    for ax in panels:
        ax.axvline(solution.u[0], color=color, linestyle=":", label="trial point")
        ax.legend(fontsize="small")
