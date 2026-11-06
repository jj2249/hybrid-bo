"""Static figures, each drawn on the axes you give it. Needs matplotlib."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from hybrid_bo.gp import FittedGP, predict, unscale

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
