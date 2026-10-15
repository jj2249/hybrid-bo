"""Plots of finished runs: convergence across runs, and the figure of one iteration.

Plots are functions of a finished `Run` and the specs from the example, never
callbacks during the loop, so a saved run is plotted without running it again
(ADR 0003). matplotlib is imported on first use, so the library works without it.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy

from hybrid_bo.acquisition import EI, LCB, Scenarios, standard_acquisition
from hybrid_bo.bo import HybridBOSettings, StandardBOSettings
from hybrid_bo.errors import RootFindingError
from hybrid_bo.evaluation import Experiment, _find_root, evaluate
from hybrid_bo.gp import default_gp, fit_gp, predict
from hybrid_bo.history import incumbents
from hybrid_bo.problem import Problem, Symbolic
from hybrid_bo.run import Run
from hybrid_bo.sampling import _start_bounds, lhs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_N_GRID = 101  # grid points over u
_CONFIDENCE = 0.9  # probability inside the confidence bands
_N_ROOT_STARTS = 10  # fsolve starts per scenario state
_ROOT_TOL = 1e-10

# The settings each BO method's runs are plotted with, by the method in the record
_SETTINGS = {"hybrid_bo": HybridBOSettings, "standard_bo": StandardBOSettings}


def _axes(ax: "Axes | None") -> "Axes":
    """`ax`, or the axes of a new figure."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt  # only to draw, so the library works without it

    _, ax = plt.subplots()
    return ax


def convergence(
    runs: Sequence[Run], ax: "Axes | None" = None, label: str | None = None
) -> "Axes":
    """The incumbents of each run over its evaluations, as thin lines, and their median
    as a thick line labeled `label`, with a dotted line after the initial points.

    A shorter run (one that failed) counts in the median up to its last evaluation.
    Call it once per method on the same axes to compare methods. Raises `ValueError`
    without runs.
    """
    if not runs:
        msg = "convergence needs at least one run"
        raise ValueError(msg)
    ax = _axes(ax)

    # Each run, in one color
    curves = [incumbents(run.history) for run in runs]
    color = None
    for curve in curves:
        evaluations = np.arange(1, len(curve) + 1)
        (drawn,) = ax.plot(evaluations, curve, color=color, alpha=0.3, linewidth=1)
        color = drawn.get_color()

    # The median over the runs that reach each evaluation
    n = max(len(curve) for curve in curves)
    padded = np.full((len(curves), n), np.nan)
    for row, curve in zip(padded, curves, strict=True):
        row[: len(curve)] = curve
    median = np.nanmedian(padded, axis=0)
    ax.plot(np.arange(1, n + 1), median, color=color, linewidth=2, label=label)

    ax.axvline(runs[0].history.n_initial + 0.5, color="grey", linestyle=":")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Incumbent")
    return ax


def iteration(
    run: Run,
    i: int,
    problem: Problem,
    rng: np.random.Generator,
    settings: StandardBOSettings | HybridBOSettings | None = None,
    experiment: Experiment | None = None,
) -> "Figure":
    """The figure of iteration i of a run, over a grid of the decision variable's
    start bounds.

    Iteration i (negative counts from the end) is the state before the run's i-th
    trial point (design point, for the baselines): the first n_initial + i points,
    drawn black (initial) and red, their incumbent, and the GP fitted to them. The
    next point is drawn as a red ring. The panels, by method:

    - hybrid BO (`HybridBOSettings`): the GP of y with its confidence band, f under
      the scenarios (their mean and a band holding the central share of the
      scenario weight), and the acquisition function it minimized (-SAA-EI or
      SAA-LCB). Each scenario's states x_s are the first root of g that fsolve finds
      from the plot's starts; where g has several roots, this may not be the one
      the acquisition problem's solver chose. A scenario with no root is left out
      at that grid point, and the weights of the others are renormalized.
    - standard BO (`StandardBOSettings`): the GP of f with its confidence band and
      the acquisition function it minimized (-EI or LCB).
    - the baselines (no settings): the GP of f, the default GP fitted here.

    With `experiment`, the true curves (the oracle values, else the reconstruction)
    are drawn dashed. The experiment is run at every grid point, so pass an oracle
    experiment, never a real system. `rng` draws the baselines' GP fit first, then the
    experiment and root-finding starts.

    Raises `ValueError` unless the problem has one decision variable (and, for hybrid
    BO, h takes only u), `TypeError` if the settings aren't the run's method's, and
    `IndexError` if the run has no such iteration.
    """
    # The checks: one decision variable, the method's settings, the iteration
    system = problem.system
    if system.n_u != 1:
        msg = f"The plots need one decision variable, got {system.n_u}"
        raise ValueError(msg)
    method = run.record.get("method", "")
    expected = _SETTINGS.get(method, type(None))
    if not isinstance(settings, expected):
        msg = (
            f"A {method} run is plotted with {expected.__name__} settings, got "
            f"{type(settings).__name__}"
        )
        raise TypeError(msg)
    is_hybrid = isinstance(settings, HybridBOSettings)
    if is_hybrid and list(system.h_inputs) != list(system.u):
        msg = "The GP of y is drawn over u, so h_inputs must be the decision variable"
        raise ValueError(msg)
    history = run.history
    n_iterations = len(history.u) - history.n_initial
    if not -n_iterations <= i < n_iterations:
        msg = f"The run has {n_iterations} iterations, so no iteration {i}"
        raise IndexError(msg)

    # The points so far, the next one, and the grid
    n_so_far = history.n_initial + i % n_iterations
    so_far = slice(0, n_so_far)
    incumbent = float(history.f[so_far].min())
    grid = np.linspace(*_start_bounds(system.u), _N_GRID)

    # The surrogate: the run's, or for a baseline the default GP fitted here
    if settings is None:
        gp = default_gp(system.n_u)
        fitted = fit_gp(gp, history.u[so_far], history.f[so_far], rng)
    else:
        gp = default_gp(system.n_u) if settings.gp is None else settings.gp
        fitted = run.surrogates[i]
    mean, std = predict(gp, fitted, grid)

    # The true curves, from the experiment at every grid point
    truth = None if experiment is None else evaluate(problem, experiment, grid, rng)
    y_true = None if truth is None else _true_values(truth.y_oracle, truth.y)[:, 0]
    f_true = None if truth is None else _true_values(truth.f_oracle, truth.f)

    # The figure; matplotlib is imported only to draw
    import matplotlib.pyplot as plt

    u, u_points = grid[:, 0], history.u[:, 0]
    label = "Design points" if settings is None else "Points found with BO"

    # Hybrid BO: the GP of y, f under the scenarios, the acquisition function
    if isinstance(settings, HybridBOSettings):
        assert run.scenarios is not None  # hybrid BO runs hold their scenarios
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
        _points(axes[0], u_points, history.y[:, 0], history.n_initial, n_so_far, label)
        _gp_panel(axes[0], u, mean, std, y_true)
        axes[0].set_ylabel("Variable $y$")
        f = _scenario_objectives(problem, grid, mean, std, run.scenarios, rng)
        _points(axes[1], u_points, history.f, history.n_initial, n_so_far, label)
        _scenarios_panel(axes[1], u, f, run.scenarios.w, f_true)
        values = _saa_acquisition(f, run.scenarios.w, incumbent, settings.acquisition)
        _acquisition_panel(axes[2], u, values, settings.acquisition)

    # Standard BO: the GP of f, the acquisition function
    elif isinstance(settings, StandardBOSettings):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
        _points(axes[0], u_points, history.f, history.n_initial, n_so_far, label)
        _gp_panel(axes[0], u, mean, std, f_true)
        axes[0].set_ylabel("Objective")
        nlp, _ = standard_acquisition(
            problem, gp, fitted, incumbent, settings.acquisition
        )
        values = np.array([nlp.f(point).full().item() for point in grid])
        _acquisition_panel(axes[1], u, values, settings.acquisition)

    # The baselines: the GP of f
    else:
        figure, ax = plt.subplots()
        axes = [ax]
        _points(ax, u_points, history.f, history.n_initial, n_so_far, label)
        _gp_panel(ax, u, mean, std, f_true)
        ax.set_ylabel("Objective")

    axes[0].legend(
        ncols=2,
        loc="lower right",
        mode="expand",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        borderaxespad=0,
    )
    axes[-1].set_xlabel(system.u[0].name)
    figure.tight_layout()
    return figure


def _true_values(oracle: np.ndarray | None, measured: np.ndarray) -> np.ndarray:
    """The oracle values, or the measured and reconstructed ones without an oracle."""
    return measured if oracle is None else oracle


def _points(
    ax: "Axes",
    u: np.ndarray,
    values: np.ndarray,
    n_initial: int,
    n: int,
    label: str,
) -> None:
    """The first n points, the initial ones in black and the rest in red, and point n,
    the next one, as a red ring."""
    ax.scatter(u[:n_initial], values[:n_initial], c="k", label="Initial points")
    ax.scatter(u[n_initial:n], values[n_initial:n], c="r", label=label)
    ax.scatter(u[n], values[n], facecolors="none", edgecolors="r", label="Next point")


def _gp_panel(
    ax: "Axes",
    u: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    true: np.ndarray | None,
) -> None:
    """A GP's mean and confidence band over u, and the true curve if known."""
    if true is not None:
        ax.plot(u, true, color="k", linestyle="--", label="True")
    ax.plot(u, mean, color="b", label="Predicted mean")
    # The band holds _CONFIDENCE of the normal's probability
    half_width = scipy.stats.norm.ppf(0.5 + _CONFIDENCE / 2) * std
    ax.fill_between(
        u,
        mean - half_width,
        mean + half_width,
        alpha=0.2,
        color="b",
        label="Confidence interval",
    )


def _scenario_objectives(
    problem: Problem,
    grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    scenarios: Scenarios,
    rng: np.random.Generator,
) -> np.ndarray:
    """f under each scenario at each grid point, (n_grid, S).

    Scenario s has y_s = mean + std * z_s, and its mechanistic states x_s solve
    g(x_s, y_s, u) = 0 from Latin hypercube starts over their start bounds, as in the
    acquisition problem. Where no start gives a root within the bounds, f is NaN.
    Where g has several roots, the first one found may not be the one the
    acquisition problem's solver chose.
    """
    system = problem.system
    x = Symbolic.sym("x", system.n_x)
    f = np.full((len(grid), len(scenarios.w)), np.nan)
    for i, u in enumerate(grid):
        for s, z in enumerate(scenarios.z):
            y = mean[i] + std[i] * z  # one unknown state (P = 1)
            starts = lhs(*_start_bounds(system.x), _N_ROOT_STARTS, rng)
            try:
                x_s = _find_root(x, system.g(x, y, u), system.x, starts, _ROOT_TOL)
            except RootFindingError:
                continue
            f[i, s] = problem.f(x_s, y, u).full().item()
    return f


def _scenario_weights(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weight of each scenario at each grid point, (n_grid, S): the scenario
    weights w of the scenarios with a root there (f not NaN), renormalized to sum to
    1, and 0 for the others. A grid point where no scenario has a root gets NaN."""
    weights = np.where(np.isnan(f), 0.0, w)
    total = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return weights / total


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted mean of each row, skipping the entries with weight 0 (which may
    be NaN), shape (n_grid,)."""
    return np.sum(np.where(weights > 0, values, 0.0) * weights, axis=1)


def _scenarios_panel(
    ax: "Axes",
    u: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    true: np.ndarray | None,
) -> None:
    """f under the scenarios over u: the weighted mean, the band holding the central
    `_CONFIDENCE` of the scenario weight, and the true objective if known. Scenarios
    without a root at a grid point are left out there (`_scenario_weights`)."""
    if true is not None:
        ax.plot(u, true, color="k", linestyle="--", label="True")
    weights = _scenario_weights(f, w)
    ax.plot(u, _weighted_mean(f, weights), color="b", label="Scenario mean")

    # The band, per grid point over the scenarios with a root there
    tail = (1 - _CONFIDENCE) / 2
    band = np.full((len(u), 2), np.nan)
    for row, (f_row, w_row) in enumerate(zip(f, weights, strict=True)):
        has_root = w_row > 0
        if has_root.any():
            band[row] = np.quantile(
                f_row[has_root],
                [tail, 1 - tail],
                weights=w_row[has_root],
                method="inverted_cdf",
            )
    ax.fill_between(
        u, band[:, 0], band[:, 1], alpha=0.2, color="b", label="Confidence interval"
    )
    ax.set_ylabel("Objective")


def _saa_acquisition(
    f: np.ndarray, w: np.ndarray, incumbent: float, acquisition: EI | LCB
) -> np.ndarray:
    """Hybrid BO's acquisition function at each grid point from f under the scenarios
    (n_grid, S), as its acquisition problem minimizes it: -SAA-EI, the weighted
    average of min(f_s - incumbent, 0), or SAA-LCB. Scenarios without a root at a
    grid point are left out there (`_scenario_weights`)."""
    weights = _scenario_weights(f, w)
    if isinstance(acquisition, LCB):
        f_mean = _weighted_mean(f, weights)
        f_variance = _weighted_mean((f - f_mean[:, np.newaxis]) ** 2, weights)
        return f_mean - acquisition.std_factor * np.sqrt(f_variance)
    return _weighted_mean(np.fmin(f - incumbent, 0), weights)


def _acquisition_panel(
    ax: "Axes", u: np.ndarray, values: np.ndarray, acquisition: EI | LCB
) -> None:
    """The acquisition function over u, as the acquisition problem minimizes it."""
    label = (
        "Lower confidence bound"
        if isinstance(acquisition, LCB)
        else "Expected improvement"
    )
    ax.plot(u, values, color="k", label=label)
    ax.legend(loc="best")
    ax.set_ylabel("Acq.-Fun.")
