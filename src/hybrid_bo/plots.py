"""Plots of finished runs: convergence across runs, and a run's iterations.

Plots are functions of a finished `Run` and the specs from the example, never
callbacks during the loop, so a saved run is plotted without running it again
(ADR 0003). matplotlib is imported on first use, so the library works without it.
Figures are drawn in the repo's style, `STYLE` (a matplotlib style file); use it for
other figures with `plt.style.use(hb.plots.STYLE)`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy

from hybrid_bo.acquisition import EI, LCB, Scenarios, standard_acquisition
from hybrid_bo.bo import HybridBOSettings, StandardBOSettings
from hybrid_bo.errors import RootFindingError
from hybrid_bo.evaluation import Experiment, _find_root, evaluate
from hybrid_bo.gp import default_gp, predict
from hybrid_bo.history import History, incumbents
from hybrid_bo.problem import Problem, Symbolic
from hybrid_bo.run import Run
from hybrid_bo.sampling import _start_bounds, lhs

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# The repo's matplotlib style: its rcParams for every figure
STYLE = Path(__file__).with_name("hybrid_bo.mplstyle")

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

    with plt.style.context(STYLE):
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
    ax.set_xlabel("evaluations")
    ax.set_ylabel("incumbent")
    return ax


def iterations(
    run: Run,
    problem: Problem,
    rng: np.random.Generator,
    settings: StandardBOSettings | HybridBOSettings | None = None,
    experiment: Experiment | None = None,
    i: int = -1,
) -> "Figure":
    """The figure of a run's iterations over a grid of the decision variable's start
    bounds, titled with the method, with the legend beside the panels and - and +
    buttons that step through the iterations (shown first: iteration `i`; negative
    counts from the end).

    Every iteration is computed when the figure is built, so the buttons only
    redraw. Iteration i is the state before the run's i-th trial point (design
    point, for the baselines): the first n_initial + i points, drawn black (initial)
    and red, their incumbent, and the GP fitted to them. The next point is a red
    ring. The panels, by method:

    - hybrid BO (`HybridBOSettings`): the GP of y with its confidence band, f under
      the scenarios (their mean and a band holding the central share of the
      scenario weight), and the acquisition function it minimized (-SAA-EI or
      SAA-LCB). Each scenario's states x_s are the first root of g that fsolve finds
      from the plot's starts; where g has several roots, this may not be the one
      the acquisition problem's solver chose. A scenario with no root is left out
      at that grid point, and the weights of the others are renormalized.
    - standard BO (`StandardBOSettings`): the GP of f with its confidence band and
      the acquisition function it minimized (-EI or LCB).
    - the baselines (no settings): the points only, since they fit no GP.

    With `experiment`, the true curves (the oracle values, else the reconstruction)
    are drawn dashed. The experiment is run at every grid point, so pass an oracle
    experiment, never a real system. `rng` draws the experiment's starts first,
    then the root-finding starts of each iteration in turn.

    Raises `ValueError` unless the problem has one decision variable (and, for hybrid
    BO, h takes only u), `TypeError` if the settings aren't the run's method's, and
    `IndexError` if the run has no iteration `i`.
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

    # The true curves, from the experiment at every grid point
    grid = np.linspace(*_start_bounds(system.u), _N_GRID)
    truth = None if experiment is None else evaluate(problem, experiment, grid, rng)
    y_true = None if truth is None else _true_values(truth.y_oracle, truth.y)[:, 0]
    f_true = None if truth is None else _true_values(truth.f_oracle, truth.f)

    # Every iteration's panels, computed once
    frames = [_frame(run, k, problem, settings, grid, rng) for k in range(n_iterations)]

    # The figure; matplotlib is imported only to draw
    import matplotlib.pyplot as plt
    from matplotlib.layout_engine import ConstrainedLayoutEngine
    from matplotlib.widgets import Button

    u, u_points = grid[:, 0], history.u[:, 0]
    label = "design points" if settings is None else "trial points"
    n_panels = {HybridBOSettings: 3, StandardBOSettings: 2}.get(type(settings), 1)
    y_name, u_name = system.y[0].name, system.u[0].name

    def draw(k: int) -> None:
        """Draws iteration k's panels on the figure's axes, and the figure's legend."""
        frame = frames[k]
        with plt.style.context(STYLE):
            # `clear` resets each axes to the current style
            for ax in axes:
                ax.clear()

            # Hybrid BO: the GP of y, f under the scenarios, the acquisition function
            if isinstance(settings, HybridBOSettings):
                assert run.scenarios is not None  # hybrid BO runs hold their scenarios
                _points(axes[0], u_points, history.y[:, 0], history, frame, label)
                _gp_panel(axes[0], u, frame, y_true)
                axes[0].set_ylabel(y_name)
                _points(axes[1], u_points, history.f, history, frame, label)
                _scenarios_panel(axes[1], u, frame, run.scenarios.w, f_true)
                _acquisition_panel(axes[2], u, frame, settings.acquisition)

            # Standard BO: the GP of f, the acquisition function
            elif isinstance(settings, StandardBOSettings):
                _points(axes[0], u_points, history.f, history, frame, label)
                _gp_panel(axes[0], u, frame, f_true)
                axes[0].set_ylabel("f")
                _acquisition_panel(axes[1], u, frame, settings.acquisition)

            # The baselines: the points and the true objective
            else:
                _points(axes[0], u_points, history.f, history, frame, label)
                if f_true is not None:
                    axes[0].plot(u, f_true, color="k", linestyle="--", label="true")
                axes[0].set_ylabel("f")

            axes[-1].set_xlabel(u_name)
            _figure_legend(figure, axes)
            counter.set_text(f"iteration {k} of {n_iterations - 1}")

    # The panels and the title, with a strip for the buttons below them (in inches,
    # so it is the same in every figure)
    height = 2 + 2 * n_panels
    strip, button_height = 0.5 / height, 0.3 / height
    with plt.style.context(STYLE):
        figure, panels = plt.subplots(
            n_panels,
            1,
            sharex=True,
            figsize=(8, height),
            squeeze=False,
            layout=ConstrainedLayoutEngine(rect=(0, strip, 1, 1 - strip)),
        )
        axes = list(panels[:, 0])
        figure.suptitle(method)
        counter = figure.text(0.5, strip / 2, "", ha="center", va="center")
    shown = i % n_iterations
    draw(shown)
    if n_iterations == 1:
        return figure

    # The buttons: - and + step to the previous and next iteration, within the run
    bottom = (strip - button_height) / 2
    minus = Button(figure.add_axes((0.32, bottom, 0.05, button_height)), "-")
    plus = Button(figure.add_axes((0.63, bottom, 0.05, button_height)), "+")

    def step(by: int) -> None:
        """Redraws the iteration `by` after the one shown, if the run has it."""
        nonlocal shown
        if 0 <= shown + by < n_iterations:
            shown += by
            draw(shown)
            figure.canvas.draw_idle()

    minus.on_clicked(lambda _: step(-1))
    plus.on_clicked(lambda _: step(1))
    # matplotlib keeps only weak references to widgets, so the figure holds them
    setattr(figure, "_hybrid_bo_buttons", (minus, plus))  # noqa: B010
    return figure


def _figure_legend(figure: "Figure", axes: Sequence["Axes"]) -> None:
    """Replaces the figure's legend with one entry per label across the axes, outside
    them on the right."""
    for legend in list(figure.legends):
        legend.remove()
    handles: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    figure.legend(list(handles.values()), list(handles), loc="outside right upper")


@dataclass(frozen=True)
class _Frame:
    """One iteration's precomputed panels: the number of points so far, and the
    GP's mean and std, f under the scenarios (n_grid, S) and the acquisition function
    over the grid, each None where the method has none."""

    n_so_far: int
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    f: np.ndarray | None = None
    acquisition: np.ndarray | None = None


def _frame(
    run: Run,
    k: int,
    problem: Problem,
    settings: StandardBOSettings | HybridBOSettings | None,
    grid: np.ndarray,
    rng: np.random.Generator,
) -> _Frame:
    """The panels of iteration k of the run, over the grid (n_grid, 1)."""
    # The points so far; the baselines stop there, since they fit no GP
    history = run.history
    n_so_far = history.n_initial + k
    if settings is None:
        return _Frame(n_so_far)
    incumbent = float(history.f[:n_so_far].min())

    # The GP behind the iteration's trial point
    system = problem.system
    gp = default_gp(system.n_u) if settings.gp is None else settings.gp
    fitted = run.surrogates[k]
    mean, std = predict(gp, fitted, grid)

    # Hybrid BO: f under the scenarios, then the SAA acquisition function
    if isinstance(settings, HybridBOSettings):
        assert run.scenarios is not None  # hybrid BO runs hold their scenarios
        f = _scenario_objectives(problem, grid, mean, std, run.scenarios, rng)
        acquisition = _saa_acquisition(
            f, run.scenarios.w, incumbent, settings.acquisition
        )
        return _Frame(n_so_far, mean, std, f, acquisition)

    # Standard BO: the objective of its acquisition problem
    nlp, _ = standard_acquisition(problem, gp, fitted, incumbent, settings.acquisition)
    acquisition = np.array([nlp.f(point).full().item() for point in grid])
    return _Frame(n_so_far, mean, std, acquisition=acquisition)


def _true_values(oracle: np.ndarray | None, measured: np.ndarray) -> np.ndarray:
    """The oracle values, or the measured and reconstructed ones without an oracle."""
    return measured if oracle is None else oracle


def _points(
    ax: "Axes",
    u: np.ndarray,
    values: np.ndarray,
    history: History,
    frame: _Frame,
    label: str,
) -> None:
    """The points so far, the initial ones in black and the rest in red, and the
    next point as a red ring."""
    n_initial, n = history.n_initial, frame.n_so_far
    ax.scatter(u[:n_initial], values[:n_initial], c="k", label="initial points")
    ax.scatter(u[n_initial:n], values[n_initial:n], c="r", label=label)
    ax.scatter(u[n], values[n], facecolors="none", edgecolors="r", label="next point")


def _gp_panel(
    ax: "Axes", u: np.ndarray, frame: _Frame, true: np.ndarray | None
) -> None:
    """A GP's mean and confidence band over u, and the true curve if known."""
    assert frame.mean is not None and frame.std is not None  # BO frames have a GP
    if true is not None:
        ax.plot(u, true, color="k", linestyle="--", label="true")
    ax.plot(u, frame.mean, color="b", label="GP mean")
    # The band holds _CONFIDENCE of the normal's probability
    half_width = scipy.stats.norm.ppf(0.5 + _CONFIDENCE / 2) * frame.std
    ax.fill_between(
        u,
        frame.mean - half_width,
        frame.mean + half_width,
        alpha=0.2,
        color="b",
        label=f"{_CONFIDENCE:.0%} band",
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
    frame: _Frame,
    w: np.ndarray,
    true: np.ndarray | None,
) -> None:
    """f under the scenarios over u: the weighted mean, the band holding the central
    `_CONFIDENCE` of the scenario weight, and the true objective if known. Scenarios
    without a root at a grid point are left out there (`_scenario_weights`)."""
    f = frame.f
    assert f is not None  # hybrid BO frames have the scenarios
    if true is not None:
        ax.plot(u, true, color="k", linestyle="--", label="true")
    weights = _scenario_weights(f, w)
    ax.plot(u, _weighted_mean(f, weights), color="b", label="scenario mean")

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
        u, band[:, 0], band[:, 1], alpha=0.2, color="b", label=f"{_CONFIDENCE:.0%} band"
    )
    ax.set_ylabel("f")


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
    ax: "Axes", u: np.ndarray, frame: _Frame, acquisition: EI | LCB
) -> None:
    """The acquisition function over u, as the acquisition problem minimizes it."""
    assert frame.acquisition is not None  # BO frames have an acquisition function
    name = "LCB" if isinstance(acquisition, LCB) else "-EI"
    ax.plot(u, frame.acquisition, color="k", label=name)
    ax.set_ylabel(name)
