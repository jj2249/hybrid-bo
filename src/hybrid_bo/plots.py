"""Static plots of a study: convergence and regret across runs, and one iteration of a
run.

Plots read finished runs, so a saved study is plotted without running it again (ADR
0003); an iteration's acquisition panels replay the method's acquisition problem (ADR
0005). matplotlib and Polars are imported on first use. Figures are drawn in the repo's
style, `STYLE`; use it elsewhere with `plt.style.use(hb.plots.STYLE)`.
"""

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.acquisition import (
    LCB,
    AcquisitionProblem,
    AcquisitionSolution,
    Linearization,
    Scenarios,
    SoftSAA,
    _pack,
    _scenario_unknown_states,
    _sensitivity,
    _unpack,
    hybrid_acquisition,
    standard_acquisition,
)
from hybrid_bo.errors import ReplayWarning
from hybrid_bo.evaluation import Measure, evaluate
from hybrid_bo.gp import FittedGP, predict
from hybrid_bo.history import n_iterations
from hybrid_bo.problem import Problem, Symbolic, _h_inputs
from hybrid_bo.run import Run
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.settings import HybridBOSettings, MethodSettings, StandardBOSettings
from hybrid_bo.studies import Study
from hybrid_bo.views import evaluations

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# The repo's matplotlib style: its rcParams for every figure
STYLE = Path(__file__).with_name("hybrid_bo.mplstyle")

_N_GRID = 101  # grid points over u
_N_GRID_2D = 30  # grid points over each of two decision variables
_N_LEVELS = 21  # levels of a filled contour
_DECADES = 2  # a panel of f spanning more decades than this has a log color scale
_ZERO_EI_COLOR = "lightgray"
_EI_DECADES = 10  # the EI color scale stops this many decades below the largest EI
_CONFIDENCE = 0.9  # probability inside the confidence bands
_RESIDUAL_TOLERANCE = 1e-8  # a replay's root solves each equation to this
# The replay at the trial point is the stored solution to these, loose enough for
# IPOPT's and Newton's tolerances, tight enough to tell another root
_REPLAY_RTOL = 1e-4
_REPLAY_ATOL = 1e-6


def convergence(study: Study, ax: "Axes | None" = None) -> "Axes":
    """Per configuration, the median over its runs of the oracle incumbent (or, in a
    study without an oracle, the incumbent) at each evaluation, with the
    interquartile range shaded, a dotted line after the initial points (one for each
    number of initial points the runs have), and a legend.

    Raises `ValueError` if the study has no evaluations, and `ImportError` without
    Polars.
    """
    rows = evaluations(study)
    has_oracle = not rows.is_empty() and rows["oracle_incumbent"].null_count() == 0
    column = "oracle_incumbent" if has_oracle else "incumbent"
    return _over_evaluations(rows, column, _n_initials(study), ax)


def regret(
    study: Study, optimum: float | None = None, ax: "Axes | None" = None
) -> "Axes":
    """The median regret, drawn as in `convergence` on a log y-axis (the paper's
    figure). The oracle optimum is `optimum` if given, else the study's recorded one.

    Raises `ValueError` if the study has no regret (no oracle, or no optimum), and
    `ImportError` without Polars.
    """
    rows = evaluations(study, optimum)
    ax = _over_evaluations(rows, "regret", _n_initials(study), ax)
    ax.set_yscale("log")
    return ax


def _n_initials(study: Study) -> list[int]:
    """The numbers of initial points of the study's runs, each once, in increasing
    order."""
    return sorted({run.history.n_initial for run in study.runs})


def _over_evaluations(
    rows: "pl.DataFrame", column: str, n_initials: list[int], ax: "Axes | None"
) -> "Axes":
    """`convergence`'s figure of `column`, on `ax` or a new figure's axes, with a
    dotted line after each of `n_initials` initial points. A shorter run counts up to
    its last evaluation.

    Raises `ValueError` if the column has no values.
    """
    import matplotlib.pyplot as plt
    import polars as pl  # `evaluations` has checked it is installed

    rows = rows.drop_nulls(column) if column in rows.columns else rows.clear()
    if rows.is_empty():
        msg = f"The study has no {column.replace('_', ' ')} to draw"
        raise ValueError(msg)
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots()

    # Per method: the median and the quartiles at each evaluation
    values = pl.col(column)
    for method in rows["method"].unique(maintain_order=True):
        curve = (
            rows.filter(pl.col("method") == method)
            .group_by("evaluation")
            .agg(
                median=values.median(),
                lower=values.quantile(0.25, "linear"),
                upper=values.quantile(0.75, "linear"),
            )
            .sort("evaluation")
        )
        n = curve["evaluation"].to_numpy()
        (drawn,) = ax.plot(n, curve["median"].to_numpy(), label=str(method))
        lower, upper = curve["lower"].to_numpy(), curve["upper"].to_numpy()
        ax.fill_between(n, lower, upper, color=drawn.get_color(), alpha=0.2)

    for n_initial in n_initials:
        ax.axvline(n_initial + 0.5, color="grey", linestyle=":")
    ax.set_xlabel("evaluations")
    ax.set_ylabel(column.replace("_", " "))
    ax.legend()
    return ax


def budget_heatmap(
    study: Study,
    optimum: float | None = None,
    ax: "Axes | None" = None,
    tolerance: float | None = None,
) -> "Axes":
    """A sweep over initial fractions of one method by initial fraction ε =
    n_initial / B (x) and budget B (y, evaluations so far), with a colorbar, so a
    row is the choice of n_initial at one budget. A cell is the median log10 regret
    over the runs or, with `tolerance`, the fraction of runs whose regret is at
    most `tolerance` (a median can't tell runs at the optimum from runs stuck in a
    local minimum just above it). There is a cell wherever n_initial < B, read from
    the runs with that n_initial after B evaluations (the prefix property), so one
    run per n_initial gives every budget it reaches. A cell spans ε halfway to the
    neighboring numbers of initial points, so the cells of a row tile it.

    The oracle optimum is `optimum` if given, else the study's recorded one. Raises
    `ValueError` if the runs are of more than one method or the study has no
    regret, and `ImportError` without Polars.
    """
    import matplotlib.pyplot as plt
    import polars as pl  # `_sweep_regret` checks it is installed
    from matplotlib.collections import PolyCollection

    # The cells: one per n_initial and budget with runs
    regret = pl.col("regret")
    value = (
        regret.log10().median() if tolerance is None else (regret <= tolerance).mean()
    )
    rows = _sweep_regret(study, optimum).filter(
        pl.col("n_initial") < pl.col("evaluation")
    )
    cells = rows.group_by("n_initial", "evaluation").agg(value=value)

    # Each n_initial's span in initial points: halfway to its neighbors, and as far
    # on the outside (half a point either side if it is the only one)
    n_initials = np.sort(cells["n_initial"].unique().to_numpy())
    middles = (n_initials[:-1] + n_initials[1:]) / 2
    if len(n_initials) > 1:
        edges = np.r_[
            2 * n_initials[0] - middles[0], middles, 2 * n_initials[-1] - middles[-1]
        ]
    else:
        edges = n_initials[0] + np.array([-0.5, 0.5])
    spans = {n: (low, high) for n, low, high in zip(n_initials, edges[:-1], edges[1:])}

    # The heatmap: one rectangle per cell, its span over its budget in ε (within
    # [0, 1]), one unit of budget high
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots()
    rectangles, values = [], []
    for n_initial, budget, cell_value in cells.iter_rows():
        low, high = np.clip(np.array(spans[n_initial]) / budget, 0.0, 1.0)
        bottom, top = budget - 0.5, budget + 0.5
        rectangles.append([(low, bottom), (high, bottom), (high, top), (low, top)])
        values.append(cell_value)
    collection = PolyCollection(rectangles, array=np.array(values), edgecolors="face")
    if tolerance is not None:
        collection.set_clim(0.0, 1.0)
    ax.add_collection(collection)
    label = (
        "median log$_{10}$ regret"
        if tolerance is None
        else f"fraction of runs with regret ≤ {tolerance:g}"
    )
    ax.figure.colorbar(collection, ax=ax, label=label)
    budgets = cells["evaluation"].to_numpy()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(budgets.min() - 0.5, budgets.max() + 0.5)
    ax.set_xlabel("initial fraction ε")
    ax.set_ylabel("evaluations (budget)")
    ax.grid(visible=False)
    return ax


def regret_at_budget(
    study: Study,
    budget: int,
    optimum: float | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """The median regret after `budget` evaluations of a sweep over initial
    fractions of one method, against the initial fraction ε = n_initial / budget,
    for every n_initial < budget, with the interquartile range shaded, on a log
    y-axis.

    The oracle optimum is `optimum` if given, else the study's recorded one. Raises
    `ValueError` if the runs are of more than one method or have no regret after
    `budget` evaluations with an iteration among them, and `ImportError` without
    Polars.
    """
    import matplotlib.pyplot as plt
    import polars as pl  # `_sweep_regret` checks it is installed

    # Per n_initial: the median and the quartiles after `budget` evaluations
    regret = pl.col("regret")
    rows = _sweep_regret(study, optimum).filter(
        (pl.col("evaluation") == budget) & (pl.col("n_initial") < budget)
    )
    if rows.is_empty():
        msg = f"The study has no regret after {budget} evaluations with iterations"
        raise ValueError(msg)
    points = (
        rows.group_by("n_initial")
        .agg(
            median=regret.median(),
            lower=regret.quantile(0.25, "linear"),
            upper=regret.quantile(0.75, "linear"),
        )
        .sort("n_initial")
    )

    # The median against ε, with the quartiles shaded
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots()
    epsilon = points["n_initial"].to_numpy() / budget
    (drawn,) = ax.plot(epsilon, points["median"].to_numpy(), marker="o", label="median")
    lower, upper = points["lower"].to_numpy(), points["upper"].to_numpy()
    ax.fill_between(epsilon, lower, upper, color=drawn.get_color(), alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("initial fraction ε")
    ax.set_ylabel("regret")
    ax.set_title(f"after {budget} evaluations")
    return ax


def regret_by_initial_fraction(
    study: Study,
    optimum: float | None = None,
    ax: "Axes | None" = None,
    max_lines: int = 8,
) -> "Axes":
    """The median regret over evaluations of a sweep over initial fractions of one
    method, one line per number of initial points, on a log y-axis. With more than
    `max_lines` of them, an evenly spread subset is drawn, the smallest and the
    largest included.

    The oracle optimum is `optimum` if given, else the study's recorded one. Raises
    `ValueError` if the runs are of more than one method or the study has no
    regret, and `ImportError` without Polars.
    """
    import matplotlib.pyplot as plt
    import polars as pl  # `_sweep_regret` checks it is installed

    # The numbers of initial points to draw, evenly spread
    rows = _sweep_regret(study, optimum)
    n_initials = sorted(rows["n_initial"].unique().to_list())
    n_lines = min(max_lines, len(n_initials))
    chosen = np.unique(np.linspace(0, len(n_initials) - 1, n_lines).round())
    n_initials = [n_initials[int(i)] for i in chosen]

    # Per n_initial: the median at each evaluation, colored in order
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots()
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(n_initials)))
    for n_initial, color in zip(n_initials, colors):
        curve = (
            rows.filter(pl.col("n_initial") == n_initial)
            .group_by("evaluation")
            .agg(median=pl.col("regret").median())
            .sort("evaluation")
        )
        ax.plot(
            curve["evaluation"].to_numpy(),
            curve["median"].to_numpy(),
            color=color,
            label=f"n_initial = {n_initial}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("regret")
    ax.legend()
    return ax


def _sweep_regret(study: Study, optimum: float | None) -> "pl.DataFrame":
    """The regret of a sweep over initial fractions of one method, one row per
    evaluation of each run, with columns `n_initial` (the run's), `run` (its index),
    `evaluation` and `regret` (as in `evaluations`). Runs with the same n_initial and
    index pass through the same first evaluations (the prefix property), so only the
    longest of them is kept.

    Raises `ValueError` if the runs are of more than one method (another method, or
    the same with other settings) or the study has no regret, and `ImportError`
    without Polars.
    """
    # One method: its name and settings are the same in every run
    methods = {(run.artifact.get("method"), repr(run.settings)) for run in study.runs}
    if len(methods) > 1:
        msg = (
            "A sweep's figures need a study of one method, not "
            f"{len(methods)} (different methods or settings)"
        )
        raise ValueError(msg)

    # The longest run of each n_initial and index
    longest: dict[tuple[int, int], Run] = {}
    for run in study.runs:
        key = (run.history.n_initial, run.index)
        if key not in longest or len(run.history.f) > len(longest[key].history.f):
            longest[key] = run

    # Their regret, each run labeled with its n_initial, which with its index names
    # it uniquely
    import polars as pl  # `evaluations` checks it is installed

    kept = [
        replace(run, method=str(n_initial)) for (n_initial, _), run in longest.items()
    ]
    rows = evaluations(replace(study, runs=tuple(kept)), optimum)
    rows = rows.drop_nulls("regret") if "regret" in rows.columns else rows.clear()
    if rows.is_empty():
        msg = "The study has no regret to draw"
        raise ValueError(msg)
    n_initial = pl.col("method").cast(pl.Int64).alias("n_initial")
    return rows.select(n_initial, "run", "evaluation", "regret")


def iteration(
    run: Run,
    problem: Problem,
    i: int,
    *,
    measure: Measure | None = None,
    gp: bool = True,
    scenarios: bool = True,
    acquisition: bool = True,
) -> "Figure":
    """The figure of iteration i of a run (from 0; negative counts from the end): the
    state before its i-th trial point (design point, for the baselines), over a grid
    of the start bounds of one or two decision variables.

    The panels, each turned off by its flag:

    - `gp`: the stored surrogate's mean and band; of y for hybrid BO (one unknown
      state only), of f for standard BO.
    - `scenarios` (hybrid BO): f under the scenarios, their weighted mean and band;
      with linearization, f at the nominal states and the band f +- s, with s
      recomputed from the stored surrogate.
    - `acquisition`: the minimized acquisition function (-EI or LCB) and the stored
      value at the trial point. Standard BO's is analytic. Hybrid BO's replays the
      acquisition problem (ADR 0005) from the stored surrogate, scenarios and
      incumbent: with u fixed at each grid point, each scenario's states solve its
      equations by Newton's method, outwards from the stored solution, each grid
      point from the nearest one solved, and the problem's objective is evaluated
      there. A grid point where a scenario has no root within the bounds is a gap.
      If the replay at the trial point isn't the stored solution, it warns with
      `ReplayWarning`. SoftSAA's -log EI is drawn as -EI.

    With one decision variable the panels are stacked curves over a grid of 101
    points, each with the points so far, the incumbent and the trial point; the
    baselines (or every panel off) show the points with f. With two, they are filled
    contours over a 30 x 30 grid in rows, of y (oracle, GP mean, GP std) and of f
    (oracle, the scenarios' weighted mean or the nominal f, their weighted std or s,
    and EI), or for standard BO one row of f with its GP; each shows the points so
    far, the incumbent and the trial point in the plane of u. An oracle panel shares
    the color scale of the mean beside it. Panels of f are on a log color scale when
    their values are positive and span more than two decades. EI is drawn as log10
    EI, with the grid points where it is 0 (SAA-EI's flat regions) in gray, over at
    most 10 decades below the largest EI (smaller EI in the lowest color; SoftSAA's
    is computed from its -log EI, so it never underflows to 0); LCB is drawn as it
    is. The acquisition panel notes the stored and the replayed
    value at the trial point, which check the replay.

    `measure` adds the oracle values (curves, or the oracle panels in 2D); it runs at
    every grid point (generator seeded with 0), so pass an oracle measurement, never
    a real system.

    The GPs are the run's stored fitted GPs; the rest of the replay (the acquisition
    function, the approximation, the solver) comes from the run's stored settings.

    Raises `ValueError` unless the problem has one or two decision variables (and,
    for hybrid BO's GP panels, one unknown state and h of u only), or if `measure`
    has no oracle; and `IndexError` if the run has no iteration `i`.
    """
    # The checks: one or two decision variables, one GP of y over u, the iteration
    system, history, settings = problem.system, run.history, run.settings
    if system.n_u not in (1, 2):
        msg = f"The plots need one or two decision variables, got {system.n_u}"
        raise ValueError(msg)
    is_hybrid = isinstance(settings, HybridBOSettings)
    if is_hybrid and gp and system.n_y > 1:
        msg = (
            f"The GP panels draw one unknown state, got {system.n_y}: pass gp=False "
            "to draw the other panels"
        )
        raise ValueError(msg)
    if is_hybrid and gp and list(system.h_inputs) != list(system.u):
        msg = "The GP of y is drawn over u, so h_inputs must be the decision variables"
        raise ValueError(msg)
    n_later = n_iterations(history)
    if not -n_later <= i < n_later:
        msg = f"The run has {n_later} iterations, so no iteration {i}"
        raise IndexError(msg)
    i %= n_later

    # The panels this method has
    shown = {
        "gp": gp and settings is not None,
        "scenarios": scenarios and is_hybrid,
        "acquisition": acquisition and settings is not None,
    }
    if system.n_u == 1:
        return _iteration_1d(run, problem, i, measure, shown)
    return _iteration_2d(run, problem, i, measure, shown)


def _truth(
    problem: Problem, measure: Measure | None, u: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The oracle first unknown state and f at the rows of u, or None without a
    measurement.

    Raises `ValueError` if the measurement has no oracle.
    """
    if measure is None:
        return None, None
    truth = evaluate(problem, measure, u, np.random.default_rng(0))
    if truth.f_oracle is None or truth.y_oracle is None:
        msg = "The true curves need an oracle measurement"
        raise ValueError(msg)
    return truth.y_oracle[:, 0], truth.f_oracle


def _iteration_1d(
    run: Run,
    problem: Problem,
    i: int,
    measure: Measure | None,
    shown: dict[str, bool],
) -> "Figure":
    """`iteration`'s figure for one decision variable: a panel per entry of `shown`
    that is on, stacked."""
    system, history, settings = problem.system, run.history, run.settings
    is_hybrid = isinstance(settings, HybridBOSettings)

    # The grid, with the trial point in it so the replay starts there
    n = history.n_initial + i
    trial = history.u[n, 0]
    u = np.sort(np.append(np.linspace(*_start_bounds(system.u), _N_GRID), trial))
    y_true, f_true = _truth(problem, measure, u[:, None])

    # The replay the panels need
    panels = [panel for panel, on in shown.items() if on] or ["points"]
    curve = f_scenarios = f_std = None
    if settings is not None and (shown["scenarios"] or shown["acquisition"]):
        curve, f_scenarios, f_std = _replay(run, problem, i, settings, u[:, np.newaxis])

    import matplotlib.pyplot as plt

    with plt.style.context(STYLE):
        figure, axes = plt.subplots(
            len(panels),
            squeeze=False,
            sharex=True,
            figsize=(8, 2 + 2 * len(panels)),
            layout="constrained",
        )
        figure.suptitle(f"{run.method}, iteration {i}")
        for panel, ax in zip(panels, axes[:, 0], strict=True):
            # The acquisition function, and the stored value at the trial point.
            # SoftSAA minimizes -log EI, which is steep far from the incumbent and
            # squashes the rest, so it is drawn as -EI like the other EI approximations
            if panel == "acquisition":
                assert settings is not None and curve is not None
                value = run.acquisitions[i].value
                if _is_soft(settings):
                    curve, value = -np.exp(-curve), -np.exp(-value)
                label = "LCB" if isinstance(settings.acquisition, LCB) else "-EI"
                ax.plot(u, curve, color="k", label=label)
                ax.scatter(
                    trial, value, facecolors="none", edgecolors="r", label="trial point"
                )
                ax.set_ylabel(label)
                continue

            # A panel of values: its curves, the true curve and the points
            values, true, label = history.f, f_true, "f"
            if panel == "gp" and is_hybrid:
                values, true, label = history.y[:, 0], y_true, system.y[0].name
            if panel == "gp":
                _band(ax, u, *predict(run.surrogates[i][0], u[:, None]))
            elif panel == "scenarios":
                assert run.scenarios is not None and f_scenarios is not None
                if f_std is None:
                    _scenario_band(ax, u, f_scenarios, run.scenarios.w)
                else:
                    _nominal_band(ax, u, f_scenarios[:, 0], f_std)
            if true is not None:
                ax.plot(u, true, color="k", linestyle="--", label="true")
            _points(ax, history.u[:, 0], values, history.f, history.n_initial, n)
            ax.set_ylabel(label)
        axes[-1, 0].set_xlabel(system.u[0].name)
        _figure_legend(figure, list(axes[:, 0]))
    return figure


@dataclass(frozen=True)
class _Panel:
    """A panel of the 2D figure: its title and its values over the grid (n_grid,);
    `in_f_units` panels may get a log color scale, `ei` is EI, drawn as log10 EI
    with its zeros in gray, and `note` is drawn in a corner."""

    title: str
    values: np.ndarray
    in_f_units: bool = False
    ei: bool = False
    note: str = ""


def _iteration_2d(
    run: Run,
    problem: Problem,
    i: int,
    measure: Measure | None,
    shown: dict[str, bool],
) -> "Figure":
    """`iteration`'s figure for two decision variables: filled contours in rows, a
    column each for the oracle, the mean, the std and the acquisition function."""
    system, history, settings = problem.system, run.history, run.settings
    n = history.n_initial + i

    # The grid over the start bounds, as a list of points in u's plane
    lower, upper = _start_bounds(system.u)
    u1, u2 = (np.linspace(lower[j], upper[j], _N_GRID_2D) for j in range(2))
    U1, U2 = np.meshgrid(u1, u2)
    grid = np.column_stack([U1.ravel(), U2.ravel()])
    y_true, f_true = _truth(problem, measure, grid)

    # The replay, from the trial point (the first row, split off after) over the grid
    curve = f_scenarios = f_std = None
    at_trial = np.nan
    if settings is not None and (shown["scenarios"] or shown["acquisition"]):
        u = np.vstack([history.u[n], grid])
        curve, f_scenarios, f_std = _replay(run, problem, i, settings, u)
        at_trial, curve = curve[0], curve[1:]
        f_scenarios = None if f_scenarios is None else f_scenarios[1:]
        f_std = None if f_std is None else f_std[1:]

    # The row of y (hybrid BO's GP), oracle | GP mean | GP std
    rows: list[list[_Panel | None]] = []
    if isinstance(settings, HybridBOSettings) and shown["gp"]:
        name = system.y[0].name
        mean, std = predict(run.surrogates[i][0], grid)
        oracle = None if y_true is None else _Panel(f"oracle {name}", y_true)
        rows.append(
            [
                oracle,
                _Panel(f"GP mean of {name}", mean),
                _Panel(f"GP std of {name}", std),
            ]
            + [None]
        )

    # The row of f: oracle | GP or scenario mean | its std | acquisition function
    row: list[_Panel | None] = [None] * 4
    if f_true is not None:
        row[0] = _Panel("oracle f", f_true, in_f_units=True)
    if isinstance(settings, StandardBOSettings) and shown["gp"]:
        mean, std = predict(run.surrogates[i][0], grid)
        row[1] = _Panel("GP mean of f", mean, in_f_units=True)
        row[2] = _Panel("GP std of f", std, in_f_units=True)
    if shown["scenarios"]:
        assert run.scenarios is not None and f_scenarios is not None
        if f_std is None:
            w = run.scenarios.w
            mean = f_scenarios @ w
            std = np.sqrt((f_scenarios - mean[:, np.newaxis]) ** 2 @ w)
            row[1] = _Panel("mean f over scenarios", mean, in_f_units=True)
            row[2] = _Panel("std f over scenarios", std, in_f_units=True)
        else:
            row[1] = _Panel("nominal f", f_scenarios[:, 0], in_f_units=True)
            row[2] = _Panel("s, the std of f", f_std, in_f_units=True)
    if shown["acquisition"]:
        assert settings is not None and curve is not None
        stored = run.acquisitions[i].value
        row[3] = _acquisition_panel(settings, curve, stored, at_trial)
    rows.append(row)

    # Drop the empty rows and columns
    rows = [row for row in rows if any(row)]
    columns = [c for c in range(4) if any(row[c] for row in rows)]

    # Every panel shows the points so far, the incumbent and the trial point
    points = (history.u[:, 0], history.u[:, 1], history.f, history.n_initial, n)
    import matplotlib.pyplot as plt

    with plt.style.context(STYLE):
        figure, axes = plt.subplots(
            max(len(rows), 1),
            max(len(columns), 1),
            squeeze=False,
            figsize=(2 + 4 * max(len(columns), 1), 1 + 3.4 * max(len(rows), 1)),
            layout="constrained",
        )
        figure.suptitle(f"{run.method}, iteration {i}")

        # Without panels: the points alone
        if not rows:
            ax = axes[0, 0]
            ax.set_title("points")
            ax.set_box_aspect(1)
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            _points(ax, *points)
            ax.set_xlabel(system.u[0].name)
            ax.set_ylabel(system.u[1].name)

        # Each panel; an oracle panel and the mean beside it share a color scale
        for r, row in enumerate(rows):
            shared = [panel.values for panel in row[:2] if panel is not None]
            for a, c in enumerate(columns):
                ax, panel = axes[r, a], row[c]
                if panel is None:
                    ax.remove()
                    continue
                scale = np.concatenate(shared) if c < 2 else panel.values
                if panel.ei:
                    _ei_contours(ax, U1, U2, panel.values)
                else:
                    _contours(ax, U1, U2, panel.values, scale, panel.in_f_units)
                ax.set_title(panel.title)
                if panel.note:
                    ax.text(
                        0.02,
                        0.02,
                        panel.note,
                        transform=ax.transAxes,
                        fontsize="small",
                        bbox={"facecolor": "white", "alpha": 0.8},
                    )
                _points(ax, *points)

                # Axis labels on the outer side of the panels
                below = rows[r + 1][c] if r + 1 < len(rows) else None
                if below is None:
                    ax.set_xlabel(system.u[0].name)
                if a == 0 or row[columns[a - 1]] is None:
                    ax.set_ylabel(system.u[1].name)
        _figure_legend(figure, [ax for ax in figure.axes if ax.get_title()])
    return figure


def _acquisition_panel(
    settings: MethodSettings, curve: np.ndarray, stored: float, replayed: float
) -> _Panel:
    """The acquisition panel of the 2D figure from the replayed acquisition values
    over the grid: LCB as it is (linear), else log10 EI (`_log10_ei`). Its note gives
    the stored and the replayed value at the trial point in the panel's units."""
    if isinstance(settings.acquisition, LCB):
        title, values, at_trial = "LCB", curve, np.array([stored, replayed])
    else:
        title = "log10 EI"
        values = _log10_ei(settings, curve)
        at_trial = _log10_ei(settings, np.array([stored, replayed]))
    note = f"trial point: stored {at_trial[0]:.4g}, replayed {at_trial[1]:.4g}"
    return _Panel(title, values, ei=title != "LCB", note=note)


def _contours(
    ax: "Axes",
    U1: np.ndarray,
    U2: np.ndarray,
    values: np.ndarray,
    scale: np.ndarray,
    in_f_units: bool,
    extend: str = "neither",
) -> None:
    """Filled contours of values (n_grid,) over the grid (U1, U2), with a colorbar.
    The color scale spans the finite values of `scale`; it is log if `in_f_units`
    and they are positive and span more than `_DECADES` decades. NaN leaves gaps.
    `extend` is matplotlib's: "min" colors values below the scale with its lowest
    color and puts an arrow on the colorbar."""
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogLocator, MaxNLocator

    ax.set_box_aspect(1)
    finite = scale[np.isfinite(scale)]
    if finite.size == 0:
        return
    low, high = float(finite.min()), float(finite.max())
    if in_f_units and low > 0 and high > low * 10**_DECADES:
        levels, norm = np.geomspace(low, high, _N_LEVELS), LogNorm(low, high)
        ticks = LogLocator()
    else:
        if high == low:
            low, high = low - 0.5, high + 0.5
        levels, norm = np.linspace(low, high, _N_LEVELS), Normalize(low, high)
        ticks = MaxNLocator()
    Z = np.ma.masked_invalid(values.reshape(U1.shape))
    contours = ax.contourf(U1, U2, Z, levels=levels, norm=norm, extend=extend)
    ax.figure.colorbar(contours, ax=ax, ticks=ticks)


def _log10_ei(settings: MethodSettings, values: np.ndarray) -> np.ndarray:
    """log10 EI from acquisition values: SoftSAA stores -log EI, so -value / ln 10,
    finite even where EI underflows in float64; the other EI approximations store
    -EI, so log10(-value), -inf where EI <= 0. NaN (a failed replay) stays NaN."""
    if _is_soft(settings):
        return -values / np.log(10)
    with np.errstate(divide="ignore"):
        return np.log10(np.maximum(-values, 0))


def _ei_contours(
    ax: "Axes", U1: np.ndarray, U2: np.ndarray, log_ei: np.ndarray
) -> None:
    """log10 EI (n_grid,) over the grid as filled contours, with the grid points where
    it is -inf (EI = 0) in gray cells; NaN leaves gaps. SAA-EI is exactly 0 where no
    scenario improves; analytic EI rounds to just below 0 far from the data, which
    counts as 0. SoftSAA-EI is never 0, so it has no gray cells. The color scale
    stops `_EI_DECADES` decades below the largest EI, with the smaller EI in the
    lowest color."""
    from matplotlib.colors import ListedColormap

    # Gray cells where EI is 0, with a legend entry
    zero = np.isneginf(log_ei)
    if zero.any():
        gray = np.ma.masked_array(np.ones(U1.shape), mask=~zero.reshape(U1.shape))
        cmap = ListedColormap([_ZERO_EI_COLOR])
        ax.pcolormesh(U1, U2, gray, cmap=cmap, shading="nearest")
        ax.fill([], [], color=_ZERO_EI_COLOR, label="EI = 0")

    # The color scale: at most `_EI_DECADES` decades, an arrow if values fall below
    finite = log_ei[np.isfinite(log_ei)]
    floor = finite.max() - _EI_DECADES if finite.size else -np.inf
    below = finite.size > 0 and finite.min() < floor
    scale = np.where(np.isfinite(log_ei), np.maximum(log_ei, floor), np.nan)
    extend = "min" if below else "neither"
    _contours(ax, U1, U2, log_ei, scale, in_f_units=False, extend=extend)


def _is_soft(settings: MethodSettings) -> bool:
    """Whether the settings are hybrid BO's with SoftSAA."""
    return isinstance(settings, HybridBOSettings) and isinstance(
        settings.approximation, SoftSAA
    )


def _replay(
    run: Run,
    problem: Problem,
    i: int,
    settings: MethodSettings,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Iteration i's acquisition function at each row of u (n_points, N), which holds
    the trial point; for hybrid BO f under each scenario there (n_points, S), else
    None; and with linearization the std s of f there (n_points,), else None. NaN
    where a solve failed. See `iteration` for how hybrid BO's is solved."""
    history, surrogate = run.history, run.surrogates[i]
    n = history.n_initial + i
    incumbent = float(history.f[:n].min())

    # Standard BO: its analytic acquisition function
    if isinstance(settings, StandardBOSettings):
        nlp = standard_acquisition(
            problem, surrogate[0], incumbent, settings.acquisition
        ).nlp
        values = np.array([nlp.f(_pack(point)).full().item() for point in u])
        return values, None, None

    # Hybrid BO: the scenarios' states at each u, root-found from the stored solution
    assert run.scenarios is not None  # hybrid BO runs hold their scenarios
    acquisition_problem = hybrid_acquisition(
        problem,
        surrogate,
        run.scenarios,
        incumbent,
        settings.acquisition,
        settings.approximation,
    )
    solution = run.acquisitions[i]
    assert solution.y is not None
    at_trial = int(np.flatnonzero(np.all(u == history.u[n], axis=1))[0])
    x, y, solved = _scenario_roots(
        problem, surrogate, run.scenarios, acquisition_problem, u, at_trial, solution
    )

    # The acquisition problem's objective, and f under each scenario, in one call each
    # over the solved points
    values = np.full(len(u), np.nan)
    f = np.full((len(u), len(run.scenarios.w)), np.nan)
    f_std = np.full(len(u), np.nan)
    sensitivity = None
    if isinstance(settings.approximation, Linearization):
        sensitivity = _sensitivity(problem, surrogate)
    points = np.flatnonzero(solved)
    y_solved = np.empty((0, *solution.y.shape))
    if len(points) > 0:
        w = [_pack(u[k], x[k], None if y is None else y[k]) for k in points]
        nlp = acquisition_problem.nlp
        values[points] = nlp.f.map(len(w))(np.column_stack(w)).full().ravel()
        objective = _scenario_objective(
            problem, surrogate, run.scenarios, y is not None
        )
        inputs = [u[points].T, np.hstack(list(x[points]))]
        if y is not None:
            inputs.append(np.hstack(list(y[points])))
        f_solved, y_mapped = objective.map(len(w))(*inputs)
        f[points] = f_solved.full().T
        # Mapped outputs sit side by side, (S, n_points * P)
        n_scenarios, n_y = solution.y.shape
        y_solved = y_mapped.full().reshape(n_scenarios, len(w), n_y).transpose(1, 0, 2)

    # The check: at the trial point, the stored solution's states and value
    replayed = None
    if solved[at_trial]:
        y_trial = y_solved[np.searchsorted(points, at_trial)]
        replayed = AcquisitionSolution(float(values[at_trial]), x[at_trial], y_trial)
    _check_trial_point(replayed, solution)
    if sensitivity is None:
        return values, f, None

    # Linearization's std of f, from the sensitivity at the nominal states
    for j, k in enumerate(points):
        f_std[k] = np.linalg.norm(sensitivity(x[k, 0], y_solved[j, 0], u[k]).full())
    return values, f, f_std


def _scenario_equations(
    problem: Problem, fitted: tuple[FittedGP, ...], unknown_states_as_variables: bool
) -> cas.Function:
    """One scenario's equations with u fixed, a square system in its states v: a
    function of columns v, u (N, 1) and the scenario's draw z (P, 1) to the residual.
    With the unknown states substituted, v = x and the residual is
    g(x, mean + std * z, u) (M); as variables, v = (x, y) and it is g(x, y, u) and
    y - mean - std * z (M + P), with mean and std the GPs' at the scenario's arguments
    of h, from `fitted`, one fitted GP per unknown state."""
    system = problem.system
    x = Symbolic.sym("x", system.n_x)
    u = Symbolic.sym("u", system.n_u)
    z = Symbolic.sym("z", system.n_y)
    inputs = cas.horzcat(*_h_inputs(system, u.T, x.T))
    means, stds = zip(*(predict(gp, inputs) for gp in fitted), strict=True)
    predicted = cas.vertcat(*means) + cas.vertcat(*stds) * z
    if not unknown_states_as_variables:
        residual = system.g(x, predicted, u)
        return cas.Function("scenario_equations", [x, u, z], [residual])
    y = Symbolic.sym("y", system.n_y)
    residual = cas.vertcat(system.g(x, y, u), y - predicted)
    return cas.Function("scenario_equations", [cas.vertcat(x, y), u, z], [residual])


def _scenario_roots(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    acquisition_problem: AcquisitionProblem,
    u: np.ndarray,
    at_trial: int,
    solution: AcquisitionSolution,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """The scenario states at each row of u (n_points, N): x (n_points, S, M), y
    (n_points, S, P) (None unless they are variables), and whether each point was
    solved (n_points,). With u fixed, each scenario's equations are a square system
    (`_scenario_equations`), solved by Newton's method, mapped over the (point,
    scenario) pairs of a ring of points around row `at_trial`, the trial point.

    The rings go outwards from the trial point, one grid spacing wide (in units of the
    start bounds), each point started from the nearest point solved in an earlier ring,
    the trial point's ring from the stored solution. That follows the branch the
    optimizer chose. A pair fails, and so its point, if Newton's residual is above
    `_RESIDUAL_TOLERANCE` or the root is outside a state's bounds (y's only when they
    are variables, as in the acquisition problem). Where the equations have several
    roots this can leave gaps IPOPT wouldn't: it stays within the bounds, so it could
    find a root inside them where Newton found one outside, and past a fold, where
    the branch ends, it could jump to another branch where Newton stalls.
    """
    # The unknowns of each pair, their bounds, and the stored solution as a start
    assert solution.x is not None and solution.y is not None
    system = problem.system
    as_variables = acquisition_problem.unknown_states_as_variables
    n_scenarios, n_x = len(scenarios.w), system.n_x
    nlp = acquisition_problem.nlp
    _, x_lower, y_lower = _unpack(acquisition_problem, nlp.lower)
    _, x_upper, y_upper = _unpack(acquisition_problem, nlp.upper)
    v_lower, v_upper, stored = x_lower, x_upper, solution.x
    if as_variables:
        assert y_lower is not None and y_upper is not None
        v_lower, v_upper = np.hstack([x_lower, y_lower]), np.hstack([x_upper, y_upper])
        stored = np.hstack([solution.x, solution.y])
    roots = np.full((len(u), n_scenarios, v_lower.shape[1]), np.nan)
    roots[at_trial] = stored
    solved = np.zeros(len(u), dtype=bool)

    # With no unknowns (no mechanistic states, the GP substituted) nothing is solved
    if roots.shape[2] == 0:
        return roots, None, ~solved

    # The equations and Newton's method on them; a mapped call reports no success per
    # instance, so failures are found from the residual
    equations = _scenario_equations(problem, fitted, as_variables)
    newton = cas.rootfinder("replay", "newton", equations, {"error_on_fail": False})

    # The rings: the distance from the trial point in units of the median distance
    # between nearest points (1 if that is 0 or there is one point)
    lower, upper = _start_bounds(system.u)
    scaled = (u - lower) / (upper - lower)
    distances = np.linalg.norm(scaled[:, np.newaxis] - scaled[np.newaxis], axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    spacing = spacing if 0 < spacing < np.inf else 1.0
    from_trial = np.linalg.norm(scaled - scaled[at_trial], axis=1)
    ring = np.floor(from_trial / spacing).astype(int)

    # Solve ring by ring, every pair of a ring in one call
    has_start = np.zeros(len(u), dtype=bool)
    has_start[at_trial] = True
    for r in np.unique(ring):
        points = np.flatnonzero(ring == r)
        known = np.flatnonzero(has_start)
        nearest = known[np.argmin(distances[np.ix_(points, known)], axis=1)]
        nearest[points == at_trial] = at_trial
        n_pairs = len(points) * n_scenarios
        start = roots[nearest].reshape(n_pairs, -1).T
        u_pairs = np.repeat(u[points], n_scenarios, axis=0).T
        z_pairs = np.tile(scenarios.z, (len(points), 1)).T
        v = newton.map(n_pairs)(start, u_pairs, z_pairs).full()
        residual = equations.map(n_pairs)(v, u_pairs, z_pairs).full()

        # A point is solved if every scenario converged within the bounds
        v = v.T.reshape(len(points), n_scenarios, -1)
        residual = np.abs(residual.T).reshape(len(points), n_scenarios, -1)
        converged = np.all(residual <= _RESIDUAL_TOLERANCE, axis=(1, 2))
        inside = np.all((v >= v_lower) & (v <= v_upper), axis=(1, 2))
        ring_solved = points[converged & inside]
        roots[ring_solved] = v[converged & inside]
        solved[ring_solved] = has_start[ring_solved] = True

    # The roots of the points not solved are no roots
    roots[~solved] = np.nan
    return roots[:, :, :n_x], roots[:, :, n_x:] if as_variables else None, solved


def _check_trial_point(
    replayed: AcquisitionSolution | None, solution: AcquisitionSolution
) -> None:
    """Warns with `ReplayWarning` unless the replay at the trial point (None if it
    wasn't solved), its acquisition value and scenario states, is the stored
    solution, to `_REPLAY_RTOL` relative and `_REPLAY_ATOL` absolute. The stack
    level points the warning at the caller of `iteration`."""
    if replayed is None:
        msg = (
            "The replay found no root at the trial point, so the acquisition panel "
            "may not show the function the run optimized"
        )
        warnings.warn(msg, ReplayWarning, stacklevel=5)
        return

    # The value and the states as one vector each, compared entry by entry
    def entries(acquisition: AcquisitionSolution) -> np.ndarray:
        """The value, x and y of a solution in one vector."""
        assert acquisition.x is not None and acquisition.y is not None
        parts = (acquisition.value, acquisition.x, acquisition.y)
        return np.concatenate([np.ravel(part) for part in parts])

    replayed_entries, stored_entries = entries(replayed), entries(solution)
    if not np.allclose(
        replayed_entries, stored_entries, rtol=_REPLAY_RTOL, atol=_REPLAY_ATOL
    ):
        msg = (
            f"The replay's acquisition value at the trial point is {replayed.value:.6g}, "
            f"stored {solution.value:.6g}, or its states differ: the root-finder left "
            "the branch the optimizer chose, so the acquisition panel may not show "
            "the function the run optimized"
        )
        warnings.warn(msg, ReplayWarning, stacklevel=5)


def _scenario_objective(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    unknown_states_as_variables: bool,
) -> cas.Function:
    """f under each scenario of a hybrid acquisition problem, and the scenario unknown
    states, as a function of u (N, 1), x (S, M) and, if they are variables, y (S, P)
    to f (S, 1) and y (S, P); otherwise y is the GPs' scenario prediction, from
    `fitted`, one fitted GP per unknown state."""
    system = problem.system
    u = Symbolic.sym("u", system.n_u, 1)
    x = Symbolic.sym("x", len(scenarios.w), len(system.x))
    inputs = [u, x]
    if unknown_states_as_variables:
        y = Symbolic.sym("y", len(scenarios.w), len(system.y))
        inputs.append(y)
    else:
        y = _scenario_unknown_states(system, fitted, scenarios, u, x)
    f = cas.vertcat(
        *(problem.f(x[s, :].T, y[s, :].T, u) for s in range(len(scenarios.w)))
    )
    return cas.Function("scenario_objective", inputs, [f, y])


def _figure_legend(figure: "Figure", axes: list["Axes"]) -> None:
    """One legend for the figure, an entry per label across the axes, outside them on
    the right."""
    handles: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    figure.legend(list(handles.values()), list(handles), loc="outside right upper")


def _points(
    ax: "Axes", u: np.ndarray, values: np.ndarray, f: np.ndarray, n_initial: int, n: int
) -> None:
    """The first n points (initial points black, the rest red), the incumbent's point
    among them (by the measured f) as a star, and point n, the trial point (design
    point, for the baselines), as a red ring."""
    ax.scatter(u[:n_initial], values[:n_initial], c="k", label="initial points")
    ax.scatter(u[n_initial:n], values[n_initial:n], c="r", label="points so far")
    best = int(np.argmin(f[:n]))
    ax.scatter(u[best], values[best], marker="*", s=150, c="gold", label="incumbent")
    ax.scatter(u[n], values[n], facecolors="none", edgecolors="r", label="trial point")


def _band(ax: "Axes", u: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    """A GP's mean and the band holding `_CONFIDENCE` of its probability over u."""
    ax.plot(u, mean, color="b", label="GP mean")
    half_width = scipy.stats.norm.ppf(0.5 + _CONFIDENCE / 2) * std
    label = f"{_CONFIDENCE:.0%} band"
    ax.fill_between(
        u, mean - half_width, mean + half_width, alpha=0.2, color="b", label=label
    )


def _nominal_band(ax: "Axes", u: np.ndarray, f: np.ndarray, s: np.ndarray) -> None:
    """Linearization's f at the nominal states over u and the band f +- s, its std;
    NaN leaves gaps."""
    ax.plot(u, f, color="b", label="nominal f")
    ax.fill_between(u, f - s, f + s, alpha=0.2, color="b", label="nominal f ± s")


def _scenario_band(ax: "Axes", u: np.ndarray, f: np.ndarray, w: np.ndarray) -> None:
    """f under the scenarios (n_u, S) over u: the weighted mean and the band holding
    the central `_CONFIDENCE` of the scenario weight; NaN rows leave gaps."""
    ax.plot(u, f @ w, color="b", label="scenario mean")
    tail = (1 - _CONFIDENCE) / 2
    band = np.full((len(u), 2), np.nan)
    for row, f_row in enumerate(f):
        if not np.isnan(f_row).any():
            band[row] = np.quantile(
                f_row, [tail, 1 - tail], weights=w, method="inverted_cdf"
            )
    label = f"{_CONFIDENCE:.0%} band"
    ax.fill_between(u, band[:, 0], band[:, 1], alpha=0.2, color="b", label=label)
