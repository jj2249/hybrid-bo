"""Static plots of a study: convergence and regret across runs, and one iteration of a
run.

Plots read finished runs, so a saved study is plotted without running it again (ADR
0003); an iteration's acquisition panels replay the method's acquisition problem (ADR
0005). matplotlib and Polars are imported on first use. Figures are drawn in the repo's
style, `STYLE`; use it elsewhere with `plt.style.use(hb.plots.STYLE)`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.acquisition import (
    LCB,
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
from hybrid_bo.evaluation import Measure, evaluate
from hybrid_bo.gp import FittedGP, predict
from hybrid_bo.history import n_iterations
from hybrid_bo.problem import Problem, Symbolic
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
_CONFIDENCE = 0.9  # probability inside the confidence bands


def convergence(study: Study, ax: "Axes | None" = None) -> "Axes":
    """Per method, the median over its runs of the oracle incumbent (or, in a study
    without an oracle, the incumbent) at each evaluation, with the interquartile
    range shaded, a dotted line after the initial points, and a legend.

    Raises `ValueError` if the study has no evaluations, and `ImportError` without
    Polars.
    """
    rows = evaluations(study)
    has_oracle = not rows.is_empty() and rows["oracle_incumbent"].null_count() == 0
    column = "oracle_incumbent" if has_oracle else "incumbent"
    return _over_evaluations(rows, column, study.settings.n_initial, ax)


def regret(
    study: Study, optimum: float | None = None, ax: "Axes | None" = None
) -> "Axes":
    """The median regret, drawn as in `convergence` on a log y-axis (the paper's
    figure). The oracle optimum is `optimum` if given, else the study's recorded one.

    Raises `ValueError` if the study has no regret (no oracle, or no optimum), and
    `ImportError` without Polars.
    """
    rows = evaluations(study, optimum)
    ax = _over_evaluations(rows, "regret", study.settings.n_initial, ax)
    ax.set_yscale("log")
    return ax


def _over_evaluations(
    rows: "pl.DataFrame", column: str, n_initial: int, ax: "Axes | None"
) -> "Axes":
    """`convergence`'s figure of `column`, on `ax` or a new figure's axes. A shorter
    run counts up to its last evaluation.

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

    ax.axvline(n_initial + 0.5, color="grey", linestyle=":")
    ax.set_xlabel("evaluations")
    ax.set_ylabel(column.replace("_", " "))
    ax.legend()
    return ax


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

    - `gp`: the stored surrogate's mean and band; of y for hybrid BO, of f for
      standard BO.
    - `scenarios` (hybrid BO): f under the scenarios, their weighted mean and band;
      with linearization, f at the nominal states and the band f +- s, with s
      recomputed from the stored surrogate.
    - `acquisition`: the minimized acquisition function (-EI or LCB) and the stored
      value at the trial point. Standard BO's is analytic. Hybrid BO's replays the
      acquisition problem (ADR 0005) from the stored surrogate, scenarios and
      incumbent, with u fixed at each grid point, solved outwards from the stored
      solution, each grid point from the nearest one solved; a failed solve leaves a
      gap. SoftSAA's -log EI is drawn as -EI.

    With one decision variable the panels are stacked curves over a grid of 101
    points, each with the points so far, the incumbent and the trial point; the
    baselines (or every panel off) show the points with f. With two, they are filled
    contours over a 30 x 30 grid in rows, of y (oracle, GP mean, GP std) and of f
    (oracle, the scenarios' weighted mean or the nominal f, their weighted std or s,
    and EI), or for standard BO one row of f with its GP; each shows the points so
    far, the incumbent and the trial point in the plane of u. An oracle panel shares
    the color scale of the mean beside it. Panels of f are on a log color scale when
    their values are positive and span more than two decades. EI is drawn as log10
    EI, with the grid points where it is 0 (SAA-EI's flat regions) in gray;
    LCB is drawn as it is. The acquisition panel notes the stored and the replayed
    value at the trial point, which check the replay.

    `measure` adds the oracle values (curves, or the oracle panels in 2D); it runs at
    every grid point (generator seeded with 0), so pass an oracle measurement, never
    a real system.

    The GP is the run's stored fitted GP; the rest of the replay (the acquisition
    function, the approximation, the solver) comes from the run's stored settings.

    Raises `ValueError` unless the problem has one or two decision variables (and,
    for hybrid BO's GP panels, h takes only u), or if `measure` has no oracle; and
    `IndexError` if the run has no iteration `i`.
    """
    # The checks: one or two decision variables, the GP of y over u, the iteration
    system, history, settings = problem.system, run.history, run.settings
    if system.n_u not in (1, 2):
        msg = f"The plots need one or two decision variables, got {system.n_u}"
        raise ValueError(msg)
    is_hybrid = isinstance(settings, HybridBOSettings)
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
    """The oracle y and f at the rows of u, or None without a measurement.

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
                _band(ax, u, *predict(run.surrogates[i], u[:, None]))
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
        mean, std = predict(run.surrogates[i], grid)
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
        mean, std = predict(run.surrogates[i], grid)
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
    over the grid: LCB as it is (linear), else EI (-value, or exp(-value) for
    SoftSAA's -log EI). Its note gives the stored and the replayed value at the trial
    point in the panel's units."""
    if isinstance(settings.acquisition, LCB):
        title, values, at_trial = "LCB", curve, np.array([stored, replayed])
    else:
        title = "log10 EI"
        to_ei = (lambda v: np.exp(-v)) if _is_soft(settings) else np.negative
        values, at_trial = to_ei(curve), to_ei(np.array([stored, replayed]))
        with np.errstate(divide="ignore", invalid="ignore"):
            at_trial = np.log10(at_trial)
    note = f"trial point: stored {at_trial[0]:.4g}, replayed {at_trial[1]:.4g}"
    return _Panel(title, values, ei=title != "LCB", note=note)


def _contours(
    ax: "Axes",
    U1: np.ndarray,
    U2: np.ndarray,
    values: np.ndarray,
    scale: np.ndarray,
    in_f_units: bool,
) -> None:
    """Filled contours of values (n_grid,) over the grid (U1, U2), with a colorbar.
    The color scale spans the finite values of `scale`; it is log if `in_f_units`
    and they are positive and span more than `_DECADES` decades. NaN leaves gaps."""
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
    contours = ax.contourf(U1, U2, Z, levels=levels, norm=norm)
    ax.figure.colorbar(contours, ax=ax, ticks=ticks)


def _ei_contours(ax: "Axes", U1: np.ndarray, U2: np.ndarray, ei: np.ndarray) -> None:
    """log10 EI (n_grid,) over the grid as filled contours, with the grid points where
    EI is 0 in gray cells; NaN leaves gaps. SAA-EI is exactly 0 where no scenario
    improves; analytic EI rounds to just below 0 far from the data, which counts as 0.
    """
    from matplotlib.colors import ListedColormap

    zero = ei <= 0
    if zero.any():
        gray = np.ma.masked_array(np.ones(U1.shape), mask=~zero.reshape(U1.shape))
        cmap = ListedColormap([_ZERO_EI_COLOR])
        ax.pcolormesh(U1, U2, gray, cmap=cmap, shading="nearest")
        ax.fill([], [], color=_ZERO_EI_COLOR, label="EI = 0")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ei = np.where(ei > 0, np.log10(ei), np.nan)
    _contours(ax, U1, U2, log_ei, log_ei, in_f_units=False)


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
            problem, surrogate, incumbent, settings.acquisition
        ).nlp
        values = np.array([nlp.f(_pack(point)).full().item() for point in u])
        return values, None, None

    # Hybrid BO: the acquisition problem with u fixed by its bounds
    assert run.scenarios is not None  # hybrid BO runs hold their scenarios
    acquisition_problem = hybrid_acquisition(
        problem,
        surrogate,
        run.scenarios,
        incumbent,
        settings.acquisition,
        settings.approximation,
    )
    nlp = acquisition_problem.nlp
    assert nlp.g is not None  # hybrid BO's problem has the known equations
    w = Symbolic.sym("w", len(nlp.lower))
    solver = cas.nlpsol(
        "replay",
        "ipopt",
        {"x": w, "f": nlp.f(w), "g": nlp.g(w)},
        settings.solver.plugin_options | {"ipopt": settings.solver.solver_options},
    )
    _, x_lower, y_lower = _unpack(acquisition_problem, nlp.lower)
    _, x_upper, y_upper = _unpack(acquisition_problem, nlp.upper)

    # The stored solution's states, the first start
    solution = run.acquisitions[i]
    assert solution.x is not None and solution.y is not None
    as_variables = settings.approximation.unknown_states_as_variables
    y_stored = solution.y if as_variables else None

    # Linearization's std of f comes from the sensitivity at the nominal states
    sensitivity = None
    if isinstance(settings.approximation, Linearization):
        sensitivity = _sensitivity(problem, surrogate)

    # Solve at each u outwards from the trial point, in order of distance from it (in
    # units of the start bounds), each from the nearest solution so far. In 1D that is
    # each from its neighbor towards the trial point
    lower, upper = _start_bounds(problem.system.u)
    scaled = (u - lower) / (upper - lower)
    at_trial = int(np.flatnonzero(np.all(u == history.u[n], axis=1))[0])
    values = np.full(len(u), np.nan)
    states: dict[int, tuple[np.ndarray, np.ndarray | None]] = {
        at_trial: (solution.x, y_stored)
    }
    order = np.argsort(np.linalg.norm(scaled - scaled[at_trial], axis=1), kind="stable")
    for k in order:
        known = list(states)
        nearest = known[np.argmin(np.linalg.norm(scaled[known] - scaled[k], axis=1))]
        x_start, y_start = states[nearest]
        result = solver(
            x0=_pack(u[k], x_start, y_start),
            lbx=_pack(u[k], x_lower, y_lower),
            ubx=_pack(u[k], x_upper, y_upper),
            lbg=0,
            ubg=0,
        )
        if not solver.stats()["success"]:
            continue
        _, x_k, y_k = _unpack(acquisition_problem, result["x"].full().ravel())
        states[int(k)] = (x_k, y_k)
        values[k] = float(result["f"])

    # f under each scenario at the solutions, in one call over the solved points (the
    # stored solution, if the trial point's solve failed, isn't one)
    f = np.full((len(u), len(run.scenarios.w)), np.nan)
    f_std = np.full(len(u), np.nan)
    solved = np.flatnonzero(~np.isnan(values)).tolist()
    if len(solved) == 0:
        return values, f, None if sensitivity is None else f_std
    objective = _scenario_objective(problem, surrogate, run.scenarios, as_variables)
    x_solved = [states[k][0] for k in solved]
    inputs = [u[solved].T, np.hstack(x_solved)]
    if as_variables:
        inputs.append(np.hstack([y for k in solved if (y := states[k][1]) is not None]))
    f_solved, y_solved = objective.map(len(solved))(*inputs)
    f[solved] = f_solved.full().T
    if sensitivity is None:
        return values, f, None

    # Linearization's std of f, from the sensitivity at the nominal states
    n_y = len(problem.system.y)
    y_nominal = y_solved.full()[0].reshape(len(solved), n_y)
    for x_k, y_k, k in zip(x_solved, y_nominal, solved, strict=True):
        f_std[k] = np.linalg.norm(sensitivity(x_k[0], y_k, u[k]).full())
    return values, f, f_std


def _scenario_objective(
    problem: Problem,
    fitted: FittedGP,
    scenarios: Scenarios,
    unknown_states_as_variables: bool,
) -> cas.Function:
    """f under each scenario of a hybrid acquisition problem, and the scenario unknown
    states, as a function of u (N, 1), x (S, M) and, if they are variables, y (S, P)
    to f (S, 1) and y (S, P); otherwise y is the GP's scenario prediction."""
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
