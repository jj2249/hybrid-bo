"""Static plots of a study: convergence and regret across runs, and one iteration of a
run.

Plots are functions of finished runs, never callbacks during the loop, so a saved
study is plotted without running it again (ADR 0003). An iteration's acquisition
panels replay the method's acquisition problem from what the run stored (ADR 0005).
matplotlib (and Polars, for the curves over evaluations) is imported on first use, so
the library works without it. Figures are drawn in the repo's style, `STYLE` (a
matplotlib style file); use it for other figures with `plt.style.use(hb.plots.STYLE)`.
"""

import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.acquisition import (
    EI,
    LCB,
    _scenario_states,
    hybrid_acquisition,
    standard_acquisition,
)
from hybrid_bo.bo import HybridBOSettings, StandardBOSettings, _settings_text
from hybrid_bo.evaluation import Experiment, evaluate
from hybrid_bo.gp import GP, _gp_record, default_gp, predict
from hybrid_bo.problem import Problem, Symbolic
from hybrid_bo.run import Run
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.studies import Study
from hybrid_bo.views import evaluations

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# The repo's matplotlib style: its rcParams for every figure
STYLE = Path(__file__).with_name("hybrid_bo.mplstyle")

_N_GRID = 101  # grid points over u
_CONFIDENCE = 0.9  # probability inside the confidence bands

# The settings each BO method's runs are replayed with, by the method in the record
_SETTINGS = {"hybrid_bo": HybridBOSettings, "standard_bo": StandardBOSettings}


def _over_evaluations(
    rows: "pl.DataFrame", column: str, n_initial: int, ax: "Axes | None"
) -> "Axes":
    """Per method, the median over the runs of `column` at each evaluation, with the
    interquartile range shaded, a dotted line after the initial points, and a legend,
    on `ax` or a new figure's axes. A shorter run counts up to its last evaluation.

    Raises `ValueError` if the column has no values.
    """
    import matplotlib.pyplot as plt  # only to draw, so the library works without it
    import polars as pl  # `evaluations` has checked it is installed

    if column not in rows.columns or rows[column].null_count() == len(rows):
        msg = f"The study has no {column.replace('_', ' ')} to draw"
        raise ValueError(msg)
    if ax is None:
        with plt.style.context(STYLE):
            _, ax = plt.subplots()

    # Per method, sorted by evaluation: the median and the quartiles
    values = pl.col(column)
    curves = (
        rows.drop_nulls(column)
        .group_by("method", "evaluation", maintain_order=True)
        .agg(
            values.median().alias("median"),
            values.quantile(0.25, "linear").alias("lower"),
            values.quantile(0.75, "linear").alias("upper"),
        )
    )
    by_method = curves.sort("evaluation").group_by("method", maintain_order=True)
    for (method,), curve in by_method:
        n = curve["evaluation"].to_numpy()
        (drawn,) = ax.plot(n, curve["median"].to_numpy(), label=str(method))
        lower, upper = curve["lower"].to_numpy(), curve["upper"].to_numpy()
        ax.fill_between(n, lower, upper, color=drawn.get_color(), alpha=0.2)

    ax.axvline(n_initial + 0.5, color="grey", linestyle=":")
    ax.set_xlabel("evaluations")
    ax.set_ylabel(column.replace("_", " "))
    ax.legend()
    return ax


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
    """Per method, the median regret over the runs at each evaluation, as in
    `convergence`, on a log y-axis as in the paper. The oracle optimum is `optimum`
    when given, else the study's recorded one (`hb.evaluations`).

    Raises `ValueError` if the study has no regret (no oracle, or no optimum), and
    `ImportError` without Polars.
    """
    rows = evaluations(study, optimum)
    ax = _over_evaluations(rows, "regret", study.settings.n_initial, ax)
    ax.set_yscale("log")
    return ax


def iteration(
    run: Run,
    problem: Problem,
    i: int,
    *,
    experiment: Experiment | None = None,
    gp: bool = True,
    scenarios: bool = True,
    acquisition: bool = True,
    settings: StandardBOSettings | HybridBOSettings | None = None,
) -> "Figure":
    """The figure of iteration i of a run (from 0; negative counts from the end), the
    state before its i-th trial point (design point, for the baselines), over a grid
    of the decision variable's start bounds.

    Each panel of values shows the points so far (initial points black, the rest
    red), the incumbent's point (a star) and the next point (a red ring). The panels,
    each turned off by its flag:

    - `gp`: the stored surrogate's mean and confidence band; of y for hybrid BO, of f
      for standard BO.
    - `scenarios` (hybrid BO): f under the scenarios, their weighted mean and a band
      holding the central share of the scenario weight.
    - `acquisition`: the acquisition function as the acquisition problem minimizes it
      (-EI or LCB), and the stored value at the trial point. Standard BO's is
      analytic. Hybrid BO's replays the acquisition problem (ADR 0005), rebuilt from
      the stored surrogate, scenarios and incumbent and solved with u fixed at each
      grid point, outwards from the stored solution at the trial point, each solve
      started from its neighbor's; where a solve fails, the panels have a gap.

    The baselines (and a figure with every panel off) show the points with f.
    `experiment` adds the oracle's true curves, dashed; it runs at every grid point
    with a generator seeded with 0, so pass an oracle experiment, never a real system.

    The acquisition function is read from the run's record. The record holds the GP
    as text, not its kernel function, so a run with a GP other than `default_gp`
    needs its `settings`.

    Raises `ValueError` unless the problem has one decision variable (and, for hybrid
    BO's GP panel, h takes only u), if the settings or the default GP don't match
    the record, or if `experiment` gives no oracle values; `TypeError` if the
    settings aren't the run's method's; and `IndexError` if the run has no iteration
    `i`.
    """
    # The checks: one decision variable, the GP of y over u, the iteration
    system, history = problem.system, run.history
    if system.n_u != 1:
        msg = f"The plots need one decision variable, got {system.n_u}"
        raise ValueError(msg)
    is_hybrid = run.record.get("method") == "hybrid_bo"
    if is_hybrid and gp and list(system.h_inputs) != list(system.u):
        msg = "The GP of y is drawn over u, so h_inputs must be the decision variable"
        raise ValueError(msg)
    n_iterations = len(history.u) - history.n_initial
    if not -n_iterations <= i < n_iterations:
        msg = f"The run has {n_iterations} iterations, so no iteration {i}"
        raise IndexError(msg)
    i %= n_iterations
    settings, gp_spec = _replay_settings(run, problem, settings)

    # The grid, with the trial point in it so the replay starts there
    n, n_initial = history.n_initial + i, history.n_initial
    trial = history.u[n, 0]
    u = np.sort(np.append(np.linspace(*_start_bounds(system.u), _N_GRID), trial))

    # The true curves: the oracle values at every grid point
    y_true = f_true = None
    if experiment is not None:
        truth = evaluate(problem, experiment, u[:, None], np.random.default_rng(0))
        if truth.f_oracle is None or truth.y_oracle is None:
            msg = "The true curves need an oracle experiment"
            raise ValueError(msg)
        y_true, f_true = truth.y_oracle[:, 0], truth.f_oracle

    # The panels this method has, in order, and the replay they need
    is_bo = settings is not None
    flags = {"gp": gp and is_bo, "scenarios": scenarios and is_hybrid}
    flags["acquisition"] = acquisition and is_bo
    panels = [panel for panel, shown in flags.items() if shown] or ["points"]
    acquisition_curve = f_scenarios = None
    if settings is not None and (flags["scenarios"] or flags["acquisition"]):
        acquisition_curve, f_scenarios = _replay(run, problem, i, settings, gp_spec, u)

    # The figure; matplotlib is imported only to draw
    import matplotlib.pyplot as plt

    with plt.style.context(STYLE):
        figure, axes = plt.subplots(
            len(panels),
            1,
            sharex=True,
            figsize=(8, 2 + 2 * len(panels)),
            squeeze=False,
            layout="constrained",
        )
        figure.suptitle(f"{run.method}, iteration {i}")
        for panel, ax in zip(panels, axes[:, 0], strict=True):
            values, true, label = history.f, f_true, "f"
            if panel == "gp":
                if is_hybrid:
                    values, true, label = history.y[:, 0], y_true, system.y[0].name
                mean, std = predict(gp_spec, run.surrogates[i], u[:, None])
                _gp_panel(ax, u, mean, std)
            elif panel == "scenarios":
                assert run.scenarios is not None and f_scenarios is not None
                _scenarios_panel(ax, u, f_scenarios, run.scenarios.w)
            elif panel == "acquisition":
                assert settings is not None and acquisition_curve is not None
                label = "LCB" if isinstance(settings.acquisition, LCB) else "-EI"
                ax.plot(u, acquisition_curve, color="k", label=label)
                ax.scatter(
                    trial,
                    run.acquisitions[i].value,
                    facecolors="none",
                    edgecolors="r",
                    label="next point",
                )
                ax.set_ylabel(label)
                continue

            # The true curve and the points, on every panel of values
            if true is not None:
                ax.plot(u, true, color="k", linestyle="--", label="true")
            _points(ax, history.u[:, 0], values, history.f, n_initial, n)
            ax.set_ylabel(label)
        axes[-1, 0].set_xlabel(system.u[0].name)
        _figure_legend(figure, list(axes[:, 0]))
    return figure


def _replay_settings(
    run: Run,
    problem: Problem,
    settings: StandardBOSettings | HybridBOSettings | None,
) -> tuple[StandardBOSettings | HybridBOSettings | None, GP]:
    """The settings a run is replayed with, and its GP spec: `settings` if given,
    else the acquisition function and `unknown_states_as_variables` from the record
    with `default_gp`. None and `default_gp` for the baselines.

    Raises `TypeError` if `settings` aren't the run's method's, and `ValueError` if
    they, or the GP, don't match the record.
    """
    # The method's settings type; the baselines have none
    method = run.record.get("method", "")
    expected = _SETTINGS.get(method)
    system = problem.system
    gp_inputs = system.h_inputs if expected is HybridBOSettings else system.u
    if settings is not None and (
        expected is None or not isinstance(settings, expected)
    ):
        name = "no" if expected is None else expected.__name__
        msg = f"A {method} run takes {name} settings, got {type(settings).__name__}"
        raise TypeError(msg)
    if expected is None:
        return None, default_gp(len(gp_inputs))

    # The settings: given ones must be the recorded ones; else read from the record
    recorded = re.sub(r"^n_iterations=\d+, ", "", run.record.get("settings", ""))
    if settings is not None and _settings_text(settings) != recorded:
        msg = f"The settings don't match the run's record: {recorded}"
        raise ValueError(msg)
    if settings is None:
        lcb = re.search(r"acquisition=LCB\(std_factor=([^)]*)\)", recorded)
        settings = expected(acquisition=EI() if lcb is None else LCB(float(lcb[1])))
        if isinstance(settings, HybridBOSettings):
            as_variables = "unknown_states_as_variables=False" not in recorded
            settings = replace(settings, unknown_states_as_variables=as_variables)

    # The GP spec, whose text must be the recorded one
    gp_spec = default_gp(len(gp_inputs)) if settings.gp is None else settings.gp
    gp_text = _gp_record(gp_spec, [var.name for var in gp_inputs])
    if any(run.record.get(key) != text for key, text in gp_text.items()):
        msg = "The run's GP isn't default_gp; pass the settings it was run with"
        raise ValueError(msg)
    return settings, gp_spec


def _replay(
    run: Run,
    problem: Problem,
    i: int,
    settings: StandardBOSettings | HybridBOSettings,
    gp_spec: GP,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Iteration i's acquisition function at each u (n_u,), which holds the trial
    point, and for hybrid BO f under each scenario there (n_u, S), else None; NaN
    where a solve failed. See `iteration` for how hybrid BO's is solved."""
    history, surrogate = run.history, run.surrogates[i]
    n = history.n_initial + i
    incumbent = float(history.f[:n].min())

    # Standard BO: its analytic acquisition function
    if isinstance(settings, StandardBOSettings):
        nlp, _ = standard_acquisition(
            problem, gp_spec, surrogate, incumbent, settings.acquisition
        )
        return np.array([nlp.f(value).full().item() for value in u]), None

    # Hybrid BO: the acquisition problem with u as a parameter, the states as variables
    assert run.scenarios is not None  # hybrid BO runs hold their scenarios
    as_variables = settings.unknown_states_as_variables
    nlp, _ = hybrid_acquisition(
        problem,
        gp_spec,
        surrogate,
        run.scenarios,
        incumbent,
        settings.acquisition,
        as_variables,
    )
    assert nlp.g is not None  # hybrid BO's problem has the known equations
    n_u = problem.system.n_u
    fixed_u, states = Symbolic.sym("u", n_u), Symbolic.sym("w", len(nlp.lower) - n_u)
    w = cas.vertcat(fixed_u, states)
    solver = cas.nlpsol(
        "replay",
        "ipopt",
        {"x": states, "p": fixed_u, "f": nlp.f(w), "g": nlp.g(w)},
        settings.solver.plugin_options | {"ipopt": settings.solver.solver_options},
    )

    # The stored solution's states, the first start
    solution = run.acquisitions[i]
    assert solution.x is not None and solution.y is not None
    blocks = [solution.x, solution.y] if as_variables else [solution.x]
    stored = np.concatenate([block.ravel(order="F") for block in blocks])

    # Solve at each u, outwards from the trial point
    values = np.full(len(u), np.nan)
    f = np.full((len(u), len(run.scenarios.w)), np.nan)
    trial = int(np.flatnonzero(u == history.u[n, 0])[0])
    for indices in (range(trial, len(u)), range(trial, -1, -1)):
        start = stored
        for k in indices:
            bounds = {"lbx": nlp.lower[n_u:], "ubx": nlp.upper[n_u:]}
            result = solver(x0=start, p=u[k], lbg=0, ubg=0, **bounds)
            if not solver.stats()["success"]:
                continue
            start = result["x"].full().ravel()
            values[k] = float(result["f"])

            # f under each scenario, at that scenario's states
            w_k = np.concatenate([[u[k]], start])
            x, y = _scenario_states(
                problem, gp_spec, surrogate, run.scenarios, w_k, as_variables
            )
            f[k] = [
                problem.f(*state, u[k]).full().item()
                for state in zip(x, y, strict=True)
            ]
    return values, f


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
    among them (by the measured f) as a star, and the next point as a red ring."""
    ax.scatter(u[:n_initial], values[:n_initial], c="k", label="initial points")
    ax.scatter(u[n_initial:n], values[n_initial:n], c="r", label="points so far")
    best = int(np.argmin(f[:n]))
    ax.scatter(u[best], values[best], marker="*", s=150, c="gold", label="incumbent")
    ax.scatter(u[n], values[n], facecolors="none", edgecolors="r", label="next point")


def _gp_panel(ax: "Axes", u: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    """A GP's mean and the band holding `_CONFIDENCE` of its probability over u."""
    ax.plot(u, mean, color="b", label="GP mean")
    half_width = scipy.stats.norm.ppf(0.5 + _CONFIDENCE / 2) * std
    lower, upper = mean - half_width, mean + half_width
    label = f"{_CONFIDENCE:.0%} band"
    ax.fill_between(u, lower, upper, alpha=0.2, color="b", label=label)


def _scenarios_panel(ax: "Axes", u: np.ndarray, f: np.ndarray, w: np.ndarray) -> None:
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
