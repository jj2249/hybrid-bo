"""Static plots: convergence and regret across a study's runs, and the figure of one
iteration of a run.

The plots are drawn without a window (the Agg backend, set in `conftest.py`), and the
tests read the drawn data back from the matplotlib axes by their labels.
"""

import pickle
from collections.abc import Callable
from dataclasses import replace

import casadi as cas
import illustrative_example as ex
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from hybrid_bo import (
    EI,
    LCB,
    History,
    HybridBOSettings,
    Problem,
    Run,
    StandardBOSettings,
    Study,
    StudySettings,
    System,
    Var,
    hybrid_bo,
    initial_points,
    latin_hypercube_sampling,
    plots,
    standard_bo,
)
from hybrid_bo.acquisition import _pack, hybrid_acquisition, standard_acquisition
from hybrid_bo.gp import default_gp, predict

# The illustrative example's hybrid BO settings: the GP's prediction substituted for
# the unknown states
HYBRID_SETTINGS = HybridBOSettings(unknown_states_as_variables=False)


def pickled(run: Run) -> Run:
    """The run after a round trip through pickle, as a saved run is loaded."""
    return pickle.loads(pickle.dumps(run))


def illustrative_run(
    method: Callable[..., Run],
    settings: HybridBOSettings | StandardBOSettings | None = None,
) -> Run:
    """A pickled run of `method` on the illustrative example: 2 initial points, then
    2 iterations (or design points)."""
    rng = np.random.default_rng(0)
    u = initial_points(ex.system.u, 2, rng)
    if settings is None:
        return pickled(method(ex.problem, ex.measure, u, 2, rng))
    return pickled(method(ex.problem, ex.measure, u, 2, rng, settings=settings))


# A fixture is a value pytest builds once and passes to every test that names it as an
# argument; `scope="module"` shares it between the tests of this file, since runs are
# slow to produce
@pytest.fixture(scope="module")
def hybrid_run() -> Run:
    """A pickled hybrid BO run with the illustrative example's settings (EI, the GP's
    prediction substituted for the unknown states)."""
    return illustrative_run(hybrid_bo, HYBRID_SETTINGS)


@pytest.fixture(scope="module")
def hybrid_lcb_run() -> Run:
    """A pickled hybrid BO run with LCB and the unknown states as variables."""
    return illustrative_run(hybrid_bo, HybridBOSettings(acquisition=LCB(2.0)))


@pytest.fixture(scope="module")
def standard_run() -> Run:
    """A pickled standard BO run with the illustrative example's settings."""
    return illustrative_run(standard_bo, StandardBOSettings())


@pytest.fixture(scope="module")
def lhs_run() -> Run:
    """A pickled Latin hypercube sampling run."""
    return illustrative_run(latin_hypercube_sampling)


def drawn_line(ax: Axes, label: str) -> Line2D:
    """The line labeled `label` on the axes."""
    (match,) = [line for line in ax.get_lines() if line.get_label() == label]
    return match


def line(ax: Axes, label: str) -> np.ndarray:
    """The y data of the line labeled `label` on the axes."""
    return np.asarray(drawn_line(ax, label).get_ydata())


def has_grid(ax: Axes) -> bool:
    """Whether the axes draw grid lines (the repo style turns them on)."""
    return all(line.get_visible() for line in ax.xaxis.get_gridlines())


def legend_labels(figure: Figure) -> list[str]:
    """The entries of the figure's one legend."""
    (legend,) = figure.legends
    return [text.get_text() for text in legend.get_texts()]


# Convergence and regret


def fake_run(
    method: str, index: int, f: list[float], f_oracle: list[float] | None
) -> Run:
    """A run of one decision variable, after 2 initial points, with these measured
    and oracle objective values."""
    n = len(f)
    history = History(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 1)),
        y=np.zeros((n, 1)),
        f=np.array(f),
        x_oracle=None if f_oracle is None else np.zeros((n, 1)),
        y_oracle=None if f_oracle is None else np.zeros((n, 1)),
        f_oracle=None if f_oracle is None else np.array(f_oracle),
        n_initial=2,
    )
    return Run(history, record={"method": method}, method=method, index=index)


def fake_study(runs: list[Run], record: dict[str, str] | None = None) -> Study:
    """A study of these runs, 2 initial points and 2 iterations each."""
    settings = StudySettings(n_runs=2, n_initial=2, n_iterations=2, seed=0)
    return Study(tuple(runs), settings, "fake", record or {})


def band(ax: Axes) -> np.ndarray:
    """The vertices (x, y) of the one shaded band on the axes."""
    (collection,) = ax.collections
    return np.asarray(collection.get_paths()[0].vertices)


def test_convergence_draws_the_median_oracle_incumbent_with_its_iqr():
    """Per method, the median over the runs of the oracle incumbent at each
    evaluation is drawn with the method's label, the interquartile range is shaded,
    and a dotted line marks the end of the initial points."""
    # Setup: the oracle incumbents are [3, 1, 1, 1] and [2, 2, 2, 0.5] (the oracle f
    # at each incumbent's point)
    runs = [
        fake_run("a", 0, [3.0, 1.0, 2.0, 1.5], [3.0, 1.0, 9.0, 9.0]),
        fake_run("a", 1, [2.0, 4.0, 3.0, 1.0], [2.0, 9.0, 9.0, 0.5]),
    ]

    ax = plots.convergence(fake_study(runs))

    # Check: the median, the band's quartiles (linear), the line, the legend
    np.testing.assert_allclose(line(ax, "a"), [2.5, 1.5, 1.5, 0.75])
    vertices = band(ax)
    for x, lower, upper in [(1, 2.25, 2.75), (2, 1.25, 1.75), (4, 0.625, 0.875)]:
        assert np.isclose(vertices, [x, lower]).all(axis=1).any()
        assert np.isclose(vertices, [x, upper]).all(axis=1).any()
    (dotted,) = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    np.testing.assert_array_equal(dotted.get_xdata(), [2.5, 2.5])
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["a"]
    assert ax.get_ylabel() == "oracle incumbent"
    assert has_grid(ax)
    plt.close("all")


def test_convergence_without_an_oracle_draws_the_incumbent():
    """Without oracle values the median incumbent is drawn instead, and a shorter
    (failed) run counts only up to its last evaluation."""
    runs = [
        fake_run("a", 0, [3.0, 1.0, 2.0], None),
        fake_run("a", 1, [2.0, 4.0], None),
    ]

    ax = plots.convergence(fake_study(runs))

    np.testing.assert_allclose(line(ax, "a"), [2.5, 1.5, 1.0])
    assert ax.get_ylabel() == "incumbent"
    plt.close("all")


def test_convergence_needs_a_run():
    """Without runs there is nothing to draw."""
    # `pytest.raises` passes only if the block raises this exception, with a message
    # matching the regular expression `match`
    with pytest.raises(ValueError, match="no incumbent"):
        plots.convergence(fake_study([]))


def test_regret_is_the_median_regret_on_a_log_axis():
    """Regret is the oracle incumbent minus the recorded optimum, or the one given,
    drawn on a log y-axis."""
    runs = [fake_run("a", 0, [3.0, 1.0], [3.0, 1.0])]
    study = fake_study(runs, {"oracle_optimum": "0.5"})

    ax = plots.regret(study)
    given = plots.regret(study, optimum=0.0)

    np.testing.assert_allclose(line(ax, "a"), [2.5, 0.5])
    np.testing.assert_allclose(line(given, "a"), [3.0, 1.0])
    assert ax.get_yscale() == "log"
    plt.close("all")


@pytest.mark.parametrize(
    ("f_oracle", "record"),
    [(None, {"oracle_optimum": "0.5"}), ([3.0, 1.0], {})],
)
def test_regret_needs_an_oracle_and_an_optimum(f_oracle, record):
    """Without oracle values, or without an optimum, there is no regret to draw."""
    study = fake_study([fake_run("a", 0, [3.0, 1.0], f_oracle)], record)
    with pytest.raises(ValueError, match="no regret"):
        plots.regret(study)


# The figure of one iteration


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize(
    "run_name", ["hybrid_run", "hybrid_lcb_run", "standard_run", "lhs_run"]
)
def test_every_iteration_of_every_method_is_drawn(request, run_name, tmp_path):
    """Each iteration of a run of each method draws a figure that saves as a PNG.

    `tmp_path` is a pytest fixture: a fresh temporary folder for this test.
    """
    # `request.getfixturevalue` builds the fixture named by the parameter
    run = request.getfixturevalue(run_name)
    for i in range(2):
        figure = plots.iteration(run, ex.problem, i)
        figure.savefig(tmp_path / f"{run_name}-{i}.png")
        assert (tmp_path / f"{run_name}-{i}.png").stat().st_size > 0
    plt.close("all")


@pytest.mark.parametrize("run_name", ["hybrid_run", "hybrid_lcb_run", "standard_run"])
@pytest.mark.parametrize("i", [0, 1])
def test_replayed_acquisition_at_the_trial_point_is_the_stored_value(
    request, run_name, i
):
    """The acquisition curve, replayed from the stored surrogate (and for hybrid BO
    the scenarios and the stored solution), passes through the acquisition value
    the run stored at its trial point."""
    # Act: the figure with the acquisition panel alone
    run = request.getfixturevalue(run_name)
    figure = plots.iteration(run, ex.problem, i, gp=False, scenarios=False)

    # Check: the curve at the trial point
    (ax,) = figure.axes
    name = "LCB" if run_name == "hybrid_lcb_run" else "-EI"
    curve = drawn_line(ax, name)
    trial = run.history.u[run.history.n_initial + i, 0]
    (at_trial,) = np.flatnonzero(np.asarray(curve.get_xdata()) == trial)
    stored = run.acquisitions[i].value
    replayed = np.asarray(curve.get_ydata())[at_trial]
    np.testing.assert_allclose(replayed, stored, rtol=1e-8)
    plt.close("all")


def scenario_states(run: Run, iteration: int, u: np.ndarray) -> np.ndarray:
    """x_s at each u and scenario, (n_u, S), computed independently: y_s from the GP,
    then x_s from x + exp(x) = y_s with scipy's brentq."""
    assert run.scenarios is not None
    mean, std = predict(default_gp(1), run.surrogates[iteration], u[:, np.newaxis])
    y = mean[:, np.newaxis] + std[:, np.newaxis] * run.scenarios.z[:, 0]
    return np.vectorize(
        lambda y_s: scipy.optimize.brentq(lambda x: x + np.exp(x) - y_s, -50, 50)
    )(y)


def test_hybrid_bo_figure_from_a_pickled_run(hybrid_run):
    """Hybrid BO's figure is titled with the method and iteration and has three
    gridded panels: the GP of y with its true curve, f under the scenarios, and the
    acquisition function, the objective of the acquisition problem at each u with
    each scenario's states solving g, all with one legend for the figure."""
    # Act: iteration 1, with the oracle for the true curves
    figure = plots.iteration(hybrid_run, ex.problem, 1, measure=ex.measure)

    # The title, three gridded panels, the figure's legend
    surrogate, scenarios, acquisition = figure.axes
    assert figure.get_suptitle() == "hybrid_bo, iteration 1"
    assert all(has_grid(ax) for ax in figure.axes)
    assert [ax.get_legend() for ax in figure.axes] == [None] * 3
    labels = legend_labels(figure)
    assert len(set(labels)) == len(labels)
    assert {"initial points", "incumbent", "trial point", "true"} <= set(labels)

    # The GP of y: its prediction, and the oracle y = sin(u) as the true curve
    u = np.asarray(drawn_line(surrogate, "GP mean").get_xdata())
    mean, _ = predict(default_gp(1), hybrid_run.surrogates[1], u[:, np.newaxis])
    np.testing.assert_allclose(line(surrogate, "GP mean"), mean, rtol=1e-12)
    np.testing.assert_allclose(line(surrogate, "true"), np.sin(u), atol=1e-8)

    # f under the scenarios: their mean, with x_s solving g (to IPOPT's tolerance)
    x = scenario_states(hybrid_run, 1, u)
    f = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    np.testing.assert_allclose(
        line(scenarios, "scenario mean"), f.mean(axis=1), atol=1e-4
    )

    # The acquisition function: the acquisition problem's objective at (u, x_s)
    assert hybrid_run.scenarios is not None
    incumbent = hybrid_run.history.f[:3].min()
    nlp = hybrid_acquisition(
        ex.problem,
        default_gp(1),
        hybrid_run.surrogates[1],
        hybrid_run.scenarios,
        incumbent,
        EI(),
        unknown_states_as_variables=False,
    ).nlp
    expected = [
        nlp.f(_pack(np.array([u_k]), x_row[:, np.newaxis])).full().item()
        for u_k, x_row in zip(u, x, strict=True)
    ]
    np.testing.assert_allclose(line(acquisition, "-EI"), expected, atol=1e-4)
    plt.close("all")


def test_standard_bo_figure_from_a_pickled_run(standard_run):
    """Standard BO's figure has two panels: the GP of f and the acquisition function,
    the objective of its acquisition problem at each u."""
    # Act: iteration 1, after 2 initial points and 1 trial point
    figure = plots.iteration(standard_run, ex.problem, 1)

    # The GP panel is the GP's prediction
    gp_ax, acquisition_ax = figure.axes
    assert figure.get_suptitle() == "standard_bo, iteration 1"
    u = np.asarray(drawn_line(gp_ax, "GP mean").get_xdata())
    surrogate = standard_run.surrogates[1]
    mean, _ = predict(default_gp(1), surrogate, u[:, np.newaxis])
    np.testing.assert_allclose(line(gp_ax, "GP mean"), mean, rtol=1e-12)

    # The acquisition panel is the acquisition problem's objective
    incumbent = standard_run.history.f[:3].min()
    nlp = standard_acquisition(
        ex.problem, default_gp(1), surrogate, incumbent, EI()
    ).nlp
    expected = [nlp.f(_pack(np.array([u_k]))).full().item() for u_k in u]
    np.testing.assert_allclose(line(acquisition_ax, "-EI"), expected, rtol=1e-12)
    plt.close("all")


def test_baseline_figure_shows_only_the_points_so_far(lhs_run):
    """A baseline fits no GP, so its figure shows the points before the iteration's
    design point, their incumbent and the next one, with no curves."""
    # Act: iteration 1, without the true curve
    figure = plots.iteration(lhs_run, ex.problem, 1)

    # Check: one panel, no curves, and the points so far
    (ax,) = figure.axes
    assert figure.get_suptitle() == "latin_hypercube_sampling, iteration 1"
    assert ax.get_lines() == []
    history, n = lhs_run.history, lhs_run.history.n_initial + 1
    points = np.column_stack([history.u[:, 0], history.f])
    initial, so_far, incumbent, trial_point = ax.collections
    np.testing.assert_array_equal(initial.get_offsets(), points[:2])
    np.testing.assert_array_equal(so_far.get_offsets(), points[2:n])
    best = np.argmin(history.f[:n])
    np.testing.assert_array_equal(incumbent.get_offsets(), points[[best]])
    np.testing.assert_array_equal(trial_point.get_offsets(), points[[n]])
    plt.close("all")


def test_flags_turn_panels_off(hybrid_run, standard_run):
    """Each panel has a flag; with every panel off, the points with f remain."""
    only_scenarios = plots.iteration(
        hybrid_run, ex.problem, 0, gp=False, acquisition=False
    )
    nothing = plots.iteration(standard_run, ex.problem, 0, gp=False, acquisition=False)

    assert [ax.get_ylabel() for ax in only_scenarios.axes] == ["f"]
    assert drawn_line(only_scenarios.axes[0], "scenario mean") is not None
    (ax,) = nothing.axes
    assert ax.get_lines() == []
    plt.close("all")


def test_iteration_counts_from_the_end_when_negative(standard_run):
    """Iteration -1 is the last one."""
    figure = plots.iteration(standard_run, ex.problem, -1)
    assert figure.get_suptitle() == "standard_bo, iteration 1"
    plt.close("all")


@pytest.mark.parametrize("i", [2, -3])
def test_iteration_must_exist(standard_run, i):
    """Two iterations ran, so only iterations 0 and 1 (or -2 and -1) can be drawn."""
    with pytest.raises(IndexError, match="iteration"):
        plots.iteration(standard_run, ex.problem, i)


def test_settings_matching_the_record_are_used(hybrid_run):
    """A run's settings, with the GP given, can be passed when they match the
    record."""
    settings = replace(HYBRID_SETTINGS, gp=default_gp(1))
    figure = plots.iteration(hybrid_run, ex.problem, 0, settings=settings)
    assert len(figure.axes) == 3
    plt.close("all")


def test_a_gp_other_than_the_default_needs_the_settings(hybrid_run):
    """The record holds the GP as text, so a run whose GP isn't `default_gp` can't
    be replayed without its settings."""
    run = replace(hybrid_run, record=hybrid_run.record | {"kernel": "another"})
    with pytest.raises(ValueError, match="settings"):
        plots.iteration(run, ex.problem, 0)


def test_a_custom_solver_needs_the_settings(hybrid_run):
    """The replay solves with the run's solver options, so a run whose record shows
    other options than the defaults can't be replayed without its settings."""
    settings = hybrid_run.record["settings"].replace(
        "'print_level': 0", "'print_level': 5"
    )
    run = replace(hybrid_run, record=hybrid_run.record | {"settings": settings})
    with pytest.raises(ValueError, match="settings"):
        plots.iteration(run, ex.problem, 0)


@pytest.mark.parametrize(
    ("run_name", "settings", "error"),
    [
        ("hybrid_run", StandardBOSettings(), TypeError),
        ("standard_run", HybridBOSettings(), TypeError),
        ("lhs_run", StandardBOSettings(), TypeError),
        ("hybrid_run", HybridBOSettings(), ValueError),
    ],
)
def test_settings_must_be_the_runs(request, run_name, settings, error):
    """Settings must be the run's method's (else `TypeError`) and match its record
    (else `ValueError`)."""
    run = request.getfixturevalue(run_name)
    with pytest.raises(error, match="settings"):
        plots.iteration(run, ex.problem, 0, settings=settings)


def test_figure_needs_one_decision_variable(lhs_run):
    """The panels are drawn over a grid of one decision variable."""
    u1, u2 = Var("u1", lower=0.0, upper=1.0), Var("u2", lower=0.0, upper=1.0)
    x = Var("x", start=(-1.0, 1.0))
    y = Var("y", start=(-1.0, 1.0))
    system = System(
        u=[u1, u2],
        x=[x],
        y=[y],
        g=lambda v: [v.x - v.y],
        h_inputs=[u1, u2],
        measured=[y],
    )
    problem = Problem(system, f=lambda v: v.x)

    with pytest.raises(ValueError, match="one decision variable"):
        plots.iteration(lhs_run, problem, 0)


def test_hybrid_bo_gp_panel_needs_h_to_take_only_u(hybrid_run):
    """The GP of y is drawn over u, so h must take u alone."""
    u = Var("u", lower=-2.0, upper=2.0)
    x = Var("x", start=(-2.5, 1.0))
    y = Var("y", start=(-2.0, 2.0))
    system = System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        h_inputs=[u, x],
        measured=[y],
    )
    problem = Problem(system, f=lambda v: v.x)

    with pytest.raises(ValueError, match="h_inputs"):
        plots.iteration(hybrid_run, problem, 0)
