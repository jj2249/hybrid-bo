"""Static plots: convergence and regret across a study's runs, and the figure of one
iteration of a run.

The plots are drawn without a window (the Agg backend, set in `conftest.py`), and the
tests read the drawn data back from the matplotlib axes by their labels.
"""

import pickle
from collections.abc import Callable

import casadi as cas
import illustrative_example as ex
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rosenbrock as rb
import scipy
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from hybrid_bo import (
    EI,
    LCB,
    SAA,
    History,
    HybridBOSettings,
    Linearization,
    Problem,
    Run,
    SoftSAA,
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
from hybrid_bo.gp import predict

# The illustrative example's hybrid BO settings: the GP's prediction substituted for
# the unknown states
HYBRID_SETTINGS = HybridBOSettings(
    gp=ex.gp, approximation=SAA(unknown_states_as_variables=False)
)


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
    return illustrative_run(hybrid_bo, HybridBOSettings(gp=ex.gp, acquisition=LCB(2.0)))


@pytest.fixture(scope="module")
def hybrid_lin_run() -> Run:
    """A pickled hybrid BO run with Lin-EI, the GP's prediction substituted for the
    unknown states."""
    approximation = Linearization(unknown_states_as_variables=False)
    return illustrative_run(
        hybrid_bo, HybridBOSettings(gp=ex.gp, approximation=approximation)
    )


@pytest.fixture(scope="module")
def hybrid_soft_run() -> Run:
    """A pickled hybrid BO run with SoftSAA-EI at a temperature of 0.5, the GP's
    prediction substituted for the unknown states."""
    approximation = SoftSAA(0.5, n_scenarios=10, unknown_states_as_variables=False)
    return illustrative_run(
        hybrid_bo, HybridBOSettings(gp=ex.gp, approximation=approximation)
    )


@pytest.fixture(scope="module")
def standard_run() -> Run:
    """A pickled standard BO run with the illustrative example's settings."""
    return illustrative_run(standard_bo, StandardBOSettings(gp=ex.gp))


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
    return Run(history, artifact={"method": method}, method=method, index=index)


def fake_study(runs: list[Run], artifact: dict[str, str] | None = None) -> Study:
    """A study of these runs, 2 initial points and 2 iterations each."""
    settings = StudySettings(n_runs=2, n_initial=2, n_iterations=2, seed=0)
    return Study(tuple(runs), settings, "fake", artifact or {})


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
    ("f_oracle", "artifact"),
    [(None, {"oracle_optimum": "0.5"}), ([3.0, 1.0], {})],
)
def test_regret_needs_an_oracle_and_an_optimum(f_oracle, artifact):
    """Without oracle values, or without an optimum, there is no regret to draw."""
    study = fake_study([fake_run("a", 0, [3.0, 1.0], f_oracle)], artifact)
    with pytest.raises(ValueError, match="no regret"):
        plots.regret(study)


# The figure of one iteration


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize(
    "run_name",
    ["hybrid_run", "hybrid_lcb_run", "hybrid_lin_run", "standard_run", "lhs_run"],
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


@pytest.mark.parametrize(
    "run_name",
    [
        "hybrid_run",
        "hybrid_lcb_run",
        "hybrid_lin_run",
        "hybrid_soft_run",
        "standard_run",
    ],
)
@pytest.mark.parametrize("i", [0, 1])
def test_replayed_acquisition_at_the_trial_point_is_the_stored_value(
    request, run_name, i
):
    """The acquisition curve, replayed from the stored surrogate (and for hybrid BO
    the scenarios and the stored solution), passes through the acquisition value
    the run stored at its trial point. SoftSAA stores -log EI and is drawn as -EI,
    -exp(-value), so it reads like the other EI approximations."""
    # Act: the figure with the acquisition panel alone
    run = request.getfixturevalue(run_name)
    figure = plots.iteration(run, ex.problem, i, gp=False, scenarios=False)

    # Check: the curve at the trial point
    (ax,) = figure.axes
    curve = drawn_line(ax, "LCB" if run_name == "hybrid_lcb_run" else "-EI")
    trial = run.history.u[run.history.n_initial + i, 0]
    (at_trial,) = np.flatnonzero(np.asarray(curve.get_xdata()) == trial)
    stored = run.acquisitions[i].value
    if run_name == "hybrid_soft_run":
        stored = -np.exp(-stored)
    replayed = np.asarray(curve.get_ydata())[at_trial]
    np.testing.assert_allclose(replayed, stored, rtol=1e-8)
    plt.close("all")


def scenario_states(run: Run, iteration: int, u: np.ndarray) -> np.ndarray:
    """x_s at each u and scenario, (n_u, S), computed independently: y_s from the GP,
    then x_s from x + exp(x) = y_s with scipy's brentq."""
    assert run.scenarios is not None
    mean, std = predict(run.surrogates[iteration], u[:, np.newaxis])
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
    mean, _ = predict(hybrid_run.surrogates[1], u[:, np.newaxis])
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
        hybrid_run.surrogates[1],
        hybrid_run.scenarios,
        incumbent,
        EI(),
        SAA(unknown_states_as_variables=False),
    ).nlp
    expected = [
        nlp.f(_pack(np.array([u_k]), x_row[:, np.newaxis])).full().item()
        for u_k, x_row in zip(u, x, strict=True)
    ]
    np.testing.assert_allclose(line(acquisition, "-EI"), expected, atol=1e-4)
    plt.close("all")


def test_lin_figure_draws_the_nominal_f_and_its_std(hybrid_lin_run):
    """With Lin-EI, the scenarios panel draws f at the nominal states, x solving g
    with y the GP's mean, and the band f +- s, with s recomputed from the stored
    surrogate: on the illustrative system, s = |f_x| L / (1 + exp(x))."""
    # Act: iteration 1, the scenarios panel alone
    figure = plots.iteration(hybrid_lin_run, ex.problem, 1, gp=False, acquisition=False)
    (ax,) = figure.axes

    # The nominal f: x solves x + exp(x) = mean (to IPOPT's tolerance)
    u = np.asarray(drawn_line(ax, "nominal f").get_xdata())
    x = scenario_states(hybrid_lin_run, 1, u)[:, 0]
    f = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    np.testing.assert_allclose(line(ax, "nominal f"), f, atol=1e-4)

    # The band: f +- s at each u
    _, std = predict(hybrid_lin_run.surrogates[1], u[:, np.newaxis])
    f_x = 12 * (6 * x - 2) * np.sin(12 * x - 4)
    f_x += 12 * (6 * x - 2) ** 2 * np.cos(12 * x - 4)
    s = np.abs(f_x) * std / (1 + np.exp(x))
    (shaded,) = [c for c in ax.collections if c.get_label() == "nominal f ± s"]
    vertices = np.asarray(shaded.get_paths()[0].vertices)
    for u_k, lower, upper in zip(u[::10], (f - s)[::10], (f + s)[::10], strict=True):
        assert np.isclose(vertices, [u_k, lower], atol=1e-4).all(axis=1).any()
        assert np.isclose(vertices, [u_k, upper], atol=1e-4).all(axis=1).any()
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
    mean, _ = predict(surrogate, u[:, np.newaxis])
    np.testing.assert_allclose(line(gp_ax, "GP mean"), mean, rtol=1e-12)

    # The acquisition panel is the acquisition problem's objective
    incumbent = standard_run.history.f[:3].min()
    nlp = standard_acquisition(ex.problem, surrogate, incumbent, EI()).nlp
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


def test_figure_needs_one_or_two_decision_variables(lhs_run):
    """The panels are drawn over a grid of one or two decision variables."""
    u = [Var(f"u{j}", lower=0.0, upper=1.0) for j in range(3)]
    x = Var("x", start=(-1.0, 1.0))
    y = Var("y", start=(-1.0, 1.0))
    system = System(
        u=u, x=[x], y=[y], g=lambda v: [v.x - v.y], h_inputs=u, measured=[y]
    )
    problem = Problem(system, f=lambda v: v.x)

    with pytest.raises(ValueError, match="one or two decision variables"):
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


# The figure of one iteration with two decision variables, on the Rosenbrock problem


def rosenbrock_run(
    method: Callable[..., Run],
    settings: HybridBOSettings | StandardBOSettings | None = None,
) -> Run:
    """A pickled run of `method` on the Rosenbrock problem: 2 initial points, then 2
    iterations (or design points)."""
    rng = np.random.default_rng(0)
    u = initial_points(rb.system.u, 2, rng)
    if settings is None:
        return pickled(method(rb.problem, rb.measure, u, 2, rng))
    return pickled(method(rb.problem, rb.measure, u, 2, rng, settings=settings))


@pytest.fixture(scope="module")
def hybrid_2d_run() -> Run:
    """A pickled hybrid BO run on the Rosenbrock problem with SAA-EI over 10
    scenarios, the GP's prediction substituted for the unknown state."""
    approximation = SAA(n_scenarios=10, unknown_states_as_variables=False)
    return rosenbrock_run(
        hybrid_bo, HybridBOSettings(gp=rb.gp, approximation=approximation)
    )


@pytest.fixture(scope="module")
def hybrid_soft_2d_run() -> Run:
    """As `hybrid_2d_run`, with SoftSAA-EI at a temperature of 10."""
    approximation = SoftSAA(10.0, n_scenarios=10, unknown_states_as_variables=False)
    return rosenbrock_run(
        hybrid_bo, HybridBOSettings(gp=rb.gp, approximation=approximation)
    )


@pytest.fixture(scope="module")
def standard_2d_run() -> Run:
    """A pickled standard BO run on the Rosenbrock problem."""
    return rosenbrock_run(standard_bo, StandardBOSettings(gp=rb.gp))


@pytest.fixture(scope="module")
def lhs_2d_run() -> Run:
    """A pickled Latin hypercube sampling run on the Rosenbrock problem."""
    return rosenbrock_run(latin_hypercube_sampling)


def titles(figure: Figure) -> list[str]:
    """The titles of the figure's panels, row by row (colorbars have none)."""
    return [ax.get_title() for ax in figure.axes if ax.get_title()]


HYBRID_2D_TITLES = [
    "oracle y",
    "GP mean of y",
    "GP std of y",
    "oracle f",
    "mean f over scenarios",
    "std f over scenarios",
    "log10 EI",
]


@pytest.mark.parametrize(
    ("run_name", "expected"),
    [
        ("hybrid_2d_run", HYBRID_2D_TITLES),
        ("hybrid_soft_2d_run", HYBRID_2D_TITLES),
        ("standard_2d_run", ["oracle f", "GP mean of f", "GP std of f", "log10 EI"]),
        ("lhs_2d_run", ["oracle f"]),
    ],
)
def test_2d_figure_has_a_panel_per_quantity(request, run_name, expected, tmp_path):
    """With two decision variables each panel is a map over u: hybrid BO has a row
    for y (oracle, GP mean, GP std) and a row for f (oracle, the mean and std over
    the scenarios, EI); standard BO the row for f with its GP; a baseline the oracle
    f. Every panel shows the points, and the figure saves as a PNG."""
    # Act: iteration 1, with the oracle
    run = request.getfixturevalue(run_name)
    figure = plots.iteration(run, rb.problem, 1, measure=rb.measure)

    # Check: the titles, the points on every panel, one legend
    assert titles(figure) == expected
    assert figure.get_suptitle() == f"{run.method}, iteration 1"
    panels = [ax for ax in figure.axes if ax.get_title()]
    n_points = run.history.n_initial + 1
    for ax in panels:
        offsets = np.concatenate([c.get_offsets() for c in ax.collections[-4:]])
        assert len(offsets) == n_points + 2  # the incumbent again, the trial point
    assert {"initial points", "incumbent", "trial point"} <= set(legend_labels(figure))
    figure.savefig(tmp_path / "figure.png")
    plt.close("all")


def test_2d_figure_without_measure_drops_the_oracle_column(hybrid_2d_run):
    """Without a measurement there are no oracle values, so the oracle panels go."""
    figure = plots.iteration(hybrid_2d_run, rb.problem, 0)

    assert titles(figure) == [
        title for title in HYBRID_2D_TITLES if not title.startswith("oracle")
    ]
    plt.close("all")


def test_2d_flags_turn_panels_off(hybrid_2d_run):
    """The flags drop the same parts as in 1D: `gp` the row of y, `scenarios` the
    panels over the scenarios, `acquisition` the EI panel."""
    no_gp = plots.iteration(hybrid_2d_run, rb.problem, 0, gp=False)
    no_scenarios = plots.iteration(hybrid_2d_run, rb.problem, 0, scenarios=False)
    nothing = plots.iteration(
        hybrid_2d_run, rb.problem, 0, gp=False, scenarios=False, acquisition=False
    )

    assert titles(no_gp) == [
        "mean f over scenarios",
        "std f over scenarios",
        "log10 EI",
    ]
    assert titles(no_scenarios) == ["GP mean of y", "GP std of y", "log10 EI"]
    assert titles(nothing) == ["points"]
    plt.close("all")


def test_2d_ei_mask_covers_the_points_where_saa_ei_is_zero(hybrid_2d_run):
    """The EI panel shades in gray the grid points (30 x 30 over the start bounds)
    where SAA-EI is exactly 0: no scenario improves on the incumbent. On the
    Rosenbrock problem x = u1 - 1, so SAA-EI has a closed form: f_s = 100 y_s^2 + x^2
    with y_s the GP's scenario prediction."""
    # Act: iteration 1's EI panel alone
    run, i = hybrid_2d_run, 1
    figure = plots.iteration(run, rb.problem, i, gp=False, scenarios=False)
    (mesh,) = [c for c in figure.axes[0].collections if isinstance(c, QuadMesh)]
    gray = mesh.get_array()
    assert gray is not None
    zero = ~np.ma.getmaskarray(np.ma.asarray(gray)).ravel()

    # SAA-EI's improvement at each grid point, from the GP's scenario prediction
    assert run.scenarios is not None
    u1, u2 = (np.linspace(-2.048, 2.048, 30) for _ in range(2))
    grid = np.column_stack([coordinate.ravel() for coordinate in np.meshgrid(u1, u2)])
    mean, std = predict(run.surrogates[i], grid)
    y = mean[:, np.newaxis] + std[:, np.newaxis] * run.scenarios.z[:, 0]
    f = 100 * y**2 + (grid[:, [0]] - 1) ** 2
    incumbent = run.history.f[: run.history.n_initial + i].min()
    best_improvement = (incumbent - f).max(axis=1)

    # Check: gray exactly where no scenario improves, away from the boundary of that
    # region, where the replay's solver tolerance decides
    clear = np.abs(best_improvement) > 1e-6
    np.testing.assert_array_equal(zero[clear], (best_improvement <= 0)[clear])
    assert zero.any() and not zero.all()
    plt.close("all")
