"""Plots of finished runs: convergence across runs and the figure of one iteration.

The plots are drawn without a window (the Agg backend, set in `conftest.py`), and the
tests read the drawn data back from the matplotlib axes by their labels.
"""

import pickle

import casadi as cas
import illustrative_example as ex
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy

from hybrid_bo import (
    EI,
    LCB,
    HybridBOSettings,
    Problem,
    Run,
    StandardBOSettings,
    System,
    Var,
    default_gp,
    hybrid_acquisition,
    hybrid_bo,
    initial_points,
    latin_hypercube_sampling,
    plots,
    predict,
    standard_acquisition,
    standard_bo,
)
from hybrid_bo.history import History

GRID = np.linspace(-2.0, 2.0, 101)  # the illustrative example's u start bounds


def pickled(run: Run) -> Run:
    """The run after a round trip through pickle, as a saved run is loaded."""
    return pickle.loads(pickle.dumps(run))


def illustrative_run(method, settings=None) -> Run:
    """A pickled run of `method` on the illustrative example: 2 initial points, then
    2 iterations (or design points)."""
    rng = np.random.default_rng(0)
    u = initial_points(ex.system.u, 2, rng)
    if settings is None:
        return pickled(method(ex.problem, ex.experiment, u, 2, rng))
    return pickled(method(ex.problem, ex.experiment, u, 2, settings, rng))


# A fixture is a value pytest builds once and passes to every test that names it as an
# argument; `scope="module"` shares it between the tests of this file, since runs are
# slow to produce
@pytest.fixture(scope="module")
def hybrid_run() -> Run:
    """A pickled hybrid BO run with the illustrative example's settings."""
    return illustrative_run(hybrid_bo, ex.settings["hybrid_bo"])


@pytest.fixture(scope="module")
def standard_run() -> Run:
    """A pickled standard BO run with the illustrative example's settings."""
    return illustrative_run(standard_bo, ex.settings["standard_bo"])


@pytest.fixture(scope="module")
def lhs_run() -> Run:
    """A pickled Latin hypercube sampling run."""
    return illustrative_run(latin_hypercube_sampling)


def line(ax, label: str) -> np.ndarray:
    """The y data of the line labeled `label` on the axes."""
    (match,) = [line for line in ax.get_lines() if line.get_label() == label]
    return np.asarray(match.get_ydata())


def has_grid(ax) -> bool:
    """Whether the axes draw grid lines (the repo style turns them on)."""
    return all(line.get_visible() for line in ax.xaxis.get_gridlines())


def fake_run(f: list[float], n_initial: int = 2) -> Run:
    """A run of one decision variable with these objective values and no oracle."""
    n = len(f)
    history = History(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 1)),
        y=np.zeros((n, 1)),
        f=np.array(f),
        x_oracle=None,
        y_oracle=None,
        f_oracle=None,
        n_initial=n_initial,
    )
    return Run(history, record={"method": "test"})


def scenario_states(run: Run, iteration: int) -> np.ndarray:
    """x_s at each grid point and scenario, (n_grid, S), computed independently: y_s
    from the GP, then x_s from x + exp(x) = y_s with scipy's brentq."""
    assert run.scenarios is not None
    mean, std = predict(default_gp(1), run.surrogates[iteration], GRID[:, np.newaxis])
    y = mean[:, np.newaxis] + std[:, np.newaxis] * run.scenarios.z[:, 0]
    return np.vectorize(
        lambda y_s: scipy.optimize.brentq(lambda x: x + np.exp(x) - y_s, -50, 50)
    )(y)


# Convergence


def test_convergence_draws_each_runs_incumbents_and_their_median():
    """Each run is drawn as its running minimum of f over the evaluations, and the
    median over the runs is drawn with the label."""
    runs = [fake_run([3.0, 1.0, 2.0, 0.5]), fake_run([2.0, 4.0, 1.5, 1.0])]

    ax = plots.convergence(runs, label="hybrid BO")

    drawn = [np.asarray(line.get_ydata()) for line in ax.get_lines()]
    np.testing.assert_array_equal(drawn[0], [3.0, 1.0, 1.0, 0.5])
    np.testing.assert_array_equal(drawn[1], [2.0, 2.0, 1.5, 1.0])
    np.testing.assert_array_equal(line(ax, "hybrid BO"), [2.5, 1.5, 1.25, 0.75])
    assert has_grid(ax)
    plt.close("all")


def test_convergence_takes_the_median_over_the_runs_that_reach_an_evaluation():
    """A shorter (failed) run counts in the median only up to its last evaluation."""
    runs = [fake_run([3.0, 1.0, 2.0]), fake_run([2.0, 4.0])]

    ax = plots.convergence(runs, label="median")

    np.testing.assert_array_equal(line(ax, "median"), [2.5, 1.5, 1.0])
    plt.close("all")


def test_convergence_needs_a_run():
    """Without runs there is nothing to draw."""
    # `pytest.raises` passes only if the block raises this exception, with a message
    # matching the regular expression `match`
    with pytest.raises(ValueError, match="at least one run"):
        plots.convergence([])


# The figure of a run's iterations


def test_hybrid_bo_figure_from_a_pickled_run(hybrid_run):
    """Hybrid BO's figure is titled with the method and has three gridded panels: the
    GP of y with its true curve, f under the scenarios, and the acquisition
    function, all over the grid of u, with one legend for the figure."""
    # Act: the figure, showing iteration 1, with the oracle for the true curves
    settings = ex.settings["hybrid_bo"]
    rng = np.random.default_rng(0)
    figure = plots.iterations(hybrid_run, ex.problem, rng, settings, ex.experiment, i=1)

    # The title, three gridded panels (then the buttons' axes), the figure's legend
    surrogate, scenarios, acquisition, *_buttons = figure.axes
    assert figure.get_suptitle() == "hybrid_bo"
    assert all(has_grid(ax) for ax in (surrogate, scenarios, acquisition))
    assert [ax.get_legend() for ax in (surrogate, scenarios, acquisition)] == [None] * 3
    (legend,) = figure.legends
    assert len({text.get_text() for text in legend.get_texts()}) == len(
        legend.get_texts()
    )

    # The GP of y: its prediction, and the oracle y = sin(u) as the true curve
    mean, _ = predict(default_gp(1), hybrid_run.surrogates[1], GRID[:, np.newaxis])
    np.testing.assert_allclose(line(surrogate, "GP mean"), mean, rtol=1e-12)
    np.testing.assert_allclose(line(surrogate, "true"), np.sin(GRID), atol=1e-8)

    # f under the scenarios: their mean, and the true objective
    x = scenario_states(hybrid_run, 1)
    f = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    np.testing.assert_allclose(line(scenarios, "scenario mean"), f.mean(axis=1))
    assert len(line(scenarios, "true")) == len(GRID)
    assert acquisition.get_ylabel() == "-EI"
    plt.close("all")


def click(button) -> None:
    """Clicks a matplotlib button, as a mouse click would."""
    button._observers.process("clicked", None)


def test_buttons_step_through_the_iterations(hybrid_run):
    """The - and + buttons redraw the previous and next iteration, still gridded, and
    stop at the first and last."""
    # Setup: the figure shows the last of the two iterations
    settings = ex.settings["hybrid_bo"]
    figure = plots.iterations(
        hybrid_run, ex.problem, np.random.default_rng(0), settings
    )
    minus, plus = figure._hybrid_bo_buttons  # ty: ignore[unresolved-attribute]

    def gp_mean_is(k: int) -> None:
        """Checks the GP panel shows iteration k's GP."""
        mean, _ = predict(default_gp(1), hybrid_run.surrogates[k], GRID[:, np.newaxis])
        np.testing.assert_allclose(line(figure.axes[0], "GP mean"), mean, rtol=1e-12)

    # + at the last iteration stays there; - steps back, and again stays at the first
    click(plus)
    gp_mean_is(1)
    click(minus)
    gp_mean_is(0)
    click(minus)
    gp_mean_is(0)
    assert has_grid(figure.axes[0])

    # + steps forward again
    click(plus)
    gp_mean_is(1)
    plt.close("all")


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize("acquisition", [EI(), LCB(2.0)])
def test_hybrid_bo_acquisition_is_the_objective_of_the_acquisition_problem(
    hybrid_run, acquisition
):
    """The drawn acquisition function is the objective of hybrid BO's acquisition
    problem at each grid point, with each scenario's states solving g."""
    # Act: the figure of iteration 0, with this acquisition function
    settings = HybridBOSettings(
        acquisition=acquisition, unknown_states_as_variables=False
    )
    iteration = 0
    figure = plots.iterations(
        hybrid_run, ex.problem, np.random.default_rng(0), settings, i=iteration
    )

    # Reference: the acquisition problem at (u, x_s), with x_s from brentq, and the
    # incumbent of the initial points
    incumbent = hybrid_run.history.f[: hybrid_run.history.n_initial].min()
    assert hybrid_run.scenarios is not None
    nlp, _ = hybrid_acquisition(
        ex.problem,
        default_gp(1),
        hybrid_run.surrogates[iteration],
        hybrid_run.scenarios,
        incumbent,
        acquisition,
        unknown_states_as_variables=False,
    )
    x_s = scenario_states(hybrid_run, iteration)
    expected = [
        nlp.f(np.concatenate([[u], x_row])).full().item()
        for u, x_row in zip(GRID, x_s, strict=True)
    ]

    # Check: the third panel's curve, named by the acquisition function
    name = "-EI" if isinstance(acquisition, EI) else "LCB"
    drawn = line(figure.axes[2], name)
    np.testing.assert_allclose(drawn, expected, rtol=1e-6, atol=1e-9)
    plt.close("all")


def test_standard_bo_figure_from_a_pickled_run(standard_run):
    """Standard BO's figure has two panels: the GP of f and the acquisition function,
    the objective of its acquisition problem at each grid point."""
    # Act: the figure of iteration 1, after 2 initial points and 1 trial point
    settings = ex.settings["standard_bo"]
    figure = plots.iterations(
        standard_run, ex.problem, np.random.default_rng(0), settings, i=1
    )

    # The GP panel is the GP's prediction
    gp_ax, acquisition_ax, *_buttons = figure.axes
    assert figure.get_suptitle() == "standard_bo"
    surrogate = standard_run.surrogates[1]
    mean, _ = predict(default_gp(1), surrogate, GRID[:, np.newaxis])
    np.testing.assert_allclose(line(gp_ax, "GP mean"), mean, rtol=1e-12)

    # The acquisition panel is the acquisition problem's objective
    incumbent = standard_run.history.f[:3].min()
    nlp, _ = standard_acquisition(
        ex.problem, default_gp(1), surrogate, incumbent, settings.acquisition
    )
    expected = [nlp.f(u).full().item() for u in GRID]
    np.testing.assert_allclose(line(acquisition_ax, "-EI"), expected, rtol=1e-12)
    plt.close("all")


def test_baseline_figure_shows_only_the_points_so_far(lhs_run):
    """A baseline fits no GP, so its figure shows the points before the iteration's
    design point and the next one, with no GP."""
    # Act: the figure of iteration 1, without the true curve
    figure = plots.iterations(lhs_run, ex.problem, np.random.default_rng(0), i=1)

    # Check: one panel titled with the method, no curves, and the points so far
    ax, *_buttons = figure.axes
    assert figure.get_suptitle() == "latin_hypercube_sampling"
    assert ax.get_lines() == []
    history, n = lhs_run.history, lhs_run.history.n_initial + 1
    initial, design, next_point = ax.collections
    np.testing.assert_array_equal(
        initial.get_offsets(), np.column_stack([history.u[:2, 0], history.f[:2]])
    )
    np.testing.assert_array_equal(
        design.get_offsets(), np.column_stack([history.u[2:n, 0], history.f[2:n]])
    )
    np.testing.assert_array_equal(
        next_point.get_offsets(), [[history.u[n, 0], history.f[n]]]
    )
    plt.close("all")


@pytest.mark.parametrize("iteration", [2, -3])
def test_iteration_must_exist(standard_run, iteration):
    """Two iterations ran, so only iterations 0 and 1 (or -2 and -1) can be shown."""
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError, match="iteration"):
        plots.iterations(
            standard_run, ex.problem, rng, ex.settings["standard_bo"], i=iteration
        )


@pytest.mark.parametrize(
    ("run_name", "settings"),
    [
        ("hybrid_run", StandardBOSettings()),
        ("standard_run", HybridBOSettings()),
        ("standard_run", None),
        ("lhs_run", StandardBOSettings()),
    ],
)
def test_settings_must_be_the_runs_method(request, run_name, settings):
    """Hybrid BO runs take hybrid BO settings, standard BO runs standard BO settings,
    and the baselines none."""
    # `request.getfixturevalue` builds the fixture named by the parameter
    run = request.getfixturevalue(run_name)
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="settings"):
        plots.iterations(run, ex.problem, rng, settings)


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
        plots.iterations(lhs_run, problem, np.random.default_rng(0))


def test_hybrid_bo_figure_needs_h_to_take_only_u(hybrid_run):
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
        plots.iterations(
            hybrid_run, problem, np.random.default_rng(0), HybridBOSettings()
        )


@pytest.mark.parametrize("acquisition", [EI(), LCB(2.0)])
def test_scenarios_without_a_root_are_left_out_with_the_weights_renormalized(
    acquisition,
):
    """At a grid point where a scenario has no root (NaN), the acquisition function
    is computed from the other scenarios, their weights renormalized; where no
    scenario has a root, it is NaN."""
    # Setup: the second scenario has no root at the first grid point, and no
    # scenario has one at the second
    f = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    w = np.array([0.5, 0.25, 0.25])

    values = plots._saa_acquisition(f, w, 2.0, acquisition)

    # Check: the first grid point is the other two scenarios, weighted 2/3 and 1/3
    expected = plots._saa_acquisition(
        f[:1, [0, 2]], np.array([2 / 3, 1 / 3]), 2.0, acquisition
    )
    np.testing.assert_allclose(values[0], expected[0])
    assert np.isnan(values[1])
