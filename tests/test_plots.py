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
    fit_gp,
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


# The figure of one iteration


def test_hybrid_bo_figure_from_a_pickled_run(hybrid_run):
    """Hybrid BO's figure has three panels: the GP of y with its true curve, f under
    the scenarios, and the acquisition function, all over the grid of u."""
    settings = ex.settings["hybrid_bo"]
    rng = np.random.default_rng(0)

    figure = plots.iteration(hybrid_run, 1, ex.problem, rng, settings, ex.experiment)

    # The GP of y: its prediction, and the oracle y = sin(u) as the true curve
    surrogate, scenarios, acquisition = figure.axes
    mean, _ = predict(default_gp(1), hybrid_run.surrogates[1], GRID[:, np.newaxis])
    np.testing.assert_allclose(line(surrogate, "Predicted mean"), mean, rtol=1e-12)
    np.testing.assert_allclose(line(surrogate, "True"), np.sin(GRID), atol=1e-8)

    # f under the scenarios: their mean, and the true objective
    x = scenario_states(hybrid_run, 1)
    f = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    np.testing.assert_allclose(line(scenarios, "Scenario mean"), f.mean(axis=1))
    assert len(line(scenarios, "True")) == len(GRID)
    assert acquisition.get_ylabel() == "Acq.-Fun."
    plt.close("all")


@pytest.mark.parametrize("acquisition", [EI(), LCB(2.0)])
def test_hybrid_bo_acquisition_is_the_objective_of_the_acquisition_problem(
    hybrid_run, acquisition
):
    """The drawn acquisition function is the objective of hybrid BO's acquisition
    problem at each grid point, with each scenario's states solving g."""
    settings = HybridBOSettings(
        acquisition=acquisition, unknown_states_as_variables=False
    )
    iteration = 0
    incumbent = hybrid_run.history.f[: hybrid_run.history.n_initial].min()

    figure = plots.iteration(
        hybrid_run, iteration, ex.problem, np.random.default_rng(0), settings
    )

    # Reference: the acquisition problem at (u, x_s), with x_s from brentq
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
    label = (
        "Expected improvement"
        if isinstance(acquisition, EI)
        else ("Lower confidence bound")
    )
    drawn = line(figure.axes[2], label)
    np.testing.assert_allclose(drawn, expected, rtol=1e-6, atol=1e-9)
    plt.close("all")


def test_standard_bo_figure_from_a_pickled_run(standard_run):
    """Standard BO's figure has two panels: the GP of f and the acquisition function,
    the objective of its acquisition problem at each grid point."""
    settings = ex.settings["standard_bo"]
    surrogate = standard_run.surrogates[1]
    incumbent = standard_run.history.f[:3].min()

    figure = plots.iteration(
        standard_run, 1, ex.problem, np.random.default_rng(0), settings
    )

    gp_ax, acquisition_ax = figure.axes
    mean, _ = predict(default_gp(1), surrogate, GRID[:, np.newaxis])
    np.testing.assert_allclose(line(gp_ax, "Predicted mean"), mean, rtol=1e-12)
    nlp, _ = standard_acquisition(
        ex.problem, default_gp(1), surrogate, incumbent, settings.acquisition
    )
    expected = [nlp.f(u).full().item() for u in GRID]
    np.testing.assert_allclose(
        line(acquisition_ax, "Expected improvement"), expected, rtol=1e-12
    )
    plt.close("all")


def test_baseline_figure_fits_a_gp_to_the_points_so_far(lhs_run):
    """A baseline run has no surrogates, so its figure fits the default GP of u -> f
    to the points before the iteration's design point, with the plot's generator."""
    iteration = 1
    n = lhs_run.history.n_initial + iteration

    figure = plots.iteration(lhs_run, iteration, ex.problem, np.random.default_rng(1))

    # Reference: the same fit with a generator in the same state
    history = lhs_run.history
    fitted = fit_gp(
        default_gp(1), history.u[:n], history.f[:n], np.random.default_rng(1)
    )
    mean, _ = predict(default_gp(1), fitted, GRID[:, np.newaxis])
    (ax,) = figure.axes
    np.testing.assert_allclose(line(ax, "Predicted mean"), mean, rtol=1e-12)
    plt.close("all")


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize("iteration", [2, -3])
def test_iteration_must_exist(standard_run, iteration):
    """Two iterations ran, so only iterations 0 and 1 (or -2 and -1) have a figure."""
    rng = np.random.default_rng(0)
    with pytest.raises(IndexError, match="iteration"):
        plots.iteration(
            standard_run, iteration, ex.problem, rng, ex.settings["standard_bo"]
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
        plots.iteration(run, 0, ex.problem, rng, settings)


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
        plots.iteration(lhs_run, 0, problem, np.random.default_rng(0))


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
        plots.iteration(
            hybrid_run, 0, problem, np.random.default_rng(0), HybridBOSettings()
        )
