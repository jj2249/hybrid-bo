"""Static plots: convergence and regret across a study's runs, and the figure of one
iteration of a run.

The plots are drawn without a window (the Agg backend, set in `conftest.py`), and the
tests read the drawn data back from the matplotlib axes by their labels.
"""

import functools
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
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.contour import ContourSet
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
    ReplayWarning,
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
    oracle_measure,
    plots,
    standard_bo,
)
from hybrid_bo.acquisition import _pack, hybrid_acquisition, standard_acquisition
from hybrid_bo.gp import predict

# The illustrative example's hybrid BO settings: the GP's prediction substituted for
# the unknown states
HYBRID_SETTINGS = HybridBOSettings(
    gp=ex.gp,
    acquisition=EI(),
    approximation=SAA(n_scenarios=25, unknown_states_as_variables=False),
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
    return illustrative_run(
        hybrid_bo,
        HybridBOSettings(
            gp=ex.gp, acquisition=LCB(std_factor=2.0), approximation=SAA(n_scenarios=25)
        ),
    )


@pytest.fixture(scope="module")
def hybrid_lin_run() -> Run:
    """A pickled hybrid BO run with Lin-EI, the GP's prediction substituted for the
    unknown states."""
    approximation = Linearization(unknown_states_as_variables=False)
    return illustrative_run(
        hybrid_bo,
        HybridBOSettings(gp=ex.gp, acquisition=EI(), approximation=approximation),
    )


@pytest.fixture(scope="module")
def hybrid_soft_run() -> Run:
    """A pickled hybrid BO run with SoftSAA-EI at a temperature of 0.5, the GP's
    prediction substituted for the unknown states."""
    approximation = SoftSAA(
        temperature=0.5, n_scenarios=10, unknown_states_as_variables=False
    )
    return illustrative_run(
        hybrid_bo,
        HybridBOSettings(gp=ex.gp, acquisition=EI(), approximation=approximation),
    )


@pytest.fixture(scope="module")
def standard_run() -> Run:
    """A pickled standard BO run with the illustrative example's settings."""
    return illustrative_run(standard_bo, StandardBOSettings(gp=ex.gp, acquisition=EI()))


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
    method: str,
    index: int,
    f: list[float],
    f_oracle: list[float] | None,
    n_initial: int = 2,
) -> Run:
    """A run of one decision variable, after `n_initial` initial points, with these
    measured and oracle objective values."""
    n = len(f)
    history = History(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 1)),
        y=np.zeros((n, 1)),
        f=np.array(f),
        x_oracle=None if f_oracle is None else np.zeros((n, 1)),
        y_oracle=None if f_oracle is None else np.zeros((n, 1)),
        f_oracle=None if f_oracle is None else np.array(f_oracle),
        n_initial=n_initial,
    )
    return Run(history, artifact={"method": method}, method=method, index=index)


def fake_study(runs: list[Run], artifact: dict[str, str] | None = None) -> Study:
    """A study of these runs, two of each method."""
    settings = StudySettings(n_runs=2, seed=0)
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


def test_convergence_marks_the_end_of_each_number_of_initial_points():
    """Configurations with different numbers of initial points each get a dotted line
    after their initial points; configurations that share one share the line."""
    f = [3.0, 1.0, 2.0, 1.5]
    runs = [
        fake_run(label, 0, f, None, n_initial)
        for label, n_initial in (("a", 2), ("b", 3), ("c", 2))
    ]

    ax = plots.convergence(fake_study(runs))

    dotted = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert sorted(np.asarray(line.get_xdata())[0] for line in dotted) == [2.5, 3.5]
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


# An initial-fraction sweep: one method, several numbers of initial points

SWEEP_N_INITIALS = (1, 2, 4)
SWEEP_N_EVALUATIONS = 6
SWEEP_N_RUNS = 3


def sweep_f(n_initial: int, index: int) -> np.ndarray:
    """The oracle objective values of run `index` of the configuration with
    `n_initial` initial points in `sweep_study`: positive, different for every run."""
    rng = np.random.default_rng(10 * n_initial + index)
    return rng.uniform(0.01, 10.0, SWEEP_N_EVALUATIONS)


def sweep_study() -> Study:
    """A sweep of one method over `SWEEP_N_INITIALS`, each configuration (labeled
    `eps-<n_initial>`) with `SWEEP_N_RUNS` runs of `SWEEP_N_EVALUATIONS` evaluations,
    measured without noise, and an oracle optimum of 0, so the regret is the running
    minimum of `sweep_f`."""
    runs = []
    for n_initial in SWEEP_N_INITIALS:
        for index in range(SWEEP_N_RUNS):
            f = list(sweep_f(n_initial, index))
            run = fake_run("hybrid_bo", index, f, f, n_initial)
            runs.append(replace(run, method=f"eps-{n_initial}"))
    return fake_study(runs, {"oracle_optimum": "0.0"})


def sweep_regret(n_initial: int) -> np.ndarray:
    """The regret of `sweep_study`'s runs with `n_initial` initial points, one row
    per run, computed with numpy."""
    rows = [sweep_f(n_initial, index) for index in range(SWEEP_N_RUNS)]
    return np.minimum.accumulate(np.array(rows), axis=1)


def rectangles(ax: Axes) -> list[tuple[float, float, int, float]]:
    """The heatmap's drawn cells as (low, high, budget, value): the span in initial
    fraction, the budget at the center of the height, and the cell's value."""
    (collection,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    values = np.asarray(collection.get_array())
    drawn = []
    for path, value in zip(collection.get_paths(), values):
        vertices = np.asarray(path.vertices)
        low, bottom = vertices.min(axis=0)
        high, top = vertices.max(axis=0)
        drawn.append((float(low), float(high), round((bottom + top) / 2), float(value)))
    return drawn


def cells(ax: Axes) -> dict[tuple[int, int], float]:
    """The heatmap's drawn cells, by (budget, n_initial): a cell's n_initial is the
    one in `SWEEP_N_INITIALS` whose initial fraction it spans."""
    drawn = {}
    for low, high, budget, value in rectangles(ax):
        (n_initial,) = [n for n in SWEEP_N_INITIALS if low < n / budget < high]
        drawn[budget, n_initial] = value
    return drawn


def spans(ax: Axes, budget: int) -> list[tuple[float, float]]:
    """The spans in initial fraction of the heatmap's cells at `budget`, in order."""
    return sorted((low, high) for low, high, b, _ in rectangles(ax) if b == budget)


def test_budget_heatmap_cells_are_the_median_log10_regret():
    """A cell at budget B and n_initial initial points is the median over the runs of
    log10 of the regret after B evaluations, and there is a cell wherever n_initial <
    B; the initial fraction is on x and the budget on y."""
    # Act
    ax = plots.budget_heatmap(sweep_study())

    # Expected: every (B, n_initial) with n_initial < B, from numpy
    expected = {
        (budget, n_initial): float(
            np.median(np.log10(sweep_regret(n_initial)[:, budget - 1]))
        )
        for n_initial in SWEEP_N_INITIALS
        for budget in range(n_initial + 1, SWEEP_N_EVALUATIONS + 1)
    }

    # Check: the cells, the limits and the labels
    drawn = cells(ax)
    assert drawn.keys() == expected.keys()
    for key, value in expected.items():
        assert np.isclose(drawn[key], value), key
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (1.5, SWEEP_N_EVALUATIONS + 0.5)
    assert ax.get_xlabel() == "initial fraction ε"
    assert ax.get_ylabel() == "evaluations (budget)"
    plt.close("all")


def test_budget_heatmap_with_a_tolerance_draws_the_fraction_of_runs_within_it():
    """With a tolerance, a cell is the fraction of runs whose regret after B
    evaluations is at most the tolerance, on a color scale from 0 to 1."""
    # Setup: the median regret of every run at its last evaluation, so some runs are
    # within it and some aren't
    tolerance = float(
        np.median(np.concatenate([sweep_regret(n)[:, -1] for n in SWEEP_N_INITIALS]))
    )

    # Act
    ax = plots.budget_heatmap(sweep_study(), tolerance=tolerance)

    # Check: every cell against numpy, and the color scale
    for (budget, n_initial), value in cells(ax).items():
        within = sweep_regret(n_initial)[:, budget - 1] <= tolerance
        assert np.isclose(value, within.mean()), (budget, n_initial)
    (collection,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert collection.get_clim() == (0.0, 1.0)
    plt.close("all")


def test_budget_heatmap_cells_span_halfway_to_their_neighbors():
    """A cell spans the initial fraction from halfway to the next smaller n_initial
    to halfway to the next larger one (as far on the outside), over its budget, and
    within [0, 1]: at budget 5 with n_initial 1, 2 and 4 that is 0.5-1.5, 1.5-3 and
    3-5 initial points."""
    ax = plots.budget_heatmap(sweep_study())

    np.testing.assert_allclose(
        spans(ax, 5), [(0.5 / 5, 1.5 / 5), (1.5 / 5, 3 / 5), (3 / 5, 1.0)]
    )
    plt.close("all")


def test_budget_heatmap_leaves_out_budgets_without_cells():
    """A budget no run reaches with an iteration (here 4 and 5: the n_initial = 1
    runs end at 3 evaluations, and the n_initial = 5 runs have no iterations before
    6) has no cells."""
    # Setup: n_initial = 1 to 3 evaluations, n_initial = 5 to 7
    runs = [
        fake_run("hybrid_bo", 0, [3.0, 2.0, 1.0], [3.0, 2.0, 1.0], n_initial=1),
        fake_run("hybrid_bo", 0, [5.0] * 7, [5.0] * 7, n_initial=5),
    ]
    study = fake_study(runs, {"oracle_optimum": "0.0"})

    # Act
    ax = plots.budget_heatmap(study)

    # Check: cells at budgets 2, 3, 6 and 7 only
    assert [budget for budget in range(2, 8) if spans(ax, budget)] == [2, 3, 6, 7]
    plt.close("all")


def test_a_configuration_sharing_n_initial_with_a_longer_one_counts_once():
    """Two configurations with the same n_initial (here one with a smaller budget)
    pass through the same first evaluations, so their runs count once: the figures
    are those of the longer configuration alone. (Counted twice, the medians would
    stay but the quartiles would move.)"""
    # Setup: a copy of the n_initial = 2 configuration, cut to 4 evaluations
    study = sweep_study()
    blocks = ("u", "x", "y", "f", "x_oracle", "y_oracle", "f_oracle")
    shorter = [
        replace(
            run,
            method="eps-2-short",
            history=replace(
                run.history,
                **{name: getattr(run.history, name)[:4] for name in blocks},
            ),
        )
        for run in study.runs
        if run.method == "eps-2"
    ]
    with_shorter = replace(study, runs=study.runs + tuple(shorter))

    # Act and check: the same cells and quartiles with and without the copy
    assert cells(plots.budget_heatmap(with_shorter)) == cells(
        plots.budget_heatmap(study)
    )
    np.testing.assert_array_equal(
        band(plots.regret_at_budget(with_shorter, 4)),
        band(plots.regret_at_budget(study, 4)),
    )
    plt.close("all")


def test_regret_at_budget_draws_the_median_and_iqr_against_the_initial_fraction():
    """At a budget B, the median regret over the runs is drawn against ε =
    n_initial / B for every n_initial < B, the interquartile range is shaded, and
    the y-axis is logarithmic."""
    budget = 4
    ax = plots.regret_at_budget(sweep_study(), budget)

    # Expected: n_initial 1 and 2 (4 has no iterations at this budget), from numpy
    n_initials = [1, 2]
    regret = [sweep_regret(n_initial)[:, budget - 1] for n_initial in n_initials]
    drawn = drawn_line(ax, "median")
    np.testing.assert_allclose(np.asarray(drawn.get_xdata()), [0.25, 0.5])
    np.testing.assert_allclose(
        np.asarray(drawn.get_ydata()), [np.median(r) for r in regret]
    )
    vertices = band(ax)
    for epsilon, values in zip([0.25, 0.5], regret):
        lower, upper = np.quantile(values, [0.25, 0.75])
        assert np.isclose(vertices, [epsilon, lower]).all(axis=1).any()
        assert np.isclose(vertices, [epsilon, upper]).all(axis=1).any()
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "initial fraction ε"
    assert ax.get_title() == "after 4 evaluations"
    plt.close("all")


def test_regret_at_budget_needs_runs_with_iterations_at_that_budget():
    """A budget that no run reaches, or that every configuration spends on initial
    points, has no regret to draw."""
    for budget in (1, SWEEP_N_EVALUATIONS + 1):
        with pytest.raises(ValueError, match="no regret"):
            plots.regret_at_budget(sweep_study(), budget)


def test_regret_by_initial_fraction_draws_one_median_line_per_n_initial():
    """One line per n_initial, labeled with it, of the median regret over the runs
    after each evaluation, on a log y-axis."""
    ax = plots.regret_by_initial_fraction(sweep_study())

    for n_initial in SWEEP_N_INITIALS:
        expected = np.median(sweep_regret(n_initial), axis=0)
        np.testing.assert_allclose(line(ax, f"n_initial = {n_initial}"), expected)
    assert ax.get_yscale() == "log"
    plt.close("all")


def test_regret_by_initial_fraction_thins_the_lines():
    """With more n_initial values than `max_lines`, an evenly spread subset is drawn,
    the smallest and largest included."""
    ax = plots.regret_by_initial_fraction(sweep_study(), max_lines=2)

    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["n_initial = 1", "n_initial = 4"]
    plt.close("all")


@pytest.mark.parametrize(
    "figure",
    [
        plots.budget_heatmap,
        functools.partial(plots.regret_at_budget, budget=4),
        plots.regret_by_initial_fraction,
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        # Another method
        {"artifact": {"method": "standard_bo"}},
        # The same method with other settings
        {"settings": StandardBOSettings(gp=ex.gp, acquisition=LCB(std_factor=2.0))},
    ],
)
def test_sweep_figures_need_a_study_of_one_method(figure, other):
    """A sweep's figures compare initial fractions of one method, so a study with
    runs of another method, or of the same method with other settings, is refused."""
    study = sweep_study()
    stranger = replace(study.runs[0], method="other", **other)
    with pytest.raises(ValueError, match="one method"):
        figure(replace(study, runs=(*study.runs, stranger)))


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
    mean, std = predict(run.surrogates[iteration][0], u[:, np.newaxis])
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
    mean, _ = predict(hybrid_run.surrogates[1][0], u[:, np.newaxis])
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
        SAA(n_scenarios=25, unknown_states_as_variables=False),
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
    _, std = predict(hybrid_lin_run.surrogates[1][0], u[:, np.newaxis])
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
    (surrogate,) = standard_run.surrogates[1]
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


# The figure of one iteration with two decision variables, on the 2D Rosenbrock
# function f = 100 (u2 - u1^2)^2 + (u1 - 1)^2 as a hybrid model with one unknown state,
# so hybrid BO has a row of y: the mechanistic state x = u1 - 1 is known, the unknown
# state y = u2 - u1^2 is measured by an oracle
ROSENBROCK_U = [Var(f"u{i}", lower=-2.048, upper=2.048) for i in (1, 2)]
ROSENBROCK_Y = Var("y", start=(-6.3, 2.1))
ROSENBROCK_SYSTEM = System(
    u=ROSENBROCK_U,
    x=[Var("x", start=(-3.1, 1.1))],
    y=[ROSENBROCK_Y],
    g=lambda v: [v.x - (v.u1 - 1)],
    h_inputs=ROSENBROCK_U,
    measured=[ROSENBROCK_Y],
)
ROSENBROCK = Problem(ROSENBROCK_SYSTEM, f=lambda v: 100 * v.y**2 + v.x**2)
ROSENBROCK_MEASURE = oracle_measure(
    ROSENBROCK_SYSTEM, lambda v: [v.u2 - v.u1**2], noise_std=0.0
)


def rosenbrock_run(
    method: Callable[..., Run],
    settings: HybridBOSettings | StandardBOSettings | None = None,
) -> Run:
    """A pickled run of `method` on the Rosenbrock problem: 2 initial points, then 2
    iterations (or design points)."""
    rng = np.random.default_rng(0)
    u = initial_points(ROSENBROCK_SYSTEM.u, 2, rng)
    if settings is None:
        return pickled(method(ROSENBROCK, ROSENBROCK_MEASURE, u, 2, rng))
    return pickled(method(ROSENBROCK, ROSENBROCK_MEASURE, u, 2, rng, settings=settings))


@pytest.fixture(scope="module")
def hybrid_2d_run() -> Run:
    """A pickled hybrid BO run on the Rosenbrock problem with SAA-EI over 10
    scenarios, the GP's prediction substituted for the unknown state."""
    approximation = SAA(n_scenarios=10, unknown_states_as_variables=False)
    return rosenbrock_run(
        hybrid_bo,
        HybridBOSettings(gp=ex.gp, acquisition=EI(), approximation=approximation),
    )


@pytest.fixture(scope="module")
def hybrid_soft_2d_run() -> Run:
    """As `hybrid_2d_run`, with SoftSAA-EI at a temperature of 10."""
    approximation = SoftSAA(
        temperature=10.0, n_scenarios=10, unknown_states_as_variables=False
    )
    return rosenbrock_run(
        hybrid_bo,
        HybridBOSettings(gp=ex.gp, acquisition=EI(), approximation=approximation),
    )


@pytest.fixture(scope="module")
def hybrid_cold_soft_2d_run() -> Run:
    """As `hybrid_soft_2d_run`, at a temperature of 0.01, where SoftSAA-EI underflows
    to 0 in float64 over most of the grid."""
    approximation = SoftSAA(
        temperature=0.01, n_scenarios=10, unknown_states_as_variables=False
    )
    return rosenbrock_run(
        hybrid_bo,
        HybridBOSettings(gp=ex.gp, acquisition=EI(), approximation=approximation),
    )


@pytest.fixture(scope="module")
def standard_2d_run() -> Run:
    """A pickled standard BO run on the Rosenbrock problem."""
    return rosenbrock_run(standard_bo, StandardBOSettings(gp=ex.gp, acquisition=EI()))


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
    figure = plots.iteration(run, ROSENBROCK, 1, measure=ROSENBROCK_MEASURE)

    # Check: the titles, the points on every panel, one legend
    assert titles(figure) == expected
    assert figure.get_suptitle() == f"{run.method}, iteration 1"
    panels = [ax for ax in figure.axes if ax.get_title()]
    n_points = run.history.n_initial + 1
    for ax in panels:
        # The last four collections are the markers: initial points, points so far,
        # incumbent, trial point
        offsets = np.concatenate([c.get_offsets() for c in ax.collections[-4:]])
        assert len(offsets) == n_points + 2  # the incumbent again, the trial point
    assert {"initial points", "incumbent", "trial point"} <= set(legend_labels(figure))
    figure.savefig(tmp_path / "figure.png")
    plt.close("all")


def test_2d_figure_without_measure_drops_the_oracle_column(hybrid_2d_run):
    """Without a measurement there are no oracle values, so the oracle panels go."""
    figure = plots.iteration(hybrid_2d_run, ROSENBROCK, 0)

    assert titles(figure) == [
        title for title in HYBRID_2D_TITLES if not title.startswith("oracle")
    ]
    plt.close("all")


def test_2d_flags_turn_panels_off(hybrid_2d_run):
    """The flags drop the same parts as in 1D: `gp` the row of y, `scenarios` the
    panels over the scenarios, `acquisition` the EI panel."""
    no_gp = plots.iteration(hybrid_2d_run, ROSENBROCK, 0, gp=False)
    no_scenarios = plots.iteration(hybrid_2d_run, ROSENBROCK, 0, scenarios=False)
    nothing = plots.iteration(
        hybrid_2d_run, ROSENBROCK, 0, gp=False, scenarios=False, acquisition=False
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
    figure = plots.iteration(run, ROSENBROCK, i, gp=False, scenarios=False)
    (mesh,) = [c for c in figure.axes[0].collections if isinstance(c, QuadMesh)]
    gray = mesh.get_array()
    assert gray is not None
    zero = ~np.ma.getmaskarray(np.ma.asarray(gray)).ravel()

    # SAA-EI's improvement at each grid point, from the GP's scenario prediction
    assert run.scenarios is not None
    u1, u2 = (np.linspace(-2.048, 2.048, 30) for _ in range(2))
    grid = np.column_stack([coordinate.ravel() for coordinate in np.meshgrid(u1, u2)])
    mean, std = predict(run.surrogates[i][0], grid)
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


def test_2d_softsaa_ei_has_no_gray_cells(hybrid_cold_soft_2d_run):
    """SoftSAA-EI is never 0, so its EI panel has no gray cells and no "EI = 0"
    legend entry, even at a low temperature, where EI underflows in float64."""
    # Act: iteration 1's EI panel alone
    figure = plots.iteration(
        hybrid_cold_soft_2d_run, ROSENBROCK, 1, gp=False, scenarios=False
    )

    # Check: no gray mesh, no legend entry for it
    meshes = [c for c in figure.axes[0].collections if isinstance(c, QuadMesh)]
    assert meshes == []
    assert "EI = 0" not in legend_labels(figure)
    plt.close("all")


def test_2d_ei_color_scale_spans_ten_decades_below_the_largest_ei(
    hybrid_cold_soft_2d_run,
):
    """Without gray cells SoftSAA's log10 EI can span thousands of decades, which
    would flatten the color scale, so the scale stops 10 decades below the largest
    EI: smaller EI takes the lowest color and the colorbar gets a `min` arrow."""
    # Act: iteration 1's EI panel alone
    figure = plots.iteration(
        hybrid_cold_soft_2d_run, ROSENBROCK, 1, gp=False, scenarios=False
    )

    # Check: the levels span 10 decades, with the arrow on the contours and colorbar
    (contours,) = [c for c in figure.axes[0].collections if isinstance(c, ContourSet)]
    levels = np.asarray(contours.levels)
    np.testing.assert_allclose(levels[-1] - levels[0], 10)
    assert contours.extend == "min"
    assert contours.colorbar is not None
    assert contours.colorbar.extend == "min"
    plt.close("all")


@pytest.mark.parametrize(
    "run_name", ["hybrid_2d_run", "hybrid_soft_2d_run", "standard_2d_run"]
)
def test_2d_acquisition_panel_checks_the_replay_at_the_trial_point(request, run_name):
    """The acquisition panel notes the stored and the replayed log10 EI at the trial
    point, and they agree: the replay solves the stored acquisition problem."""
    # Act: iteration 1's EI panel alone
    run = request.getfixturevalue(run_name)
    figure = plots.iteration(run, ROSENBROCK, 1, gp=False, scenarios=False)

    # Check: the note's two numbers, against the stored value in log10 EI
    (note,) = figure.axes[0].texts
    stored, replayed = (float(part.split()[-1]) for part in note.get_text().split(","))
    value = run.acquisitions[1].value
    soft = run_name == "hybrid_soft_2d_run"
    log10_ei = -value / np.log(10) if soft else np.log10(-value)
    np.testing.assert_allclose(stored, log10_ei, rtol=1e-3)
    np.testing.assert_allclose(replayed, stored, rtol=1e-3)
    plt.close("all")


def test_2d_softsaa_note_is_finite_where_its_ei_underflows(hybrid_soft_2d_run):
    """SoftSAA stores -log EI, which can exceed 745, where EI underflows to 0 in
    float64; the note's log10 EI is still finite: -value / ln 10."""
    # Setup: iteration 1's stored -log EI replaced with 1e6 (EI = e^-1e6)
    run = hybrid_soft_2d_run
    acquisitions = list(run.acquisitions)
    acquisitions[1] = replace(acquisitions[1], value=1e6)
    run = replace(run, acquisitions=acquisitions)

    # Act: iteration 1's EI panel alone. The replay's value isn't the stored one, so
    # it warns; `pytest.warns` passes only if the block warns with this category
    with pytest.warns(ReplayWarning):
        figure = plots.iteration(run, ROSENBROCK, 1, gp=False, scenarios=False)

    # Check: the stored number in the note, log10 EI = -1e6 / ln 10
    (note,) = figure.axes[0].texts
    stored = float(note.get_text().split(",")[0].split()[-1])
    np.testing.assert_allclose(stored, -1e6 / np.log(10), rtol=1e-3)
    plt.close("all")
