"""Evaluation: measurements, the oracle simulation, the root-finding problem, History.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value; `monkeypatch` is a fixture built into pytest.
"""

import functools
import importlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest
from scipy.special import lambertw

import hybrid_bo
from hybrid_bo import (
    History,
    OutOfBoundsWarning,
    Problem,
    ReconstructionWarning,
    RootFindingError,
    System,
    Var,
    incumbents,
    oracle_incumbents,
    oracle_measure,
)
from hybrid_bo.evaluation import evaluate, reconstruct
from hybrid_bo.history import n_iterations
from hybrid_bo.oracle import OracleMeasurement


def illustrative_measure(problem: Problem, noise_std: float):
    """The oracle simulation of the illustrative example, with Gaussian noise of
    `noise_std` on the measurement."""
    return oracle_measure(
        problem.system,
        oracle=lambda v: [cas.sin(v.u)],
        noise_std=noise_std,
    )


def two_branch_problem() -> Problem:
    """y = x**2 with y measured: x = +sqrt(y) and x = -sqrt(y) are both roots.

    The start bounds of x only cover the positive branch.
    """
    u = Var("u", -1.0, 1.0)
    x = Var("x", start=(0.5, 2.0))
    y = Var("y", start=(0.0, 4.0))
    system = System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x**2 - v.y],
        h_inputs=[u],
        measured=[y],
    )
    return Problem(system, f=lambda v: v.x + v.u)


def closed_form(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The illustrative example solved by hand, for u (n,): y = sin(u), and
    x + exp(x) = y gives x = y - W(exp(y)) (Lambert W)."""
    y = np.sin(u)
    x = y - lambertw(np.exp(y)).real
    return x, y, (6 * x - 2) ** 2 * np.sin(12 * x - 4)


# `filterwarnings("error")` turns warnings into failures, so the test also checks
# that the reconstruction agrees with the oracle
@pytest.mark.filterwarnings("error")
def test_noise_free_evaluation_matches_the_closed_form(illustrative_problem):
    """Without noise, the oracle values and the reconstruction both solve the
    illustrative example exactly."""
    # Trial points across the bounds of u
    u = np.array([-2.0, -0.5, 0.0, 1.0, 2.0])
    measure = illustrative_measure(illustrative_problem, noise_std=0.0)

    history = evaluate(
        illustrative_problem, measure, u[:, np.newaxis], np.random.default_rng(0)
    )

    # Measured, reconstructed and oracle values all match the closed form
    x, y, f = closed_form(u)
    assert history.x_oracle is not None and history.y_oracle is not None
    assert history.f_oracle is not None
    for actual, expected in [
        (history.x, x),
        (history.y, y),
        (history.x_oracle, x),
        (history.y_oracle, y),
    ]:
        np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(history.f, f, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(history.f_oracle, f, rtol=1e-8, atol=1e-10)
    assert history.n_initial == len(u)


def test_noise_is_the_first_stream_draw(illustrative_problem):
    """The measurement noise is the first draw from the evaluation's generator."""
    # The root-finding starts come from lhs, which spawns instead of consuming the
    # stream
    measure = illustrative_measure(illustrative_problem, noise_std=0.1)

    history = evaluate(
        illustrative_problem, measure, np.array([[-0.5]]), np.random.default_rng(0)
    )

    noise = np.random.default_rng(0).normal(0.0, 0.1, size=1)
    assert history.y_oracle is not None
    np.testing.assert_allclose(history.y - history.y_oracle, [noise], rtol=1e-12)


def test_root_finding_starts_cover_the_unmeasured_variables(monkeypatch):
    """The root-finding starts are a Latin hypercube over the start bounds of the
    variables that are not measured, and the root found solves g."""
    # Two mechanistic states and one measured unknown state: the starts are a
    # two-dimensional Latin hypercube over the start bounds of x1 and x2
    u = Var("u", -1.0, 1.0)
    x1, x2 = Var("x1", start=(0.0, 1.0)), Var("x2", start=(2.0, 3.0))
    y = Var("y", start=(-1.0, 1.0))
    system = System(
        u=[u],
        x=[x1, x2],
        y=[y],
        g=lambda v: [v.x1 - v.y - v.u, v.x2 - v.x1**2],
        h_inputs=[u],
        measured=[y],
    )
    problem = Problem(system, f=lambda v: v.x2)

    # Record the arguments of each lhs call. `monkeypatch.setattr` replaces lhs in
    # the evaluation module for this test only
    calls = []

    def spy(lower, upper, n, rng):
        calls.append((lower, upper, n))
        return hybrid_bo.sampling.lhs(lower, upper, n, rng)

    monkeypatch.setattr(hybrid_bo.evaluation, "lhs", spy)
    history = evaluate(
        problem,
        lambda u, rng: np.array([0.5]),
        np.array([[0.25]]),
        np.random.default_rng(0),
        n_starts=7,
    )

    # One lhs call over the start bounds of x1 and x2, and the root x1 = y + u
    [(lower, upper, n)] = calls
    np.testing.assert_array_equal(lower, [0.0, 2.0])
    np.testing.assert_array_equal(upper, [1.0, 3.0])
    assert n == 7
    np.testing.assert_allclose(history.x, [[0.75, 0.75**2]])
    np.testing.assert_allclose(history.f, [0.75**2])


def test_measure_without_an_oracle_gives_no_oracle_values():
    """A measurement that returns only the measured variables gives reconstructed
    values and no oracle values."""
    problem = two_branch_problem()

    history = evaluate(
        problem,
        lambda u, rng: np.array([u[0] + 2.0]),
        np.array([[-1.0], [0.0]]),
        np.random.default_rng(0),
    )

    np.testing.assert_allclose(history.x, [[1.0], [np.sqrt(2.0)]])
    np.testing.assert_allclose(history.y, [[1.0], [2.0]])
    np.testing.assert_allclose(history.f, [0.0, np.sqrt(2.0)])
    assert history.x_oracle is None
    assert history.y_oracle is None
    assert history.f_oracle is None


def test_disagreement_with_the_oracle_warns():
    """A reconstruction on a different branch from the oracle values warns, and both
    are kept."""

    # The oracle is on the negative branch; the reconstruction can only find the
    # positive one
    def measure(u, rng):
        return OracleMeasurement(
            measured=np.array([1.0]), x=np.array([-1.0]), y=np.array([1.0])
        )

    # `pytest.warns` fails the test unless the block warns with a message matching
    # `match`
    with pytest.warns(ReconstructionWarning, match="oracle"):
        history = evaluate(
            two_branch_problem(),
            measure,
            np.array([[0.0]]),
            np.random.default_rng(0),
        )

    np.testing.assert_allclose(history.x, [[1.0]])
    assert history.x_oracle is not None and history.f_oracle is not None
    np.testing.assert_allclose(history.x_oracle, [[-1.0]])
    np.testing.assert_allclose(history.f_oracle, [-1.0])


@pytest.mark.filterwarnings("error")
def test_noisy_reconstruction_on_the_oracle_branch_does_not_warn():
    """A reconstruction moved by noise, but on the oracle's branch, does not warn."""

    # Noise moves the reconstruction away from the oracle values, but along the
    # same branch
    def measure(u, rng):
        return OracleMeasurement(
            measured=np.array([1.1]), x=np.array([1.0]), y=np.array([1.0])
        )

    history = evaluate(
        two_branch_problem(), measure, np.array([[0.0]]), np.random.default_rng(0)
    )

    np.testing.assert_allclose(history.x, [[np.sqrt(1.1)]])


def test_failed_root_finding_raises_with_the_points_before():
    """When the root-finding fails, the error carries the history of the points
    evaluated before."""

    # x**2 = -1 has no real root
    def measure(u, rng):
        return np.array([1.0]) if u[0] < 0 else np.array([-1.0])

    # `pytest.raises` fails the test unless the block raises that exception; `info`
    # holds the exception
    with pytest.raises(RootFindingError) as info:
        evaluate(
            two_branch_problem(),
            measure,
            np.array([[-0.5], [0.5], [-0.5]]),
            np.random.default_rng(0),
            n_starts=5,
        )

    # Only the first point, before the failing second one
    history = info.value.history
    assert history is not None
    np.testing.assert_array_equal(history.u, [[-0.5]])
    np.testing.assert_allclose(history.x, [[1.0]])


def test_failed_oracle_simulation_raises():
    """When the oracle simulation has no solution, evaluation raises with an empty
    history."""
    # y = -1 - x**2 and y = x**2 have no common real root
    problem = two_branch_problem()
    measure = oracle_measure(
        problem.system, oracle=lambda v: -1 - v.x**2, n_starts=5, noise_std=0.0
    )

    with pytest.raises(RootFindingError) as info:
        evaluate(problem, measure, np.array([[0.0]]), np.random.default_rng(0))

    assert info.value.history is not None
    assert info.value.history.u.shape == (0, 1)


def no_known_equations_problem() -> Problem:
    """y measured and no mechanistic states, so the root-finding problem is empty."""
    u, y = Var("u", -1.0, 1.0), Var("y", start=(-1.0, 1.0))
    system = System(u=[u], y=[y], h_inputs=[u], measured=[y])
    return Problem(system, f=lambda v: v.y)


def measured_outside_g_problem() -> Problem:
    """x = u, with y measured but in no known equation: only f sees it, so the
    root-finding problem has a root whatever y is."""
    u = Var("u", -1.0, 1.0)
    x = Var("x", start=(-1.0, 1.0))
    y = Var("y", start=(-1.0, 1.0))
    system = System(
        u=[u], x=[x], y=[y], g=lambda v: [v.x - v.u], h_inputs=[u], measured=[y]
    )
    return Problem(system, f=lambda v: v.x + v.y)


# Two `parametrize` marks run the test once per combination of their entries; `ids`
# names the entries in pytest's output
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "problem",
    [no_known_equations_problem(), two_branch_problem(), measured_outside_g_problem()],
    ids=["no-known-equations", "in-g", "outside-g"],
)
def test_a_measurement_that_is_not_finite_raises(problem: Problem, value: float):
    """A measured value that isn't finite fails the evaluation, whether or not the
    root-finding problem would notice it, and the error keeps the points before."""

    # A measurement that is finite at the first point and not at the second
    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A finite value for negative u, else `value`."""
        return np.array([0.5]) if u[0] < 0 else np.array([value])

    # Evaluating the points stops at the second
    with pytest.raises(RootFindingError, match="isn't finite") as info:
        evaluate(
            problem,
            measure,
            np.array([[-0.5], [0.5], [-0.5]]),
            np.random.default_rng(0),
            n_starts=5,
        )

    # Only the first point, before the failing second one
    history = info.value.history
    assert history is not None
    np.testing.assert_array_equal(history.u, [[-0.5]])
    np.testing.assert_array_equal(history.y, [[0.5]])


@pytest.mark.parametrize(
    "method",
    [
        functools.partial(
            hybrid_bo.hybrid_bo,
            gp=ex.gp,
            acquisition=hybrid_bo.EI(),
            approximation=hybrid_bo.SAA(n_scenarios=5),
        ),
        hybrid_bo.latin_hypercube_sampling,
    ],
    ids=["hybrid-bo", "lhs"],
)
def test_a_nan_measured_variable_outside_g_fails_the_run(method: Callable):
    """The known equations have a root whatever a measured variable outside `g` is, so
    a run used to complete with NaN at its last point. It fails, and the partial run
    holds only finite points."""
    # A measurement that returns NaN at the last of 4 evaluations
    calls = []

    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """y = sin(u), and NaN at the fourth call."""
        calls.append(u)
        return np.array([np.nan]) if len(calls) == 4 else np.sin(u)

    # The run stops at that evaluation
    with pytest.raises(RootFindingError, match="isn't finite") as info:
        method(
            measured_outside_g_problem(),
            measure,
            initial_points=np.array([[-0.5], [0.5]]),
            n_iterations=2,
            rng=np.random.default_rng(0),
        )

    # The partial run: the three points before it, all of them finite
    run = info.value.run
    assert run is not None
    assert len(run.history.u) == 3
    assert np.isfinite(run.history.x).all()
    assert np.isfinite(run.history.y).all()
    assert np.isfinite(run.history.f).all()


def test_measure_returning_the_wrong_shape_raises():
    """A measurement returning more values than there are measured variables raises."""
    with pytest.raises(ValueError, match="measured"):
        evaluate(
            two_branch_problem(),
            lambda u, rng: np.array([1.0, 2.0]),
            np.array([[0.0]]),
            np.random.default_rng(0),
        )


def test_oracle_must_give_one_value_per_unknown_state():
    """An oracle with more expressions than unknown states is rejected."""
    with pytest.raises(ValueError, match="oracle"):
        oracle_measure(
            two_branch_problem().system, oracle=lambda v: [v.u, 2 * v.u], noise_std=0.0
        )


# `parametrize` runs the test once per experiment, with its module name as `name`
@pytest.mark.parametrize("name", ["illustrative_example", "illustrative_example_2d"])
def test_every_experiment_measurement_has_the_same_repr_in_another_process(name):
    """Each experiment's measurement is described by the same text in another
    process: a run's artifact holds that text, so it can't depend on the process."""
    measure = importlib.import_module(name).measure

    other = subprocess.run(
        [sys.executable, "-c", f"import {name}; print(repr({name}.measure))"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )

    assert other.stdout == repr(measure) + "\n"


def history_of(f: list[float], f_oracle: list[float] | None = None) -> History:
    """A history with the given measured and oracle objective values, the first two
    of them initial points; the other fields are zero."""
    n = len(f)
    return History(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 1)),
        y=np.zeros((n, 1)),
        f=np.array(f),
        x_oracle=None if f_oracle is None else np.zeros((n, 1)),
        y_oracle=None if f_oracle is None else np.zeros((n, 1)),
        f_oracle=None if f_oracle is None else np.array(f_oracle),
        n_initial=2,
    )


def test_n_iterations_counts_the_points_after_the_initial_ones():
    """A history of five points, two of them initial, has three iterations."""
    assert n_iterations(history_of([3.0, 1.0, 2.0, 0.5, 0.2])) == 3


def test_incumbents_are_the_running_minimum_of_measured_f():
    """The incumbent after each evaluation is the best measured f so far."""
    history = history_of([3.0, 1.0, 2.0, 0.5])

    np.testing.assert_array_equal(incumbents(history), [3.0, 1.0, 1.0, 0.5])


def test_oracle_incumbents_are_the_oracle_f_at_the_measured_incumbent():
    """Under noise, the benchmark metric follows the point the measured f ranks best,
    not the best oracle f so far."""
    # The third point measures worse than the second but is better at the oracle;
    # the fourth measures best but is worse at the oracle
    history = history_of(f=[3.0, 1.0, 2.0, 0.5], f_oracle=[3.0, 1.5, 0.5, 2.0])

    np.testing.assert_array_equal(oracle_incumbents(history), [3.0, 1.5, 1.5, 2.0])


def test_oracle_incumbents_keep_the_earlier_point_on_a_tie():
    """A later point that only equals the incumbent's measured f doesn't replace it."""
    history = history_of(f=[1.0, 1.0, 1.0], f_oracle=[2.0, 0.0, 3.0])

    np.testing.assert_array_equal(oracle_incumbents(history), [2.0, 2.0, 2.0])


def test_oracle_incumbents_without_noise_are_the_running_minimum_of_oracle_f(
    illustrative_problem,
):
    """Without noise the measured and oracle f agree, so the benchmark metric is the
    running minimum of the oracle f (the paper's convergence figures)."""
    u = np.array([[2.0], [-0.5], [1.0], [0.0], [-2.0], [0.4]])
    measure = illustrative_measure(illustrative_problem, noise_std=0.0)

    history = evaluate(illustrative_problem, measure, u, np.random.default_rng(0))

    assert history.f_oracle is not None
    np.testing.assert_allclose(
        oracle_incumbents(history), np.minimum.accumulate(history.f_oracle)
    )


def test_oracle_incumbents_need_oracle_values():
    """A history from a measurement without an oracle has no benchmark metric."""
    with pytest.raises(ValueError, match="oracle"):
        oracle_incumbents(history_of([3.0, 1.0]))


def test_oracle_optimum_solves_the_problem_with_the_oracle_in_place_of_h():
    """With y = u known, min (x - 1)^2 + y subject to x = y over u in [-2, 2] is at
    u = 0.5, where f = 0.75."""
    u = hybrid_bo.Var("u", -2.0, 2.0)
    x = hybrid_bo.Var("x", start=(-3.0, 3.0))
    y = hybrid_bo.Var("y", start=(-3.0, 3.0))
    system = hybrid_bo.System(
        u=[u], x=[x], y=[y], g=lambda v: [v.x - v.y], h_inputs=[u], measured=[y]
    )
    problem = hybrid_bo.Problem(system, f=lambda v: (v.x - 1) ** 2 + v.y)

    optimum = hybrid_bo.oracle_optimum(
        problem, lambda v: [v.u], np.random.default_rng(0)
    )

    assert optimum == pytest.approx(0.75, abs=1e-8)


def test_oracle_optimum_rejects_an_oracle_of_the_wrong_size(illustrative_problem):
    """The oracle must give one value per unknown state, as for the measurement."""
    with pytest.raises(ValueError, match="one value per unknown state"):
        hybrid_bo.oracle_optimum(
            illustrative_problem,
            lambda v: [v.u, v.u],
            np.random.default_rng(0),
        )


def test_reconstruct_solves_the_unmeasured_states_of_given_measurements(
    illustrative_problem,
):
    """Given u and y, `reconstruct` gives x from x + exp(x) = y, in closed form with
    the Lambert W function, and f at those states, calling no measurement."""
    # Three measurements of y
    u = np.array([[-1.0], [0.0], [1.0]])
    measured = np.array([[-0.5], [0.3], [1.0]])

    history = reconstruct(
        illustrative_problem, u=u, measured=measured, rng=np.random.default_rng(0)
    )

    # The closed form of the unmeasured state, and f there
    x = measured - lambertw(np.exp(measured)).real
    f = ((6 * x - 2) ** 2 * np.sin(12 * x - 4)).ravel()
    np.testing.assert_allclose(history.x, x, atol=1e-8)
    np.testing.assert_array_equal(history.y, measured)
    np.testing.assert_array_equal(history.u, u)
    np.testing.assert_allclose(history.f, f, atol=1e-6)

    # Given points have no oracle values, and all are initial points
    assert history.f_oracle is None
    assert history.n_initial == 3


# `parametrize` runs the test once per entry, with the entry as the argument
@pytest.mark.parametrize(
    ("u", "measured", "message"),
    [
        (np.zeros(4), np.zeros((4, 1)), "u need shape"),
        (np.zeros((4, 2)), np.zeros((4, 1)), "u need shape"),
        (np.zeros((4, 1)), np.zeros(4), "measured variables need shape"),
        (np.zeros((4, 1)), np.zeros((3, 1)), "measured variables need shape"),
    ],
    ids=["flat u", "too many u", "flat measured", "fewer measured"],
)
def test_reconstruct_rejects_malformed_measurements(
    illustrative_problem, u, measured, message
):
    """Measurements need a row per point and a column per variable."""
    with pytest.raises(ValueError, match=message):
        reconstruct(
            illustrative_problem, u=u, measured=measured, rng=np.random.default_rng(0)
        )


def square_root_problem() -> Problem:
    """x^2 = y with x in [0, 3]: y < 0 has no root and y > 9 has none within the bounds."""
    u_var, x_var = Var("u", -1.0, 1.0), Var("x", 0.0, 3.0)
    y_var = Var("y", start=(-2.0, 2.0))
    system = System(
        u=[u_var],
        x=[x_var],
        y=[y_var],
        g=lambda v: [v.x**2 - v.y],
        h_inputs=[u_var],
        measured=[y_var],
    )
    return Problem(system, f=lambda v: v.x)


def test_reconstruct_names_every_bad_row_and_keeps_the_others():
    """Every row is tried, one error names the bad rows, and `error.history` holds the
    rows that worked. A row is bad with no root, with a root outside the bounds of the
    unmeasured states, or with a value that isn't finite."""
    # Good, no root, good, root outside the bounds, not finite
    problem = square_root_problem()
    u = np.array([[-0.5], [0.0], [0.25], [0.5], [0.75]])
    measured = np.array([[1.0], [-1.0], [4.0], [16.0], [np.nan]])

    with pytest.raises(RootFindingError, match=r"rows \[1, 3, 4\]") as caught:
        reconstruct(problem, u=u, measured=measured, rng=np.random.default_rng(0))

    # The two good rows, in order, with their roots
    history = caught.value.history
    assert history is not None
    np.testing.assert_array_equal(history.u, u[[0, 2]])
    np.testing.assert_allclose(history.x, [[1.0], [2.0]], atol=1e-8)
    assert history.n_initial == 2


def test_a_reconstruction_with_no_good_rows_has_an_empty_history():
    """When every row fails, `error.history` has no points and no oracle values."""
    problem = square_root_problem()
    measured = np.array([[-1.0], [-2.0]])

    with pytest.raises(RootFindingError, match=r"rows \[0, 1\]") as caught:
        reconstruct(
            problem, u=np.zeros((2, 1)), measured=measured, rng=np.random.default_rng(0)
        )

    history = caught.value.history
    assert history is not None
    assert len(history.u) == 0
    assert history.f_oracle is None


def test_a_row_that_is_not_finite_is_a_bad_row_with_no_known_equations():
    """With M = 0 nothing is solved, so no root can fail: a NaN is still a bad row."""
    u_var, y_var = Var("u", -1.0, 1.0), Var("y", -1.0, 1.0)
    system = System(u=[u_var], y=[y_var], h_inputs=[u_var], measured=[y_var])
    problem = Problem(system, f=lambda v: v.y)
    measured = np.array([[0.5], [np.nan]])

    with pytest.raises(RootFindingError, match=r"rows \[1\]") as caught:
        reconstruct(
            problem, u=np.zeros((2, 1)), measured=measured, rng=np.random.default_rng(0)
        )

    history = caught.value.history
    assert history is not None
    np.testing.assert_array_equal(history.y, [[0.5]])


def test_reconstruct_warns_of_values_outside_their_bounds_and_keeps_them():
    """A u or a measured variable outside its bounds is still a measurement: one warning
    names the rows of each, and the history holds them."""
    # u is bounded by 1 and y by 2; x + exp(x) = y keeps a root for every y
    u_var, y_var = Var("u", -1.0, 1.0), Var("y", -2.0, 2.0)
    x_var = Var("x", start=(-3.0, 2.0))
    system = System(
        u=[u_var],
        x=[x_var],
        y=[y_var],
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        h_inputs=[u_var],
        measured=[y_var],
    )
    problem = Problem(system, f=lambda v: v.x)
    u = np.array([[0.0], [1.5], [0.5]])
    measured = np.array([[0.5], [0.5], [3.0]])

    # `warns` passes only if the block gives that warning
    with pytest.warns(OutOfBoundsWarning) as caught:
        history = reconstruct(
            problem, u=u, measured=measured, rng=np.random.default_rng(0)
        )

    # One warning, naming the rows of each variable, and every row kept
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "u at rows [1]" in message
    assert "y at rows [2]" in message
    np.testing.assert_array_equal(history.u, u)


def test_reconstruct_gives_no_warning_within_the_bounds(illustrative_problem, recwarn):
    """Values inside their bounds warn of nothing (`recwarn` records warnings)."""
    reconstruct(
        illustrative_problem,
        u=np.array([[0.0]]),
        measured=np.array([[0.3]]),
        rng=np.random.default_rng(0),
    )

    assert not [w for w in recwarn if issubclass(w.category, OutOfBoundsWarning)]
