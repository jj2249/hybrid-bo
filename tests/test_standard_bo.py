"""Standard BO, the optimization loop and the analytic acquisition functions.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest
import scipy

from hybrid_bo import (
    EI,
    LCB,
    NoSolutionError,
    RootFindingError,
    Run,
    incumbents,
    initial_points,
    measure_oracle,
    standard_bo,
)
from hybrid_bo.acquisition import AcquisitionSolution, standard_acquisition
from hybrid_bo.bo import iteration_problem, optimization_loop
from hybrid_bo.gp import FittedGP, fit_gp, predict
from hybrid_bo.hybrid_model import _pack, nominal_scenario
from hybrid_bo.settings import StandardBOSettings


def illustrative_measure(problem):
    """The oracle simulation of the illustrative example, without noise."""
    return measure_oracle(problem, oracle=lambda v: [cas.sin(v.u)], noise_std=0.0)


def illustrative_run(problem, settings=None) -> Run:
    """A standard BO run on the illustrative example: 2 initial points, 2 iterations,
    with the default settings unless `settings` is given."""
    settings = (
        StandardBOSettings(gp=ex.gp, acquisition=EI()) if settings is None else settings
    )
    rng = np.random.default_rng(0)
    u = initial_points(problem.system.u, 2, rng)
    measure = illustrative_measure(problem)
    return standard_bo(
        problem, measure, initial_points=u, n_iterations=2, rng=rng, **vars(settings)
    )


def fitted_illustrative_gp():
    """The default GP and its fit to four points of u -> f."""
    u = np.array([[-1.5], [-0.5], [0.4], [1.7]])
    f = np.array([0.3, -1.2, 2.0, 0.8])
    return fit_gp(ex.gp, u, f, np.random.default_rng(0))


def acquisition_values(nlp, u):
    """The objective of the acquisition problem at each row of u (n, N), shape (n,)."""
    return np.array([nlp.f(_pack(point)).full().item() for point in u])


# Acquisition functions


def test_ei_matches_the_closed_form(illustrative_problem):
    """The EI acquisition problem minimizes minus the closed-form expected
    improvement, computed here with scipy from the GP's mean and std."""
    fitted = fitted_illustrative_gp()
    u = np.linspace(-2.0, 2.0, 7)[:, np.newaxis]
    incumbent = -1.2

    nlp = standard_acquisition(illustrative_problem, fitted, incumbent, EI()).nlp

    # Reference: EI = (incumbent - mean) Phi(gamma) + std phi(gamma)
    mean, std = predict(fitted, u)
    gamma = (incumbent - mean) / std
    improvement = incumbent - mean
    expected = improvement * scipy.stats.norm.cdf(gamma) + std * scipy.stats.norm.pdf(
        gamma
    )
    np.testing.assert_allclose(
        acquisition_values(nlp, u), -expected, rtol=1e-8, atol=1e-12
    )


def test_lcb_is_mean_minus_std_factor_times_std(illustrative_problem):
    """The LCB acquisition problem minimizes mean - std_factor * std of the GP."""
    fitted = fitted_illustrative_gp()
    u = np.linspace(-2.0, 2.0, 7)[:, np.newaxis]

    nlp = standard_acquisition(
        illustrative_problem, fitted, 0.0, LCB(std_factor=2.0)
    ).nlp

    mean, std = predict(fitted, u)
    np.testing.assert_allclose(acquisition_values(nlp, u), mean - 2.0 * std, rtol=1e-12)


def test_acquisition_problem_is_over_u_within_its_bounds(illustrative_problem):
    """The acquisition problem's variables are the decision variables, boxed by their
    bounds, with no equality constraints; its starts come from their start bounds."""
    fitted = fitted_illustrative_gp()

    acquisition_problem = standard_acquisition(illustrative_problem, fitted, 0.0, EI())

    nlp, (lower, upper) = acquisition_problem.nlp, acquisition_problem.start_bounds
    np.testing.assert_array_equal(nlp.lower, [-2.0])
    np.testing.assert_array_equal(nlp.upper, [2.0])
    np.testing.assert_array_equal(lower, [-2.0])
    np.testing.assert_array_equal(upper, [2.0])
    assert nlp.g is None


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize("std_factor", [-1.0, np.inf, np.nan])
def test_lcb_needs_a_finite_nonnegative_std_factor(std_factor):
    """LCB rejects a negative or non-finite std_factor."""
    # `pytest.raises` passes only if the block raises this exception, with a message
    # matching the regular expression `match`
    with pytest.raises(ValueError, match="std_factor"):
        LCB(std_factor=std_factor)


@pytest.mark.parametrize("acquisition", ["ei", LCB, None])
def test_settings_reject_anything_but_ei_or_lcb(acquisition):
    """The settings accept only an EI or LCB instance, not a name or the class."""
    with pytest.raises(TypeError, match=r"EI\(\) or LCB"):
        StandardBOSettings(gp=ex.gp, acquisition=acquisition)


def test_acquisition_builder_rejects_anything_but_ei_or_lcb(illustrative_problem):
    """The acquisition builder checks its acquisition function as the settings do."""
    fitted = fitted_illustrative_gp()

    with pytest.raises(TypeError, match=r"EI\(\) or LCB"):
        standard_acquisition(illustrative_problem, fitted, 0.0, "lcb")  # ty: ignore[invalid-argument-type]


# Standard BO


def test_run_has_one_surrogate_per_iteration(illustrative_problem):
    """A run holds the initial points, then one trial point per iteration within the
    bounds, and per iteration the GP fitted to the points before its trial point."""
    run = illustrative_run(illustrative_problem)

    # The points: 2 initial points, then 2 trial points
    assert run.dataset.n_initial == 2
    assert run.dataset.u.shape == (4, 1)
    assert np.all((-2.0 <= run.dataset.u) & (run.dataset.u <= 2.0))

    # The surrogates: one per iteration, fitted on the points before its trial point,
    # each a 1-tuple of the GP of f
    assert len(run.surrogates) == 2
    assert all(isinstance(fitted, FittedGP) for (fitted,) in run.surrogates)
    assert [len(fitted.X) for (fitted,) in run.surrogates] == [2, 3]


def test_run_pickles_without_the_code_that_produced_it(
    illustrative_problem, data_only_pickle
):
    """A run, its fitted GPs and settings included, loads from a pickle without
    user code."""
    run = illustrative_run(illustrative_problem)

    loaded = data_only_pickle(run)

    assert loaded.method == run.method
    np.testing.assert_array_equal(loaded.dataset.u, run.dataset.u)
    np.testing.assert_array_equal(loaded.surrogates[-1][0].X, run.surrogates[-1][0].X)
    assert loaded.settings == run.settings


def test_standard_bo_runs_with_lcb(illustrative_problem):
    """LCB goes through the whole of standard BO, and the run keeps it in its
    settings."""
    settings = StandardBOSettings(gp=ex.gp, acquisition=LCB(std_factor=2.0))

    run = illustrative_run(illustrative_problem, settings)

    assert len(run.acquisitions) == 2
    assert run.settings == settings


def test_what_chooses_the_method_is_required_and_keyword_only(illustrative_problem):
    """Standard BO has no default GP or acquisition function, and every argument after
    the measurement is keyword-only."""
    measure = illustrative_measure(illustrative_problem)
    rng = np.random.default_rng(0)

    with pytest.raises(TypeError, match="gp"):
        standard_bo(illustrative_problem, measure, n_iterations=0, rng=rng)  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        standard_bo(illustrative_problem, measure, 0, rng)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_failed_evaluation_carries_the_partial_run(illustrative_problem):
    """When evaluating a trial point fails, the error carries the run so far."""
    # A `measure` that returns NaN at its third evaluation (the first trial point),
    # which is rejected before the root-finding problem
    calls = []

    def measure(u, rng):
        calls.append(u)
        return np.sin(u) if len(calls) < 3 else np.array([np.nan])

    # `as info` keeps the raised exception, as `info.value`, for the checks below
    with pytest.raises(RootFindingError, match="isn't finite") as info:
        standard_bo(
            illustrative_problem,
            measure,
            initial_points=np.array([[-0.5], [0.5]]),
            n_iterations=2,
            rng=np.random.default_rng(0),
            gp=ex.gp,
            acquisition=EI(),
        )

    # The partial run: the initial points, and no surrogate (the surrogate behind the
    # failed trial point is dropped with it); the error's dataset is the run's
    run = info.value.run
    assert isinstance(run, Run)
    assert run.dataset.u.shape == (2, 1)
    assert run.dataset.n_initial == 2
    assert run.surrogates == []
    assert run.method == "standard_bo"
    assert info.value.dataset is run.dataset


def test_stored_acquisition_value_is_the_objective_at_the_trial_point(
    illustrative_problem,
):
    """Each iteration stores the acquisition problem's objective (-EI) at its trial
    point; standard BO has no scenario states."""
    run = illustrative_run(illustrative_problem)

    dataset = run.dataset
    assert len(run.acquisitions) == 2
    for i, (solution, (fitted,)) in enumerate(
        zip(run.acquisitions, run.surrogates, strict=True)
    ):
        point = dataset.n_initial + i
        incumbent = float(incumbents(dataset)[point - 1])
        nlp = standard_acquisition(illustrative_problem, fitted, incumbent, EI()).nlp
        expected = acquisition_values(nlp, dataset.u[point : point + 1])[0]
        assert solution.value == pytest.approx(expected, abs=1e-12)
        assert solution.x is None and solution.y is None


def test_each_iteration_problem_gives_the_stored_acquisition_value(
    illustrative_problem,
):
    """`iteration_problem` rebuilds each iteration's problem with the incumbent before
    its trial point, so its objective at the trial point is the stored value."""
    run = illustrative_run(illustrative_problem)

    for i, solution in enumerate(run.acquisitions):
        acquisition_problem = iteration_problem(run, illustrative_problem, i)
        trial = run.dataset.u[run.dataset.n_initial + i]
        assert float(acquisition_problem.nlp.f(trial)) == pytest.approx(
            solution.value, rel=1e-12
        )


# The optimization loop, with test trial-point functions in place of a method


def test_loop_evaluates_each_trial_point_after_the_initial_points(
    illustrative_problem,
):
    """The loop evaluates the initial points, then each trial point in turn, and keeps
    the surrogate returned with each trial point."""
    # A trial-point function that returns fixed points, and a label for a surrogate
    problem = illustrative_problem
    rng = np.random.default_rng(0)
    trial_points = iter([np.array([1.0]), np.array([-1.0])])

    def trial_point(problem, dataset, rng):
        n = len(dataset.u)
        u = next(trial_points)
        return u, f"surrogate behind point {n}", AcquisitionSolution(u, n)

    run = optimization_loop(
        problem,
        illustrative_measure(problem),
        np.array([[0.5]]),
        2,
        trial_point,
        rng,
        "test",
        settings=None,
        scenarios=None,
    )

    # Every point is evaluated (y = sin(u) is the oracle), in order
    np.testing.assert_array_equal(run.dataset.u, [[0.5], [1.0], [-1.0]])
    np.testing.assert_allclose(run.dataset.y, np.sin(run.dataset.u), atol=1e-8)
    assert run.dataset.n_initial == 1
    assert run.surrogates == ["surrogate behind point 1", "surrogate behind point 2"]
    assert [solution.value for solution in run.acquisitions] == [1, 2]


def test_failed_trial_point_carries_the_partial_run(illustrative_problem):
    """When the trial-point function raises, the error carries the run so far."""
    # A trial-point function that fails once 3 points have been evaluated
    problem = illustrative_problem
    rng = np.random.default_rng(0)

    def trial_point(problem, dataset, rng):
        if len(dataset.u) == 3:
            msg = "no acquisition solution"
            raise NoSolutionError(msg)
        return np.array([1.0]), "surrogate", AcquisitionSolution(np.array([1.0]), 0.0)

    with pytest.raises(NoSolutionError) as info:
        optimization_loop(
            problem,
            illustrative_measure(problem),
            np.array([[0.5], [-0.5]]),
            5,
            trial_point,
            rng,
            "test",
            settings=None,
            scenarios=None,
        )

    # The partial run: 2 initial points and the one trial point before the failure
    run = info.value.run
    assert run is not None
    assert run.dataset.u.shape == (3, 1)
    assert run.dataset.n_initial == 2
    assert run.surrogates == ["surrogate"]
    assert len(run.acquisitions) == 1
    assert info.value.dataset is run.dataset


def test_the_loop_keeps_its_scenarios_in_a_complete_and_a_partial_run(
    illustrative_problem,
):
    """The scenarios given to the loop are in the run it returns, and in the partial
    run of an error."""
    # A trial-point function that fails once 2 points have been evaluated, so 1
    # iteration completes and a second fails
    problem = illustrative_problem
    rng = np.random.default_rng(0)
    scenarios = nominal_scenario(1)

    def trial_point(problem, dataset, rng):
        """A fixed trial point, or an error from the second iteration on."""
        if len(dataset.u) == 2:
            msg = "no acquisition solution"
            raise NoSolutionError(msg)
        return np.array([1.0]), "surrogate", AcquisitionSolution(np.array([1.0]), 0.0)

    def loop(n_iterations):
        """The loop from one initial point."""
        return optimization_loop(
            problem,
            illustrative_measure(problem),
            np.array([[0.5]]),
            n_iterations,
            trial_point,
            rng,
            "test",
            settings=None,
            scenarios=scenarios,
        )

    # One iteration completes the run, and a second fails
    complete = loop(1)
    with pytest.raises(NoSolutionError) as info:
        loop(2)

    # Both runs hold the scenarios
    partial = info.value.run
    assert complete.scenarios is scenarios
    assert partial is not None
    assert partial.scenarios is scenarios


def test_failed_initial_point_carries_the_points_before_it(illustrative_problem):
    """When an initial point fails, the partial run holds the initial points before
    it."""
    # A `measure` that returns NaN for u >= 1, the second initial point
    problem = illustrative_problem
    rng = np.random.default_rng(0)

    def measure(u, rng):
        return np.sin(u) if u[0] < 1.0 else np.array([np.nan])

    def trial_point(problem, dataset, rng):
        return np.array([0.0]), "surrogate", AcquisitionSolution(np.array([0.0]), 0.0)

    with pytest.raises(RootFindingError) as info:
        optimization_loop(
            problem,
            measure,
            np.array([[0.5], [1.5]]),
            2,
            trial_point,
            rng,
            "test",
            settings=None,
            scenarios=None,
        )

    run = info.value.run
    assert run is not None
    assert run.dataset.u.shape == (1, 1)
    assert run.dataset.n_initial == 1
    assert run.surrogates == []
