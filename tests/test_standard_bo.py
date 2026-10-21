"""Standard BO, the optimization loop and the analytic acquisition functions.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

import io
import pickle

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
    StandardBOSettings,
    incumbents,
    initial_points,
    oracle_measure,
    standard_bo,
)
from hybrid_bo.acquisition import (
    AcquisitionSolution,
    _pack,
    standard_acquisition,
)
from hybrid_bo.bo import optimization_loop
from hybrid_bo.gp import FittedGP, fit_gp, predict
from hybrid_bo.run import run_record


def illustrative_measure(problem):
    """The oracle simulation of the illustrative example, without noise."""
    return oracle_measure(problem.system, oracle=lambda v: [cas.sin(v.u)])


def illustrative_run(problem, settings=None) -> Run:
    """A standard BO run on the illustrative example: 2 initial points, 2 iterations,
    with the default settings unless `settings` is given."""
    settings = StandardBOSettings(gp=ex.gp) if settings is None else settings
    rng = np.random.default_rng(0)
    u = initial_points(problem.system.u, 2, rng)
    measure = illustrative_measure(problem)
    return standard_bo(problem, measure, u, 2, rng, settings=settings)


def fitted_illustrative_gp():
    """The default GP and its fit to four points of u -> f."""
    u = np.array([[-1.5], [-0.5], [0.4], [1.7]])
    f = np.array([0.3, -1.2, 2.0, 0.8])
    return fit_gp(ex.gp, u, f, np.random.default_rng(0))


def acquisition_values(nlp, u):
    """The objective of the acquisition problem at each row of u (n, N), shape (n,)."""
    return np.array([nlp.f(_pack(point)).full().item() for point in u])


def loop_record(problem, rng):
    """A record for runs of the optimization loop with a test trial-point function."""
    return run_record(problem, None, rng, "test", "")


class _DataOnlyUnpickler(pickle.Unpickler):
    """Loads only the library's data classes, numpy and builtins: no user code."""

    allowed = (
        "hybrid_bo.run",
        "hybrid_bo.history",
        "hybrid_bo.gp",
        "hybrid_bo.acquisition",
        "hybrid_bo.settings",
        "hybrid_bo.solvers",
        "numpy",
        "builtins",
    )

    def find_class(self, module, name):
        """The class `module.name`, if the module is allowed."""
        if not module.startswith(self.allowed):
            msg = f"the pickle needs {module}.{name}"
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)


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

    nlp = standard_acquisition(illustrative_problem, fitted, 0.0, LCB(2.0)).nlp

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
        LCB(std_factor)


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
    assert run.history.n_initial == 2
    assert run.history.u.shape == (4, 1)
    assert np.all((-2.0 <= run.history.u) & (run.history.u <= 2.0))

    # The surrogates: one per iteration, fitted on the points before its trial point
    assert len(run.surrogates) == 2
    assert all(isinstance(fitted, FittedGP) for fitted in run.surrogates)
    assert [len(fitted.X) for fitted in run.surrogates] == [2, 3]


def test_run_pickles_without_the_code_that_produced_it(illustrative_problem):
    """A run, its fitted GPs included, loads from a pickle without user code."""
    run = illustrative_run(illustrative_problem)

    loaded = _DataOnlyUnpickler(io.BytesIO(pickle.dumps(run))).load()

    assert loaded.record == run.record
    np.testing.assert_array_equal(loaded.history.u, run.history.u)
    np.testing.assert_array_equal(loaded.surrogates[-1].X, run.surrogates[-1].X)


def test_record_holds_the_settings_and_the_gp(illustrative_problem):
    """The record holds the settings, and the prior GP as its repr in an entry of
    its own."""
    settings = StandardBOSettings(gp=ex.gp, acquisition=LCB(2.0))

    record = illustrative_run(illustrative_problem, settings).record

    # The method and its settings, without the GP
    assert record["method"] == "standard_bo"
    assert "n_iterations=2" in record["settings"]
    assert "acquisition=LCB(std_factor=2.0)" in record["settings"]
    assert "n_starts=10" in record["settings"]
    assert "GP(" not in record["settings"]

    # The GP: its repr, which names its parts and their Params
    assert record["gp"] == repr(ex.gp)
    assert "Matern52(lengthscale=Param(init=1.0" in record["gp"]


def test_settings_are_required_and_keyword_only(illustrative_problem):
    """Standard BO has no default settings, since its GP is stated by the
    experiment, and the settings are keyword-only, so they can't be passed in
    `rng`'s place."""
    measure = illustrative_measure(illustrative_problem)
    u = np.array([[-0.5], [0.5]])

    with pytest.raises(TypeError, match="gp"):
        StandardBOSettings()  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="settings"):
        standard_bo(illustrative_problem, measure, u, 0, np.random.default_rng(0))  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        standard_bo(  # ty: ignore[missing-argument]
            illustrative_problem,
            measure,
            u,
            0,
            np.random.default_rng(0),
            StandardBOSettings(gp=ex.gp),  # ty: ignore[too-many-positional-arguments]
        )


def test_failed_evaluation_carries_the_partial_run(illustrative_problem):
    """When evaluating a trial point fails, the error carries the run so far."""
    # A `measure` that returns NaN at its third evaluation (the first trial point), so
    # the root-finding problem has no root
    calls = []

    def measure(u, rng):
        calls.append(u)
        return np.sin(u) if len(calls) < 3 else np.array([np.nan])

    # `as info` keeps the raised exception, as `info.value`, for the checks below
    with pytest.raises(RootFindingError) as info:
        standard_bo(
            illustrative_problem,
            measure,
            np.array([[-0.5], [0.5]]),
            2,
            np.random.default_rng(0),
            settings=StandardBOSettings(gp=ex.gp),
        )

    # The partial run: the initial points, and no surrogate (the surrogate behind the
    # failed trial point is dropped with it); the error's history is the run's
    run = info.value.run
    assert isinstance(run, Run)
    assert run.history.u.shape == (2, 1)
    assert run.history.n_initial == 2
    assert run.surrogates == []
    assert run.record["method"] == "standard_bo"
    assert info.value.history is run.history


def test_stored_acquisition_value_is_the_objective_at_the_trial_point(
    illustrative_problem,
):
    """Each iteration stores the acquisition problem's objective (-EI) at its trial
    point; standard BO has no scenario states."""
    run = illustrative_run(illustrative_problem)

    history = run.history
    assert len(run.acquisitions) == 2
    for i, (solution, fitted) in enumerate(
        zip(run.acquisitions, run.surrogates, strict=True)
    ):
        point = history.n_initial + i
        incumbent = float(incumbents(history)[point - 1])
        nlp = standard_acquisition(illustrative_problem, fitted, incumbent, EI()).nlp
        expected = acquisition_values(nlp, history.u[point : point + 1])[0]
        assert solution.value == pytest.approx(expected, abs=1e-12)
        assert solution.x is None and solution.y is None


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

    def trial_point(problem, history, rng):
        n = len(history.u)
        return next(trial_points), f"surrogate behind point {n}", AcquisitionSolution(n)

    run = optimization_loop(
        problem,
        illustrative_measure(problem),
        np.array([[0.5]]),
        2,
        trial_point,
        rng,
        loop_record(problem, rng),
        settings=None,
    )

    # Every point is evaluated (y = sin(u) is the oracle), in order
    np.testing.assert_array_equal(run.history.u, [[0.5], [1.0], [-1.0]])
    np.testing.assert_allclose(run.history.y, np.sin(run.history.u), atol=1e-8)
    assert run.history.n_initial == 1
    assert run.history.f_oracle is not None
    assert run.surrogates == ["surrogate behind point 1", "surrogate behind point 2"]
    assert [solution.value for solution in run.acquisitions] == [1, 2]


def test_failed_trial_point_carries_the_partial_run(illustrative_problem):
    """When the trial-point function raises, the error carries the run so far."""
    # A trial-point function that fails once 3 points have been evaluated
    problem = illustrative_problem
    rng = np.random.default_rng(0)

    def trial_point(problem, history, rng):
        if len(history.u) == 3:
            msg = "no acquisition solution"
            raise NoSolutionError(msg)
        return np.array([1.0]), "surrogate", AcquisitionSolution(0.0)

    with pytest.raises(NoSolutionError) as info:
        optimization_loop(
            problem,
            illustrative_measure(problem),
            np.array([[0.5], [-0.5]]),
            5,
            trial_point,
            rng,
            loop_record(problem, rng),
            settings=None,
        )

    # The partial run: 2 initial points and the one trial point before the failure
    run = info.value.run
    assert run is not None
    assert run.history.u.shape == (3, 1)
    assert run.history.n_initial == 2
    assert run.surrogates == ["surrogate"]
    assert len(run.acquisitions) == 1
    assert info.value.history is run.history


def test_failed_initial_point_carries_the_points_before_it(illustrative_problem):
    """When an initial point fails, the partial run holds the initial points before
    it."""
    # A `measure` that returns NaN for u >= 1, the second initial point
    problem = illustrative_problem
    rng = np.random.default_rng(0)

    def measure(u, rng):
        return np.sin(u) if u[0] < 1.0 else np.array([np.nan])

    def trial_point(problem, history, rng):
        return np.array([0.0]), "surrogate", AcquisitionSolution(0.0)

    with pytest.raises(RootFindingError) as info:
        optimization_loop(
            problem,
            measure,
            np.array([[0.5], [1.5]]),
            2,
            trial_point,
            rng,
            loop_record(problem, rng),
            settings=None,
        )

    run = info.value.run
    assert run is not None
    assert run.history.u.shape == (1, 1)
    assert run.history.n_initial == 1
    assert run.surrogates == []
