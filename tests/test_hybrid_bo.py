"""Hybrid BO: the scenarios, the hybrid acquisition problem and the method.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

import io
import pickle

import casadi as cas
import numpy as np
import pytest

from hybrid_bo import (
    EI,
    LCB,
    HybridBOSettings,
    Problem,
    RootFindingError,
    Run,
    System,
    Var,
    hybrid_bo,
    incumbents,
    initial_points,
    oracle_measure,
)
from hybrid_bo.acquisition import Scenarios, hybrid_acquisition, saa_scenarios
from hybrid_bo.gp import FittedGP, default_gp, fit_gp, predict

# Three scenarios with unequal weights, so a plain mean would give other values
SCENARIOS = Scenarios(z=np.array([[-1.0], [0.5], [2.0]]), w=np.array([0.2, 0.3, 0.5]))


def problem_with(h_inputs, u=None, x=None):
    """The illustrative system with the given arguments of h (names from "u", "x"),
    bounds and start bounds, and an objective that depends on both states."""
    u = Var("u", -2.0, 2.0) if u is None else u
    x = Var("x", start=(-2.5, 1.0)) if x is None else x
    y = Var("y", start=(-2.0, 2.0))
    system = System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        h_inputs=[{"u": u, "x": x}[name] for name in h_inputs],
        measured=[y],
    )
    return Problem(system, f=lambda v: v.x**2 + v.y)


def fitted_gp(n_inputs):
    """The default GP over n_inputs arguments of h, fitted to four points of y."""
    X = np.array([[-1.5, -1.0], [-0.5, 0.2], [0.4, -0.3], [1.7, 0.6]])[:, :n_inputs]
    y = np.array([0.3, -1.2, 0.9, 0.8])
    gp = default_gp(n_inputs)
    return gp, fit_gp(gp, X, y, np.random.default_rng(0))


def scenario_values(gp, fitted, u, x, h_inputs):
    """Reference values at u (scalar) and the scenarios' x (S,): the GP's scenarios
    y_s = mean + std * z_s at each scenario's arguments of h, and f_s = x_s^2 + y_s."""
    columns = {"u": np.full_like(x, u), "x": x}
    mean, std = predict(gp, fitted, np.column_stack([columns[n] for n in h_inputs]))
    y = mean + std * SCENARIOS.z[:, 0]
    return y, x**2 + y


def variables_at(u, x, y, unknown_states_as_variables):
    """The acquisition problem's variables: u, the x_s, then the y_s if variables."""
    parts = [[u], x, y] if unknown_states_as_variables else [[u], x]
    return np.concatenate(parts)


def illustrative_run(problem, settings, seed=0) -> Run:
    """A hybrid BO run without noise: 2 initial points, 2 iterations."""
    rng = np.random.default_rng(seed)
    u = initial_points(problem.system.u, 2, rng)
    measure = oracle_measure(problem.system, oracle=lambda v: [cas.sin(v.u)])
    return hybrid_bo(problem, measure, u, 2, settings, rng)


# Scenarios


def test_saa_scenarios_are_standard_normal_draws_with_equal_weights():
    """SAA draws z from the generator's standard normal stream, with weights 1/S."""
    scenarios = saa_scenarios(4, 1, np.random.default_rng(0))

    np.testing.assert_array_equal(
        scenarios.z, np.random.default_rng(0).standard_normal((4, 1))
    )
    np.testing.assert_array_equal(scenarios.w, [0.25, 0.25, 0.25, 0.25])


# `parametrize` runs the test once per pair of arguments, each as its own test case
@pytest.mark.parametrize(
    ("z", "w"),
    [
        (np.zeros(2), np.array([0.5, 0.5])),  # z isn't (S, P)
        (np.zeros((2, 1)), np.array([1.0])),  # one weight for two scenarios
        (np.zeros((2, 1)), np.array([0.5, 0.6])),  # the weights don't sum to 1
        (np.zeros((2, 1)), np.array([1.5, -0.5])),  # a negative weight
    ],
)
def test_scenarios_reject_inconsistent_draws_or_weights(z, w):
    """Scenarios need one weight per row of z, nonnegative and summing to 1."""
    # `pytest.raises` passes only if the block raises this exception, with a message
    # matching the regular expression `match`
    with pytest.raises(ValueError, match="Scenario"):
        Scenarios(z, w)


# The hybrid acquisition problem


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
@pytest.mark.parametrize("h_inputs", [["u"], ["x"], ["u", "x"]])
def test_saa_ei_is_the_weighted_average_over_scenarios(
    h_inputs, unknown_states_as_variables
):
    """The EI acquisition problem minimizes the weighted average of
    min(f_s - incumbent, 0), with each scenario's y_s the GP's scenario at that
    scenario's arguments of h."""
    problem = problem_with(h_inputs)
    gp, fitted = fitted_gp(len(h_inputs))
    # f_s is about 0.36, 0.62 and 2.13, so two scenarios improve on the incumbent
    u, x, incumbent = 0.3, np.array([-1.0, 0.0, 0.5]), 1.0

    nlp, _ = hybrid_acquisition(
        problem, gp, fitted, SCENARIOS, incumbent, EI(), unknown_states_as_variables
    )

    y, f = scenario_values(gp, fitted, u, x, h_inputs)
    expected = SCENARIOS.w @ np.minimum(f - incumbent, 0)
    w = variables_at(u, x, y, unknown_states_as_variables)
    np.testing.assert_allclose(nlp.f(w).full().item(), expected, rtol=1e-10)


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_saa_lcb_is_the_weighted_mean_minus_std_factor_times_weighted_std(
    unknown_states_as_variables,
):
    """The LCB acquisition problem minimizes the weighted mean of the f_s minus
    std_factor times their weighted standard deviation."""
    problem = problem_with(["u"])
    gp, fitted = fitted_gp(1)
    u, x = -0.7, np.array([-1.0, 0.0, 0.5])

    nlp, _ = hybrid_acquisition(
        problem, gp, fitted, SCENARIOS, 0.0, LCB(2.0), unknown_states_as_variables
    )

    y, f = scenario_values(gp, fitted, u, x, ["u"])
    mean = SCENARIOS.w @ f
    expected = mean - 2.0 * np.sqrt(SCENARIOS.w @ (f - mean) ** 2)
    w = variables_at(u, x, y, unknown_states_as_variables)
    np.testing.assert_allclose(nlp.f(w).full().item(), expected, rtol=1e-10)


def test_known_equations_hold_per_scenario_and_tie_the_unknown_states_to_the_gp():
    """With the unknown states as variables, the constraints are, per scenario, the
    known equations and y_s minus the GP's scenario."""
    problem = problem_with(["u"])
    gp, fitted = fitted_gp(1)
    u, x = 0.3, np.array([-1.0, 0.0, 0.5])
    y = np.array([0.1, -0.2, 0.4])

    nlp, _ = hybrid_acquisition(problem, gp, fitted, SCENARIOS, 0.0, EI())
    assert nlp.g is not None
    constraints = nlp.g(variables_at(u, x, y, True)).full().ravel()

    # Per scenario: g = x + exp(x) - y, then y minus the GP's scenario
    y_gp, _ = scenario_values(gp, fitted, u, x, ["u"])
    expected = np.column_stack([x + np.exp(x) - y, y - y_gp]).ravel()
    np.testing.assert_allclose(constraints, expected, rtol=1e-10, atol=1e-12)


def test_substituted_unknown_states_leave_only_the_known_equations():
    """With the GP substituted for the unknown states, the constraints are the known
    equations per scenario, with y_s the GP's scenario."""
    problem = problem_with(["u"])
    gp, fitted = fitted_gp(1)
    u, x = 0.3, np.array([-1.0, 0.0, 0.5])

    nlp, _ = hybrid_acquisition(
        problem, gp, fitted, SCENARIOS, 0.0, EI(), unknown_states_as_variables=False
    )
    assert nlp.g is not None
    constraints = nlp.g(variables_at(u, x, None, False)).full().ravel()

    y_gp, _ = scenario_values(gp, fitted, u, x, ["u"])
    np.testing.assert_allclose(constraints, x + np.exp(x) - y_gp, rtol=1e-10)


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_bounds_and_start_bounds_come_from_each_variable(unknown_states_as_variables):
    """Every variable of the acquisition problem is boxed by its own bounds and
    started within its own start bounds, once per scenario for the states."""
    # Start bounds narrower than the bounds, so mixing them up shows
    problem = problem_with(
        ["u"],
        u=Var("u", -2.0, 2.0, start=(-1.0, 1.5)),
        x=Var("x", -3.0, 3.0, start=(-2.5, 1.0)),
    )
    gp, fitted = fitted_gp(1)

    nlp, (lower, upper) = hybrid_acquisition(
        problem, gp, fitted, SCENARIOS, 0.0, EI(), unknown_states_as_variables
    )

    # u, then x three times, then y (unbounded) three times if variables
    n_y = 3 if unknown_states_as_variables else 0
    inf = np.inf
    np.testing.assert_array_equal(nlp.lower, [-2.0] + [-3.0] * 3 + [-inf] * n_y)
    np.testing.assert_array_equal(nlp.upper, [2.0] + [3.0] * 3 + [inf] * n_y)
    np.testing.assert_array_equal(lower, [-1.0] + [-2.5] * 3 + [-2.0] * n_y)
    np.testing.assert_array_equal(upper, [1.5] + [1.0] * 3 + [2.0] * n_y)


def test_acquisition_builder_rejects_anything_but_ei_or_lcb():
    """The hybrid acquisition builder checks its acquisition function as the settings
    do."""
    gp, fitted = fitted_gp(1)

    with pytest.raises(TypeError, match=r"EI\(\) or LCB"):
        hybrid_acquisition(problem_with(["u"]), gp, fitted, SCENARIOS, 0.0, "ei")  # ty: ignore[invalid-argument-type]


def test_scenarios_need_one_column_per_unknown_state():
    """The acquisition builder rejects scenario draws with more columns than unknown
    states."""
    gp, fitted = fitted_gp(1)
    scenarios = Scenarios(np.zeros((3, 2)), np.full(3, 1 / 3))

    with pytest.raises(ValueError, match="one column of z per unknown state"):
        hybrid_acquisition(problem_with(["u"]), gp, fitted, scenarios, 0.0, EI())


@pytest.mark.parametrize("acquisition", ["ei", LCB, None])
def test_settings_reject_anything_but_ei_or_lcb(acquisition):
    """The settings accept only an EI or LCB instance, not a name or the class."""
    with pytest.raises(TypeError, match=r"EI\(\) or LCB"):
        HybridBOSettings(acquisition=acquisition)


# Hybrid BO


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_both_settings_solve_the_illustrative_example(
    illustrative_problem, unknown_states_as_variables
):
    """Hybrid BO runs on the illustrative example either way: one trial point per
    iteration within the bounds, one GP per iteration, and the scenarios it drew."""
    settings = HybridBOSettings(unknown_states_as_variables=unknown_states_as_variables)

    run = illustrative_run(illustrative_problem, settings)

    # The points: 2 initial points, then 2 trial points
    assert run.history.n_initial == 2
    assert run.history.u.shape == (4, 1)
    assert np.all((-2.0 <= run.history.u) & (run.history.u <= 2.0))

    # The surrogates: one per iteration, fitted on the points before its trial point
    assert [len(fitted.X) for fitted in run.surrogates] == [2, 3]
    assert all(isinstance(fitted, FittedGP) for fitted in run.surrogates)

    # The scenarios: drawn after the initial points, which take no stream draws
    rng = np.random.default_rng(0)
    initial_points(illustrative_problem.system.u, 2, rng)
    assert run.scenarios is not None
    np.testing.assert_array_equal(run.scenarios.z, rng.standard_normal((25, 1)))
    np.testing.assert_array_equal(run.scenarios.w, np.full(25, 1 / 25))


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_stored_acquisition_solution_solves_the_acquisition_problem(
    illustrative_problem, unknown_states_as_variables
):
    """Each iteration stores the acquisition problem's objective at its solution, and
    the scenario states there: they satisfy the known equations, and the unknown
    states are the GP's scenarios at the trial point."""
    settings = HybridBOSettings(
        n_scenarios=5, unknown_states_as_variables=unknown_states_as_variables
    )

    run = illustrative_run(illustrative_problem, settings)

    # One solution per iteration, with every scenario's states
    history, gp = run.history, default_gp(1)
    assert run.scenarios is not None
    assert len(run.acquisitions) == 2
    for i, (solution, fitted) in enumerate(
        zip(run.acquisitions, run.surrogates, strict=True)
    ):
        assert solution.x is not None and solution.y is not None
        assert solution.x.shape == (5, 1) and solution.y.shape == (5, 1)

        # The value is the acquisition problem's objective at the stored solution
        point = history.n_initial + i
        u = history.u[point, 0]
        incumbent = float(incumbents(history)[point - 1])
        nlp, _ = hybrid_acquisition(
            illustrative_problem,
            gp,
            fitted,
            run.scenarios,
            incumbent,
            EI(),
            unknown_states_as_variables,
        )
        w = variables_at(
            u, solution.x[:, 0], solution.y[:, 0], unknown_states_as_variables
        )
        assert solution.value == pytest.approx(float(nlp.f(w)), abs=1e-12)

        # The known equations hold, and y_s is the GP's scenario at (u, x_s)
        x_s, y_s = solution.x[:, 0], solution.y[:, 0]
        np.testing.assert_allclose(x_s + np.exp(x_s), y_s, atol=1e-6)
        mean, std = predict(gp, fitted, np.full((5, 1), u))
        np.testing.assert_allclose(y_s, mean + std * run.scenarios.z[:, 0], atol=1e-6)


def test_gp_inputs_are_the_arguments_of_h():
    """The GP is fitted on the columns of `h_inputs`, here u and x, mapping to y; the
    default GP has one lengthscale per argument."""
    problem = problem_with(["u", "x"])

    run = illustrative_run(problem, HybridBOSettings(n_scenarios=5))

    # The fitted GP's inputs, unscaled, are the (u, x) of the points before it
    fitted = run.surrogates[-1]
    inputs = (fitted.X - fitted.x_scaler.intercept) / fitted.x_scaler.slope
    history = run.history
    np.testing.assert_allclose(inputs, np.column_stack([history.u, history.x])[:3])
    assert fitted.hyperparameters["lengthscale"].shape == (2,)


def test_record_holds_the_settings_and_the_gp_over_the_arguments_of_h(
    illustrative_problem,
):
    """The record names the method and its settings, and the GP the run used, over
    the arguments of h."""
    settings = HybridBOSettings(n_scenarios=5, unknown_states_as_variables=False)

    record = illustrative_run(illustrative_problem, settings).record

    assert record["method"] == "hybrid_bo"
    assert "n_scenarios=5" in record["settings"]
    assert "unknown_states_as_variables=False" in record["settings"]
    assert "GP(" not in record["settings"]
    assert "u" in record["kernel"]


def test_run_pickles_without_the_code_that_produced_it(illustrative_problem):
    """A hybrid BO run, its GPs and scenarios included, loads from a pickle without
    user code."""

    class DataOnlyUnpickler(pickle.Unpickler):
        """Loads only the library's data classes, numpy and builtins."""

        def find_class(self, module, name):
            """The class `module.name`, if the module is allowed."""
            allowed = ("hybrid_bo.run", "hybrid_bo.history", "hybrid_bo.gp")
            allowed += ("hybrid_bo.acquisition", "numpy", "builtins")
            if not module.startswith(allowed):
                msg = f"the pickle needs {module}.{name}"
                raise pickle.UnpicklingError(msg)
            return super().find_class(module, name)

    run = illustrative_run(illustrative_problem, HybridBOSettings(n_scenarios=5))

    loaded = DataOnlyUnpickler(io.BytesIO(pickle.dumps(run))).load()

    assert loaded.scenarios is not None and run.scenarios is not None
    np.testing.assert_array_equal(loaded.scenarios.z, run.scenarios.z)
    np.testing.assert_array_equal(loaded.surrogates[-1].X, run.surrogates[-1].X)


def test_failed_evaluation_carries_the_partial_run_with_its_scenarios(
    illustrative_problem,
):
    """When evaluating a trial point fails, the error carries the run so far, with
    the scenarios."""
    # A `measure` that returns NaN at its third evaluation (the first trial point), so
    # the root-finding problem has no root
    calls = []

    def measure(u, rng):
        """y = sin(u) for the first two calls, then NaN."""
        calls.append(u)
        return np.sin(u) if len(calls) < 3 else np.array([np.nan])

    # `as info` keeps the raised exception, as `info.value`, for the checks below
    with pytest.raises(RootFindingError) as info:
        hybrid_bo(
            illustrative_problem,
            measure,
            np.array([[-0.5], [0.5]]),
            2,
            HybridBOSettings(n_scenarios=5),
            np.random.default_rng(0),
        )

    run = info.value.run
    assert run is not None
    assert run.history.u.shape == (2, 1)
    assert run.surrogates == []
    assert run.scenarios is not None and run.scenarios.z.shape == (5, 1)
    assert run.record["method"] == "hybrid_bo"
