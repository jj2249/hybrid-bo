"""Hybrid BO: the scenarios, the hybrid acquisition problem and the method.

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
import scipy.optimize
import scipy.special
import scipy.stats

from hybrid_bo import (
    EI,
    LCB,
    SAA,
    HybridBOSettings,
    Linearization,
    Problem,
    RootFindingError,
    Run,
    SoftSAA,
    System,
    Var,
    hybrid_bo,
    incumbents,
    initial_points,
    oracle_measure,
)
from hybrid_bo.acquisition import (
    Scenarios,
    _pack,
    _sensitivity,
    _unpack,
    hybrid_acquisition,
    nominal_scenario,
    saa_scenarios,
)
from hybrid_bo.evaluation import evaluate
from hybrid_bo.gp import FittedGP, Matern52, Scale, fit_gp, predict

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
    return fit_gp(ex.gp, X, y, np.random.default_rng(0))


def scenario_values(fitted, u, x, h_inputs):
    """Reference values at u (scalar) and the scenarios' x (S,): the GP's scenarios
    y_s = mean + std * z_s at each scenario's arguments of h, and f_s = x_s^2 + y_s."""
    columns = {"u": np.full_like(x, u), "x": x}
    mean, std = predict(fitted, np.column_stack([columns[n] for n in h_inputs]))
    y = mean + std * SCENARIOS.z[:, 0]
    return y, x**2 + y


def variables_at(u, x, y, unknown_states_as_variables):
    """The acquisition problem's decision vector at u (scalar) and the scenarios'
    x and y (S,), with y only if the unknown states are variables."""
    y = y[:, np.newaxis] if unknown_states_as_variables else None
    return _pack(np.array([u]), x[:, np.newaxis], y)


def illustrative_run(problem, settings, seed=0) -> Run:
    """A hybrid BO run without noise: 2 initial points, 2 iterations."""
    rng = np.random.default_rng(seed)
    u = initial_points(problem.system.u, 2, rng)
    measure = oracle_measure(problem.system, oracle=lambda v: [cas.sin(v.u)])
    return hybrid_bo(problem, measure, u, 2, rng, settings=settings)


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
    fitted = fitted_gp(len(h_inputs))
    # f_s is about 0.36, 0.62 and 2.13, so two scenarios improve on the incumbent
    u, x, incumbent = 0.3, np.array([-1.0, 0.0, 0.5]), 1.0

    nlp = hybrid_acquisition(
        problem,
        fitted,
        SCENARIOS,
        incumbent,
        EI(),
        SAA(unknown_states_as_variables=unknown_states_as_variables),
    ).nlp

    y, f = scenario_values(fitted, u, x, h_inputs)
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
    fitted = fitted_gp(1)
    u, x = -0.7, np.array([-1.0, 0.0, 0.5])

    nlp = hybrid_acquisition(
        problem,
        fitted,
        SCENARIOS,
        0.0,
        LCB(2.0),
        SAA(unknown_states_as_variables=unknown_states_as_variables),
    ).nlp

    y, f = scenario_values(fitted, u, x, ["u"])
    mean = SCENARIOS.w @ f
    expected = mean - 2.0 * np.sqrt(SCENARIOS.w @ (f - mean) ** 2)
    w = variables_at(u, x, y, unknown_states_as_variables)
    np.testing.assert_allclose(nlp.f(w).full().item(), expected, rtol=1e-10)


def test_known_equations_hold_per_scenario_and_tie_the_unknown_states_to_the_gp():
    """With the unknown states as variables, the constraints are, per scenario, the
    known equations and y_s minus the GP's scenario."""
    problem = problem_with(["u"])
    fitted = fitted_gp(1)
    u, x = 0.3, np.array([-1.0, 0.0, 0.5])
    y = np.array([0.1, -0.2, 0.4])

    nlp = hybrid_acquisition(problem, fitted, SCENARIOS, 0.0, EI()).nlp
    assert nlp.g is not None
    constraints = nlp.g(variables_at(u, x, y, True)).full().ravel()

    # Per scenario: g = x + exp(x) - y, then y minus the GP's scenario
    y_gp, _ = scenario_values(fitted, u, x, ["u"])
    expected = np.column_stack([x + np.exp(x) - y, y - y_gp]).ravel()
    np.testing.assert_allclose(constraints, expected, rtol=1e-10, atol=1e-12)


def test_substituted_unknown_states_leave_only_the_known_equations():
    """With the GP substituted for the unknown states, the constraints are the known
    equations per scenario, with y_s the GP's scenario."""
    problem = problem_with(["u"])
    fitted = fitted_gp(1)
    u, x = 0.3, np.array([-1.0, 0.0, 0.5])

    nlp = hybrid_acquisition(
        problem,
        fitted,
        SCENARIOS,
        0.0,
        EI(),
        SAA(unknown_states_as_variables=False),
    ).nlp
    assert nlp.g is not None
    constraints = nlp.g(variables_at(u, x, None, False)).full().ravel()

    y_gp, _ = scenario_values(fitted, u, x, ["u"])
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
    fitted = fitted_gp(1)

    acquisition_problem = hybrid_acquisition(
        problem,
        fitted,
        SCENARIOS,
        0.0,
        EI(),
        SAA(unknown_states_as_variables=unknown_states_as_variables),
    )
    nlp, (lower, upper) = acquisition_problem.nlp, acquisition_problem.start_bounds

    # u, then x three times, then y (unbounded) three times if variables
    n_y = 3 if unknown_states_as_variables else 0
    inf = np.inf
    np.testing.assert_array_equal(nlp.lower, [-2.0] + [-3.0] * 3 + [-inf] * n_y)
    np.testing.assert_array_equal(nlp.upper, [2.0] + [3.0] * 3 + [inf] * n_y)
    np.testing.assert_array_equal(lower, [-1.0] + [-2.5] * 3 + [-2.0] * n_y)
    np.testing.assert_array_equal(upper, [1.5] + [1.0] * 3 + [2.0] * n_y)


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_unpack_undoes_pack(unknown_states_as_variables):
    """Unpacking a packed decision vector of an acquisition problem with two decision
    variables and two mechanistic states gives back u, x and y (None when the unknown
    states are substituted)."""
    # A system with two of each, and its acquisition problem
    u1, u2 = Var("u1", -2.0, 2.0), Var("u2", -2.0, 2.0)
    x1, x2 = Var("x1", start=(-2.0, 2.0)), Var("x2", start=(-2.0, 2.0))
    y = Var("y", start=(-2.0, 2.0))
    system = System(
        u=[u1, u2],
        x=[x1, x2],
        y=[y],
        g=lambda v: [v.x1 + cas.exp(v.x1) - v.y, v.x2 - v.u2],
        h_inputs=[u1],
        measured=[y],
    )
    problem = Problem(system, f=lambda v: v.x1**2 + v.x2 + v.y)
    fitted = fitted_gp(1)
    acquisition_problem = hybrid_acquisition(
        problem,
        fitted,
        SCENARIOS,
        0.0,
        EI(),
        SAA(unknown_states_as_variables=unknown_states_as_variables),
    )

    # Distinct values for every entry, packed then unpacked
    u = np.array([1.0, 2.0])
    x = np.arange(6.0).reshape(3, 2) + 10
    y = np.arange(3.0).reshape(3, 1) + 20 if unknown_states_as_variables else None
    w = _pack(u, x, y)
    u_back, x_back, y_back = _unpack(acquisition_problem, w)

    # One entry of w per variable of the problem, and the same values back
    assert w.shape == acquisition_problem.nlp.lower.shape
    np.testing.assert_array_equal(u_back, u)
    np.testing.assert_array_equal(x_back, x)
    if y is None:
        assert y_back is None
    else:
        np.testing.assert_array_equal(y_back, y)


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_unpack_rejects_a_decision_vector_of_the_wrong_length(
    unknown_states_as_variables,
):
    """A decision vector one entry too long or too short doesn't fit the problem."""
    fitted = fitted_gp(1)
    acquisition_problem = hybrid_acquisition(
        problem_with(["u"]),
        fitted,
        SCENARIOS,
        0.0,
        EI(),
        SAA(unknown_states_as_variables=unknown_states_as_variables),
    )
    w = np.zeros(len(acquisition_problem.nlp.lower))

    for wrong in (w[:-1], np.append(w, 0.0)):
        with pytest.raises(ValueError, match="doesn't fit"):
            _unpack(acquisition_problem, wrong)


def test_acquisition_builder_rejects_anything_but_ei_or_lcb():
    """The hybrid acquisition builder checks its acquisition function as the settings
    do."""
    fitted = fitted_gp(1)

    with pytest.raises(TypeError, match=r"EI\(\) or LCB"):
        hybrid_acquisition(problem_with(["u"]), fitted, SCENARIOS, 0.0, "ei")  # ty: ignore[invalid-argument-type]


def test_scenarios_need_one_column_per_unknown_state():
    """The acquisition builder rejects scenario draws with more columns than unknown
    states."""
    fitted = fitted_gp(1)
    scenarios = Scenarios(np.zeros((3, 2)), np.full(3, 1 / 3))

    with pytest.raises(ValueError, match="one column of z per unknown state"):
        hybrid_acquisition(problem_with(["u"]), fitted, scenarios, 0.0, EI())


@pytest.mark.parametrize("acquisition", ["ei", LCB, None])
def test_settings_reject_anything_but_ei_or_lcb(acquisition):
    """The settings accept only an EI or LCB instance, not a name or the class."""
    with pytest.raises(TypeError, match=r"EI\(\) or LCB"):
        HybridBOSettings(gp=ex.gp, acquisition=acquisition)


def test_settings_reject_an_unknown_approximation():
    """The settings accept only an approximation the library knows, such as SAA()."""
    with pytest.raises(TypeError, match=r"SAA\("):
        HybridBOSettings(gp=ex.gp, approximation=object())  # ty: ignore[invalid-argument-type]


def test_acquisition_builder_rejects_an_unknown_approximation():
    """The hybrid acquisition builder checks its approximation as the settings do."""
    fitted = fitted_gp(1)

    with pytest.raises(TypeError, match=r"SAA\("):
        hybrid_acquisition(
            problem_with(["u"]),
            fitted,
            SCENARIOS,
            0.0,
            EI(),
            object(),  # ty: ignore[invalid-argument-type]
        )


def test_lcb_with_linearization_is_not_implemented():
    """Linearization gives EI only, so the settings and the acquisition builder
    reject it with LCB, and name the combinations that work."""
    fitted = fitted_gp(1)

    with pytest.raises(NotImplementedError, match=r"EI\(\) with Linearization"):
        HybridBOSettings(gp=ex.gp, acquisition=LCB(), approximation=Linearization())
    with pytest.raises(NotImplementedError, match=r"EI\(\) with Linearization"):
        hybrid_acquisition(
            problem_with(["u"]),
            fitted,
            nominal_scenario(1),
            0.0,
            LCB(),
            Linearization(),
        )


def test_linearization_needs_the_nominal_scenario():
    """Linearization's acquisition problem is built over the nominal scenario alone,
    so drawn scenarios are rejected."""
    fitted = fitted_gp(1)

    with pytest.raises(ValueError, match="nominal scenario"):
        hybrid_acquisition(
            problem_with(["u"]), fitted, SCENARIOS, 0.0, EI(), Linearization()
        )


def linear_problem():
    """A problem whose linearization is exact: g = x - 2y + u and f = 3x + y are
    linear, and h takes u only, so f = 7y - 3u is normal under the GP."""
    u = Var("u", -2.0, 2.0)
    x = Var("x", start=(-5.0, 5.0))
    y = Var("y", start=(-3.0, 3.0))
    system = System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x - 2 * v.y + v.u],
        h_inputs=[u],
        measured=[y],
    )
    return Problem(system, f=lambda v: 3 * v.x + v.y)


def linear_normal(fitted, u):
    """At u (scalar), the GP's mean and std of y, and the normal of the linear
    problem's f = 7y - 3u: its mean and std."""
    mean, std = predict(fitted, np.array([[u]]))
    return mean[0], std[0], 7 * mean[0] - 3 * u, 7 * std[0]


def closed_form_ei(mean, std, incumbent):
    """EI below the incumbent of a normal with this mean and std (scipy)."""
    gamma = (incumbent - mean) / std
    cdf, pdf = scipy.stats.norm.cdf(gamma), scipy.stats.norm.pdf(gamma)
    return (incumbent - mean) * cdf + std * pdf


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_lin_ei_is_exact_for_a_linear_problem(unknown_states_as_variables):
    """With linear f and g and h of u only, f is normal, N(7 mean - 3u, (7 std)^2), so
    Lin-EI at the nominal states is that normal's closed-form EI."""
    # The normal of f at u, and an incumbent half a std below its mean
    problem, fitted, u = linear_problem(), fitted_gp(1), 0.3
    y_bar, _, f_mean, f_std = linear_normal(fitted, u)
    incumbent = f_mean - 0.5 * f_std

    # Lin-EI's objective at the nominal states, x = 2y - u
    approximation = Linearization(unknown_states_as_variables)
    nlp = hybrid_acquisition(
        problem, fitted, nominal_scenario(1), incumbent, EI(), approximation
    ).nlp
    y = np.array([y_bar])
    w = variables_at(u, 2 * y - u, y, unknown_states_as_variables)

    expected = -closed_form_ei(f_mean, f_std, incumbent)
    np.testing.assert_allclose(nlp.f(w).full().item(), expected, rtol=1e-8)


def test_saa_ei_agrees_with_lin_ei_for_a_linear_problem():
    """For the linear problem, SAA-EI over many scenarios agrees with the exact EI
    that Lin-EI gives, within 4 Monte Carlo standard errors."""
    # The normal of f at u, and 500 scenarios with each x_s solving g
    problem, fitted, u = linear_problem(), fitted_gp(1), 0.3
    y_mean, y_std, f_mean, f_std = linear_normal(fitted, u)
    incumbent = f_mean - 0.5 * f_std
    scenarios = saa_scenarios(500, 1, np.random.default_rng(0))
    y_s = y_mean + y_std * scenarios.z[:, 0]

    # SAA-EI at those scenario states
    nlp = hybrid_acquisition(
        problem, fitted, scenarios, incumbent, EI(), SAA(500, False)
    ).nlp
    saa_ei = -nlp.f(variables_at(u, 2 * y_s - u, None, False)).full().item()

    improvement = np.maximum(incumbent - (7 * y_s - 3 * u), 0)
    error = improvement.std() / np.sqrt(500)
    assert abs(saa_ei - closed_form_ei(f_mean, f_std, incumbent)) < 4 * error


@pytest.mark.parametrize("h_inputs", [["u"], ["x"], ["u", "x"]])
def test_sensitivity_is_the_total_derivative_of_f_in_the_draws(h_inputs):
    """df/dz at the nominal states matches central finite differences of a
    scenario's f_s in its draw z, where x_s solves the known equation with y_s the
    GP's scenario at its arguments of h, so x moves with y."""
    problem, fitted, u = problem_with(h_inputs), fitted_gp(len(h_inputs)), 0.3

    def scenario(z):
        """The scenario states (x_s, y_s) of draw z: x_s solves x + exp(x) = y_s."""

        def y_of(x):
            """The GP's scenario mean + std * z at (u, x)."""
            columns = {"u": u, "x": x}
            inputs = np.array([[columns[name] for name in h_inputs]])
            mean, std = predict(fitted, inputs)
            return mean[0] + std[0] * z

        x = scipy.optimize.brentq(lambda x: x + np.exp(x) - y_of(x), -10.0, 5.0)
        return x, y_of(x)

    # The sensitivity at the nominal states (z = 0)
    x_bar, y_bar = scenario(0.0)
    a = _sensitivity(problem, fitted)(x_bar, y_bar, u).full().item()

    # Central differences of f_s = x_s^2 + y_s in z
    f = [x**2 + y for x, y in (scenario(1e-5), scenario(-1e-5))]
    assert a == pytest.approx((f[0] - f[1]) / 2e-5, rel=1e-6)


# The soft sample-average approximation


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_soft_saa_needs_a_positive_finite_temperature(temperature):
    """The temperature is in units of the objective, so it must be positive and
    finite."""
    with pytest.raises(ValueError, match="temperature"):
        SoftSAA(temperature)


def test_lcb_with_soft_saa_is_rejected():
    """SoftSAA softens EI's improvement, which LCB doesn't have, so the settings and
    the acquisition builder reject it with LCB."""
    fitted = fitted_gp(1)

    # The settings, then the acquisition builder
    with pytest.raises(ValueError, match=r"EI\(\) with SoftSAA"):
        HybridBOSettings(gp=ex.gp, acquisition=LCB(), approximation=SoftSAA(1.0))
    with pytest.raises(ValueError, match=r"EI\(\) with SoftSAA"):
        hybrid_acquisition(
            problem_with(["u"]), fitted, SCENARIOS, 0.0, LCB(), SoftSAA(1.0)
        )


def ei_problems(incumbent, temperature, unknown_states_as_variables):
    """The SoftSAA and the SAA EI acquisition problems of the illustrative system with
    h of u, over SCENARIOS, and the decision vector and f_s (S,) at u = 0.3 and x =
    (-1, 0, 0.5)."""
    problem = problem_with(["u"])
    fitted = fitted_gp(1)
    u, x = 0.3, np.array([-1.0, 0.0, 0.5])

    def build(approximation):
        """The EI acquisition problem with this approximation."""
        return hybrid_acquisition(
            problem, fitted, SCENARIOS, incumbent, EI(), approximation
        )

    y, f = scenario_values(fitted, u, x, ["u"])
    soft = build(SoftSAA(temperature, 3, unknown_states_as_variables))
    saa = build(SAA(3, unknown_states_as_variables))
    return soft, saa, variables_at(u, x, y, unknown_states_as_variables), f


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_soft_saa_ei_is_minus_log_the_weighted_smooth_improvement(
    unknown_states_as_variables,
):
    """The SoftSAA acquisition problem minimizes -log sum_s w_s tau log(1 + exp(a_s /
    tau)), with a_s = incumbent - f_s."""
    # Setup: f_s is about 0.36, 0.62 and 2.13, so the incumbent sits among them
    temperature, incumbent = 0.5, 1.0
    soft, _, w, f = ei_problems(incumbent, temperature, unknown_states_as_variables)

    # Check: the objective against numpy's
    a = incumbent - f
    smooth = temperature * np.log1p(np.exp(a / temperature))
    expected = -np.log(SCENARIOS.w @ smooth)
    np.testing.assert_allclose(soft.nlp.f(w).full().item(), expected, rtol=1e-10)


def test_soft_saa_ei_is_saa_ei_where_every_scenario_improves_by_far():
    """With the incumbent far above every f_s compared with the temperature, the
    smooth maximum is the maximum, so exp(-objective) is SAA-EI, which is positive
    there."""
    # Setup: the incumbent 200 tau above the largest f_s at the point
    temperature = 0.01
    *_, f = ei_problems(0.0, temperature, False)
    soft, saa, w, _ = ei_problems(f.max() + 200 * temperature, temperature, False)

    # Check: SAA-EI is positive, and exp(-objective) is it
    saa_ei = -saa.nlp.f(w).full().item()
    assert saa_ei > 0
    np.testing.assert_allclose(np.exp(-soft.nlp.f(w).full().item()), saa_ei, rtol=1e-12)


def gradient(nlp, w):
    """The gradient of the acquisition problem's objective at w, (n,)."""
    symbol = cas.SX.sym("w", len(w))
    function = cas.Function("gradient", [symbol], [cas.gradient(nlp.f(symbol), symbol)])
    return function(w).full().ravel()


def test_soft_saa_ei_has_a_gradient_where_no_scenario_improves():
    """With the incumbent below every f_s, SAA-EI is flat (zero, with zero gradient),
    while SoftSAA-EI's objective stays finite with a finite, nonzero gradient in u."""
    # Setup: f_s is at least about 0.36, so no scenario improves on -1
    soft, saa, w, f = ei_problems(-1.0, 0.5, False)
    assert np.all(f > -1.0)

    # SAA-EI is flat there
    assert saa.nlp.f(w).full().item() == 0
    assert np.all(gradient(saa.nlp, w) == 0)

    # SoftSAA-EI's objective and its gradient in u
    assert np.isfinite(soft.nlp.f(w).full().item())
    u_gradient, _, _ = _unpack(soft, gradient(soft.nlp, w))
    assert np.all(np.isfinite(u_gradient)) and np.all(u_gradient != 0)


@pytest.mark.parametrize("incumbent", [-3.0, 5.0])
def test_soft_saa_ei_is_finite_far_from_the_incumbent(incumbent):
    """At a_s / tau beyond -1000 or 1000, where exp(a_s / tau) underflows or
    overflows, the objective and its gradient are finite; far below, the objective
    matches the asymptote -log sum_s w_s tau exp(a_s / tau)."""
    # Setup: a temperature small enough to put every a_s / tau beyond +-1000
    soft, _, w, f = ei_problems(incumbent, 0.002, False)
    a = (incumbent - f) / 0.002
    assert np.all(np.abs(a) > 1000)

    # Check: the objective and its gradient are finite, and match the asymptote below
    value = soft.nlp.f(w).full().item()
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient(soft.nlp, w)))
    if incumbent < 0:
        expected = -(np.log(0.002) + scipy.special.logsumexp(a, b=SCENARIOS.w))
        np.testing.assert_allclose(value, expected, rtol=1e-12)


# Hybrid BO


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_both_settings_solve_the_illustrative_example(
    illustrative_problem, unknown_states_as_variables
):
    """Hybrid BO runs on the illustrative example either way: one trial point per
    iteration within the bounds, one GP per iteration, and the scenarios it drew."""
    approximation = SAA(unknown_states_as_variables=unknown_states_as_variables)
    settings = HybridBOSettings(gp=ex.gp, approximation=approximation)

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
def test_soft_saa_solves_the_illustrative_example(
    illustrative_problem, unknown_states_as_variables
):
    """Hybrid BO runs with SoftSAA either way: trial points within the bounds, and
    scenarios drawn as SAA draws them; the stored acquisition value is the minimized
    -log SoftSAA-EI, finite."""
    approximation = SoftSAA(
        1.0, unknown_states_as_variables=unknown_states_as_variables
    )
    settings = HybridBOSettings(gp=ex.gp, approximation=approximation)

    run = illustrative_run(illustrative_problem, settings)

    # The points and the acquisition values
    assert run.history.u.shape == (4, 1)
    assert np.all((-2.0 <= run.history.u) & (run.history.u <= 2.0))
    assert all(np.isfinite(solution.value) for solution in run.acquisitions)

    # The scenarios: the same draws as SAA's
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
    approximation = SAA(
        n_scenarios=5, unknown_states_as_variables=unknown_states_as_variables
    )
    settings = HybridBOSettings(gp=ex.gp, approximation=approximation)

    run = illustrative_run(illustrative_problem, settings)

    # One solution per iteration, with every scenario's states
    history = run.history
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
        nlp = hybrid_acquisition(
            illustrative_problem,
            fitted,
            run.scenarios,
            incumbent,
            EI(),
            SAA(unknown_states_as_variables=unknown_states_as_variables),
        ).nlp
        w = variables_at(
            u, solution.x[:, 0], solution.y[:, 0], unknown_states_as_variables
        )
        assert solution.value == pytest.approx(float(nlp.f(w)), abs=1e-12)

        # The known equations hold, and y_s is the GP's scenario at (u, x_s)
        x_s, y_s = solution.x[:, 0], solution.y[:, 0]
        np.testing.assert_allclose(x_s + np.exp(x_s), y_s, atol=1e-6)
        mean, std = predict(fitted, np.full((5, 1), u))
        np.testing.assert_allclose(y_s, mean + std * run.scenarios.z[:, 0], atol=1e-6)


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_linearization_runs_over_the_nominal_scenario(
    illustrative_problem, unknown_states_as_variables
):
    """Hybrid BO with linearization holds the nominal scenario, and each iteration
    stores the nominal states at its trial point, where x solves the known equation
    with y the GP's mean, and the acquisition problem's objective there."""
    approximation = Linearization(unknown_states_as_variables)
    settings = HybridBOSettings(gp=ex.gp, approximation=approximation)

    run = illustrative_run(illustrative_problem, settings)

    # The nominal scenario: zero draws, weight 1
    history = run.history
    assert run.scenarios is not None
    np.testing.assert_array_equal(run.scenarios.z, np.zeros((1, 1)))
    np.testing.assert_array_equal(run.scenarios.w, np.ones(1))

    # Per iteration: the nominal states, and the value at them
    for i, (solution, fitted) in enumerate(
        zip(run.acquisitions, run.surrogates, strict=True)
    ):
        assert solution.x is not None and solution.y is not None
        assert solution.x.shape == (1, 1) and solution.y.shape == (1, 1)
        point = history.n_initial + i
        u, x, y = history.u[point, 0], solution.x[0, 0], solution.y[0, 0]
        mean, _ = predict(fitted, np.array([[u]]))
        assert x + np.exp(x) == pytest.approx(y, abs=1e-6)
        assert y == pytest.approx(mean[0], abs=1e-6)

        incumbent = float(incumbents(history)[point - 1])
        nlp = hybrid_acquisition(
            illustrative_problem,
            fitted,
            run.scenarios,
            incumbent,
            EI(),
            approximation,
        ).nlp
        w = variables_at(
            u,
            solution.x[:, 0],
            solution.y[:, 0],
            approximation.unknown_states_as_variables,
        )
        assert solution.value == pytest.approx(float(nlp.f(w)), abs=1e-12)


def test_linearization_draws_nothing_from_the_generator(illustrative_problem):
    """Linearization draws no scenarios: a run without iterations leaves the
    generator where evaluating the initial points does, while SAA draws its
    scenarios."""
    # Initial points only (0 iterations), with each approximation
    measure = oracle_measure(
        illustrative_problem.system, oracle=lambda v: [cas.sin(v.u)]
    )
    u = np.array([[-0.5], [0.5]])
    states = {}
    for approximation in (Linearization(), SAA()):
        rng = np.random.default_rng(0)
        settings = HybridBOSettings(gp=ex.gp, approximation=approximation)
        hybrid_bo(illustrative_problem, measure, u, 0, rng, settings=settings)
        states[type(approximation)] = rng.bit_generator.state

    # The initial points' evaluation alone
    rng = np.random.default_rng(0)
    evaluate(illustrative_problem, measure, u, rng)

    assert states[Linearization] == rng.bit_generator.state
    assert states[SAA] != rng.bit_generator.state


def test_gp_inputs_are_the_arguments_of_h():
    """The GP is fitted on the columns of `h_inputs`, here u and x, mapping to y; the
    default GP has one lengthscale per argument."""
    problem = problem_with(["u", "x"])

    run = illustrative_run(
        problem, HybridBOSettings(gp=ex.gp, approximation=SAA(n_scenarios=5))
    )

    # The fitted GP's inputs, unscaled, are the (u, x) of the points before it
    fitted = run.surrogates[-1]
    inputs = (fitted.X - fitted.x_scaler.intercept) / fitted.x_scaler.slope
    history = run.history
    np.testing.assert_allclose(inputs, np.column_stack([history.u, history.x])[:3])
    kernel = fitted.gp.kernel
    assert isinstance(kernel, Scale) and isinstance(kernel.kernel, Matern52)
    assert isinstance(kernel.kernel.lengthscale, tuple)
    assert len(kernel.kernel.lengthscale) == 2


def test_record_holds_the_settings_and_the_gp_over_the_arguments_of_h(
    illustrative_problem,
):
    """The record names the method and its settings, and the prior GP as its repr."""
    approximation = SAA(n_scenarios=5, unknown_states_as_variables=False)
    settings = HybridBOSettings(gp=ex.gp, approximation=approximation)

    record = illustrative_run(illustrative_problem, settings).record

    assert record["method"] == "hybrid_bo"
    assert (
        "approximation=SAA(n_scenarios=5, unknown_states_as_variables=False)"
        in record["settings"]
    )
    assert "GP(" not in record["settings"]
    assert record["gp"] == repr(ex.gp)


def test_settings_are_required_and_keyword_only(illustrative_problem):
    """Hybrid BO has no default settings, since its GP is stated by the experiment,
    and the settings are keyword-only, so they can't be passed in `rng`'s place."""
    measure = oracle_measure(
        illustrative_problem.system, oracle=lambda v: [cas.sin(v.u)]
    )
    u = np.array([[-0.5], [0.5]])

    with pytest.raises(TypeError, match="gp"):
        HybridBOSettings()  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="settings"):
        hybrid_bo(illustrative_problem, measure, u, 0, np.random.default_rng(0))  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        hybrid_bo(  # ty: ignore[missing-argument]
            illustrative_problem,
            measure,
            u,
            0,
            np.random.default_rng(0),
            HybridBOSettings(gp=ex.gp),  # ty: ignore[too-many-positional-arguments]
        )


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

    run = illustrative_run(
        illustrative_problem,
        HybridBOSettings(gp=ex.gp, approximation=SAA(n_scenarios=5)),
    )

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
            np.random.default_rng(0),
            settings=HybridBOSettings(gp=ex.gp, approximation=SAA(n_scenarios=5)),
        )

    run = info.value.run
    assert run is not None
    assert run.history.u.shape == (2, 1)
    assert run.surrogates == []
    assert run.scenarios is not None and run.scenarios.z.shape == (5, 1)
    assert run.record["method"] == "hybrid_bo"
