"""One iteration of hybrid and standard BO, and the loop a script writes around it."""

import dataclasses
import logging

import casadi as cas
import illustrative_example as ex
import matplotlib
import matplotlib.figure
import matplotlib.style
import numpy as np
import pytest
import scipy
from plain_loop import evaluate_points, hybrid_loop, standard_loop
from scipy.stats import norm

import hybrid_bo as hb
from hybrid_bo.hybrid_model import _pack

N_SCENARIOS = 4

# Three scenarios with unequal weights, so a plain mean would give other values
Z, W = np.array([[-1.0], [0.5], [2.0]]), np.array([0.2, 0.3, 0.5])


def draw_scenarios(
    n: int, n_y: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """n scenarios for the sample average: standard normal draws, equal weights."""
    return rng.standard_normal((n, n_y)), np.full(n, 1 / n)


def measure_sin(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A real system to the library: it returns the measured y = sin(u) and no more."""
    return np.sin(u)


def training_data() -> hb.Dataset:
    """Five points of the illustrative example, with y = sin(u) as if measured."""
    u = np.linspace(-1.8, 1.8, 5)[:, np.newaxis]
    rng = np.random.default_rng(1)
    return hb.reconstruct_states(ex.problem, u=u, measured=np.sin(u), rng=rng)


def h_of_x_problem() -> hb.Problem:
    """The illustrative system with h a function of u and x."""
    u = hb.Var("u", -2.0, 2.0)
    x = hb.Var("x", start=(-2.5, 1.0))
    y = hb.Var("y", start=(-2.0, 2.0))
    system = hb.System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        h_inputs=[u, x],
        measured=[y],
    )
    return hb.Problem(system, f=lambda v: v.x**2 + v.y)


def no_known_equations_problem(n_u: int = 1) -> hb.Problem:
    """No mechanistic states: one unknown state y, measured, with f a function of y."""
    u = [hb.Var(f"u{i + 1}", -2.0, 2.0) for i in range(n_u)]
    y = hb.Var("y", start=(-2.0, 2.0))
    system = hb.System(u=u, y=[y], h_inputs=u, measured=[y])
    return hb.Problem(system, f=lambda v: (v.y - 0.3) ** 2 + v.u1**2 / 10)


def two_unknown_states_problem() -> hb.Problem:
    """Two unknown states, measured, tied to one mechanistic state by g."""
    u = hb.Var("u", -2.0, 2.0)
    x = hb.Var("x", start=(-3.0, 2.0))
    y1, y2 = hb.Var("y1", start=(-2.0, 2.0)), hb.Var("y2", start=(-2.0, 2.0))
    system = hb.System(
        u=[u],
        x=[x],
        y=[y1, y2],
        g=lambda v: [v.x + cas.exp(v.x) - v.y1 - v.y2],
        h_inputs=[u],
        measured=[y1, y2],
    )
    return hb.Problem(system, f=lambda v: v.x**2 + (v.y1 - v.y2) ** 2)


# `parametrize` runs the test once per entry, with the entry as the argument
@pytest.mark.parametrize(
    ("z", "w"),
    [
        (np.zeros(3), np.full(3, 1 / 3)),
        (np.zeros((3, 1)), np.full(2, 0.5)),
        (np.zeros((2, 1)), np.array([1.5, -0.5])),
        (np.zeros((2, 1)), np.array([0.2, 0.2])),
    ],
)
def test_scenarios_reject_inconsistent_draws_or_weights(z, w):
    """The draws must be (S, P), with S weights that are >= 0 and sum to 1."""
    fitted = ex.gp.fit(
        np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), np.random.default_rng(0)
    )

    # `raises` passes only if the block raises that exception
    with pytest.raises(ValueError, match="Scenario"):
        hb.hybrid_deterministic_equivalent(
            ex.problem, (fitted,), z, w, 0.0, acquisition=hb.EI()
        )


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_solving_the_equivalent_gives_the_trial_point_and_states(
    unknown_states_as_variables,
):
    """The solution holds the trial point, the acquisition problem's objective there
    and each scenario's states, which satisfy the known equations."""
    # Setup: a surrogate model and the scenarios, drawn by the caller
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = hb.fit_surrogate(ex.system, dataset, gp=ex.gp, rng=rng)
    z, w = draw_scenarios(N_SCENARIOS, ex.system.n_y, rng)
    incumbent = hb.incumbents(dataset)[-1]

    # Act
    built = hb.hybrid_deterministic_equivalent(
        ex.problem,
        fitted,
        z,
        w,
        incumbent,
        acquisition=hb.EI(),
        unknown_states_as_variables=unknown_states_as_variables,
    )
    solution = built.solve(rng)

    # Check the shapes, and the bounds of u
    assert solution.x is not None and solution.y is not None
    assert solution.u.shape == (ex.system.n_u,)
    assert solution.x.shape == solution.y.shape == (N_SCENARIOS, 1)
    assert np.all(np.abs(solution.u) <= 2.0 + 1e-6)

    # Check the known equations of every scenario
    residuals = solution.x + np.exp(solution.x) - solution.y
    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)

    # Check the value against the problem's objective at the solution
    expected = float(built.f(solution.w))
    assert solution.value == pytest.approx(expected, rel=1e-9)


def test_solving_standard_acquisition_minimizes_analytic_ei():
    """The solution holds the trial point and -EI there, computed here from the GP's
    mean and std."""
    # Setup: a GP of the objective
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = ex.gp.fit(dataset.u, dataset.f, rng)
    incumbent = float(hb.incumbents(dataset)[-1])

    # Act
    built = hb.standard_acquisition(ex.problem, fitted, incumbent)
    solution = built.solve(rng)

    # Check against EI in closed form
    mean, std = fitted.predict(solution.w[np.newaxis])
    gamma = (incumbent - mean[0]) / std[0]
    expected_ei = (incumbent - mean[0]) * norm.cdf(gamma) + std[0] * norm.pdf(gamma)
    assert solution.value == pytest.approx(-expected_ei, rel=1e-6)

    # Check that no other point of a grid has a better value
    grid = np.linspace(-2.0, 2.0, 201)[np.newaxis]
    assert solution.value <= built.f.map(201)(grid).full().min() + 1e-9


def ei_problems(incumbent: float, temperature: float):
    """The SAA-SoftEI and SAA-EI problems of the illustrative system over Z and W, with the
    GP substituted, and the decision vector and f_s (S,) at u = 0.3 and x = (-1, 0, 0.5)."""
    problem = h_of_x_problem()
    X = np.array([[-1.5, -1.0], [-0.5, 0.2], [0.4, -0.3], [1.7, 0.6]])
    fitted = ex.gp.fit(X, np.array([0.3, -1.2, 0.9, 0.8]), np.random.default_rng(0))
    u, x = 0.3, np.array([-1.0, 0.0, 0.5])

    # The reference values: y_s = mean + std * z_s at each scenario's arguments of h
    mean, std = fitted.predict(np.column_stack([np.full_like(x, u), x]))
    f = x**2 + mean + std * Z[:, 0]

    soft, saa = (
        hb.hybrid_deterministic_equivalent(
            problem,
            (fitted,),
            Z,
            W,
            incumbent,
            acquisition=acquisition,
            unknown_states_as_variables=False,
        )
        for acquisition in (hb.SoftEI(temperature=temperature), hb.EI())
    )
    return soft, saa, _pack(np.array([u]), x[:, np.newaxis]), f


def gradient(nlp, w: np.ndarray) -> np.ndarray:
    """The gradient of the acquisition problem's objective at w, (n,)."""
    symbol = cas.SX.sym("w", len(w))
    function = cas.Function("gradient", [symbol], [cas.gradient(nlp.f(symbol), symbol)])
    return function(w).full().ravel()


def test_ei_is_the_weighted_average_over_the_scenarios():
    """SAA-EI minimizes the weighted average of min(f_s - incumbent, 0), with the unknown
    states predicted by the GP at each scenario's arguments of h."""
    _, saa, w, f = ei_problems(incumbent=1.0, temperature=0.5)

    expected = W @ np.minimum(f - 1.0, 0)
    np.testing.assert_allclose(saa.f(w).full().item(), expected, rtol=1e-10)


def test_soft_ei_ei_is_minus_log_the_weighted_smooth_improvement():
    """SAA-SoftEI minimizes -log sum_s w_s tau log(1 + exp(a_s / tau)), a_s = incumbent -
    f_s."""
    soft, _, w, f = ei_problems(incumbent=1.0, temperature=0.5)

    smooth = 0.5 * np.log1p(np.exp((1.0 - f) / 0.5))
    expected = -np.log(W @ smooth)
    np.testing.assert_allclose(soft.f(w).full().item(), expected, rtol=1e-10)


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_soft_ei_needs_a_positive_finite_temperature(temperature):
    """The temperature is in units of the objective, so it must be positive and
    finite."""
    with pytest.raises(ValueError, match="temperature"):
        hb.SoftEI(temperature=temperature)


def test_soft_ei_ei_has_a_gradient_where_no_scenario_improves():
    """With the incumbent below every f_s, SAA-EI is flat, while SAA-SoftEI stays finite
    with a nonzero gradient in u."""
    soft, saa, w, f = ei_problems(incumbent=-1.0, temperature=0.5)
    assert np.all(f > -1.0)

    # SAA-EI is flat there
    assert saa.f(w).full().item() == 0
    assert np.all(gradient(saa, w) == 0)

    # SAA-SoftEI's objective and its gradient in u
    assert np.isfinite(soft.f(w).full().item())
    u_gradient, _, _ = soft.unpack(gradient(soft, w))
    assert np.all(np.isfinite(u_gradient)) and np.all(u_gradient != 0)


@pytest.mark.parametrize("incumbent", [-3.0, 5.0])
def test_soft_ei_ei_is_finite_far_from_the_incumbent(incumbent):
    """Where exp(a_s / tau) underflows or overflows, the objective and its gradient are
    finite; far below, the objective matches its asymptote."""
    # Setup: a temperature small enough to put every a_s / tau beyond +-1000
    soft, _, w, f = ei_problems(incumbent, 0.002)
    a = (incumbent - f) / 0.002
    assert np.all(np.abs(a) > 1000)

    # Check
    value = soft.f(w).full().item()
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient(soft, w)))
    if incumbent < 0:
        expected = -(np.log(0.002) + scipy.special.logsumexp(a, b=W))
        np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_the_equivalent_needs_one_gp_and_column_per_unknown_state():
    """Scenarios or a surrogate model of the wrong size are refused."""
    fitted = ex.gp.fit(
        np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), np.random.default_rng(0)
    )
    z, w = draw_scenarios(3, 2, np.random.default_rng(0))

    with pytest.raises(ValueError, match="one column of z"):
        hb.hybrid_deterministic_equivalent(
            ex.problem, (fitted,), z, w, 0.0, acquisition=hb.EI()
        )
    with pytest.raises(ValueError, match="one fitted GP"):
        hb.hybrid_deterministic_equivalent(
            ex.problem, (fitted, fitted), Z, W, 0.0, acquisition=hb.EI()
        )


@pytest.mark.parametrize(
    ("acquisition", "unknown_states_as_variables"),
    [
        (hb.EI(), True),
        (hb.EI(), False),
        (hb.SoftEI(temperature=0.5), True),
    ],
)
@pytest.mark.parametrize(
    "problem",
    [ex.problem, no_known_equations_problem(), h_of_x_problem()],
    ids=["known_equations", "no_known_equations", "h_of_x"],
)
def test_hybrid_bo_runs_with_a_measurement_of_your_own(
    problem, acquisition, unknown_states_as_variables
):
    """A measurement that only returns an array goes through the loop, and each trial
    point joins the dataset."""
    # Setup: the initial points and the scenarios, drawn once
    rng = np.random.default_rng(0)
    points = hb.lhs_points(problem.system.u, 5, rng)
    z, w = draw_scenarios(5, problem.system.n_y, rng)
    start = evaluate_points(problem, measure_sin, points, rng)

    # Act
    dataset, solutions = hybrid_loop(
        problem,
        measure_sin,
        start,
        z,
        w,
        rng,
        n_iterations=2,
        gp=ex.gp,
        acquisition=acquisition,
        unknown_states_as_variables=unknown_states_as_variables,
    )

    # Check: the measured y, and the trial points in the dataset
    np.testing.assert_array_equal(dataset.y, np.sin(dataset.u))
    trial_points = np.array([solution.u for solution in solutions])
    np.testing.assert_array_equal(dataset.u[5:], trial_points)


def test_hybrid_bo_runs_with_two_unknown_states():
    """Each unknown state gets its own GP, and the scenarios a column each."""
    problem = two_unknown_states_problem()
    rng = np.random.default_rng(0)
    z, w = draw_scenarios(5, 2, rng)

    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Both unknown states."""
        return np.array([np.sin(u[0]), np.cos(u[0])])

    start = evaluate_points(
        problem, measure, hb.lhs_points(problem.system.u, 5, rng), rng
    )
    dataset, solutions = hybrid_loop(
        problem,
        measure,
        start,
        z,
        w,
        rng,
        n_iterations=2,
        gp=ex.gp,
        acquisition=hb.EI(),
    )

    assert dataset.y.shape == (7, 2)
    assert solutions[0].y is not None and solutions[0].y.shape == (5, 2)


def test_standard_bo_runs_with_a_measurement_of_your_own():
    """A measurement that only returns an array goes through standard BO's loop."""
    rng = np.random.default_rng(0)
    points = hb.lhs_points(ex.system.u, 5, rng)
    start = evaluate_points(ex.problem, measure_sin, points, rng)

    dataset, solutions = standard_loop(
        ex.problem, measure_sin, start, rng, n_iterations=2, gp=ex.gp
    )

    trial_points = np.array([solution.w for solution in solutions])
    np.testing.assert_array_equal(dataset.u[5:], trial_points)


def test_uniform_points_stay_within_the_start_bounds():
    """A uniform baseline is `uniform_points` through the measurement."""
    points = hb.uniform_points(ex.system.u, 50, np.random.default_rng(0))

    assert points.shape == (50, 1)
    assert np.all(np.abs(points) <= 2.0)


def test_a_saved_dataset_loads_equal_and_extends(tmp_path):
    """A dataset round-trips through `save_dataset`, and the loaded one takes new rows;
    an existing file isn't overwritten."""
    dataset = training_data()
    path = tmp_path / "dataset.npz"

    hb.save_dataset(dataset, path)
    loaded = hb.load_dataset(path)

    for field in ("u", "x", "y", "f"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(dataset, field))
    assert len(hb.extend(loaded, dataset).f) == 2 * len(dataset.f)
    with pytest.raises(FileExistsError):
        hb.save_dataset(dataset, path)


def test_regret_and_convergence_draw_the_median_of_each_label():
    """Each draws a curve per label, the median of its curves, and a shorter curve counts
    up to its last evaluation."""
    curves = {"a": [np.array([3.0, 1.0]), np.array([5.0, 2.0]), np.array([7.0])]}
    regret_ax, convergence_ax = matplotlib.figure.Figure().subplots(2)

    regret = hb.plots.regret(curves, 0.0, regret_ax)
    convergence = hb.plots.convergence(curves, convergence_ax)

    assert regret is regret_ax and convergence is convergence_ax
    (line,) = regret.get_lines()
    np.testing.assert_allclose(np.asarray(line.get_ydata()), [5.0, 1.5])
    assert regret.get_yscale() == "log"
    assert convergence.get_yscale() == "linear"


def test_surrogate_draws_the_gp_and_its_points_in_measured_units():
    """The mean is the fitted GP's prediction, and the points are the training data as measured."""
    dataset = training_data()
    fitted = ex.gp.fit(dataset.u, dataset.y[:, 0], np.random.default_rng(0))
    ax = matplotlib.figure.Figure().subplots()

    hb.plots.surrogate(fitted, -2.0, 2.0, ax)

    (line,) = ax.get_lines()
    x = np.asarray(line.get_xdata())
    np.testing.assert_allclose(
        np.asarray(line.get_ydata()), fitted.predict(x[:, np.newaxis])[0]
    )
    drawn = np.asarray(ax.collections[-1].get_offsets())
    np.testing.assert_allclose(drawn, np.column_stack([dataset.u, dataset.y]))


@pytest.mark.parametrize(
    ("acquisition", "unknown_states_as_variables"),
    [
        (hb.EI(), True),
        (hb.EI(), False),
        (hb.SoftEI(temperature=0.5), True),
    ],
)
def test_acquisition_on_grid_gives_the_solution_and_solves_the_equations(
    acquisition, unknown_states_as_variables
):
    """At the trial point the grid gives the stored solution, and over a grid each
    scenario's states solve the known equations and f is the objective there."""
    # Setup: one solved iteration
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = hb.fit_surrogate(ex.system, dataset, gp=ex.gp, rng=rng)
    z, w = draw_scenarios(N_SCENARIOS, ex.system.n_y, rng)
    incumbent = hb.incumbents(dataset)[-1]
    built = hb.hybrid_deterministic_equivalent(
        ex.problem,
        fitted,
        z,
        w,
        incumbent,
        acquisition=acquisition,
        unknown_states_as_variables=unknown_states_as_variables,
    )
    solution = built.solve(rng)

    # Act: the trial point, then a grid
    u = np.vstack([solution.u, np.linspace(-1.9, 1.9, 21)[:, np.newaxis]])
    ei, scenarios = hb.plots.acquisition_on_grid(built, solution, u)

    # Check: the stored solution at the trial point
    assert solution.x is not None
    assert ei[0] == pytest.approx(
        built.acquisition.from_objective(np.array(solution.value)), rel=1e-6, abs=1e-8
    )
    x_at_trial_point = np.array([scenario.x[0] for scenario in scenarios])
    np.testing.assert_allclose(x_at_trial_point, solution.x, atol=1e-6)

    # Check: the equations and the objective wherever a root was found
    solved = np.isfinite(ei)
    assert solved.mean() > 0.8
    for scenario in scenarios:
        x, y, f = scenario.x[solved], scenario.y[solved], scenario.f[solved]
        np.testing.assert_array_equal(scenario.u, u)
        np.testing.assert_allclose(x + np.exp(x) - y, 0.0, atol=1e-7)
        np.testing.assert_allclose(f, (6 * x[:, 0] - 2) ** 2 * np.sin(12 * x[:, 0] - 4))


def test_acquisition_on_grid_leaves_a_gap_where_a_point_fails():
    """A state that would leave its bounds is a failed point, and NaN in every output."""
    built, solution, *_ = solved_iteration()
    n_u = built.n_u
    upper = np.concatenate([built.upper[:n_u], solution.w[n_u:] + 1e-6])
    bounded = dataclasses.replace(built, upper=upper)
    u = np.array([solution.u, solution.u + 1.0, solution.u - 1.0])

    ei, scenarios = hb.plots.acquisition_on_grid(bounded, solution, u)

    assert np.isfinite(ei[0])
    assert np.isnan(ei[1:]).any()
    assert np.array_equal(np.isnan(ei), np.isnan(scenarios[0].f))


def test_acquisition_on_grid_of_standard_bo_is_analytic():
    """Standard BO has no scenarios, and its EI is analytic."""
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = ex.gp.fit(dataset.u, dataset.f, rng)
    incumbent = float(hb.incumbents(dataset)[-1])
    built = hb.standard_acquisition(ex.problem, fitted, incumbent)
    solution = built.solve(rng)

    ei, scenarios = hb.plots.acquisition_on_grid(
        built, solution, solution.w[np.newaxis]
    )

    assert ei[0] == pytest.approx(-solution.value)
    assert scenarios == ()


def test_acquisition_on_grid_works_with_no_states_to_solve():
    """With no known equations and the GP substituted, there is nothing to solve."""
    problem = no_known_equations_problem()
    rng = np.random.default_rng(0)
    points = hb.lhs_points(problem.system.u, 5, rng)
    dataset = evaluate_points(problem, measure_sin, points, rng)
    fitted = hb.fit_surrogate(problem.system, dataset, gp=ex.gp, rng=rng)
    z, w = draw_scenarios(N_SCENARIOS, 1, rng)
    incumbent = hb.incumbents(dataset)[-1]
    built = hb.hybrid_deterministic_equivalent(
        problem,
        fitted,
        z,
        w,
        incumbent,
        acquisition=hb.EI(),
        unknown_states_as_variables=False,
    )
    solution = built.solve(rng)

    u = np.array([[0.1], [solution.u[0]]])
    ei, _ = hb.plots.acquisition_on_grid(built, solution, u)

    assert np.all(np.isfinite(ei))
    assert ei[1] == pytest.approx(-solution.value)


def solved_iteration() -> tuple:
    """A hybrid iteration with SoftEI: the problem, solution, GP and data."""
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = hb.fit_surrogate(ex.system, dataset, gp=ex.gp, rng=rng)
    z, w = draw_scenarios(N_SCENARIOS, ex.system.n_y, rng)
    incumbent = hb.incumbents(dataset)[-1]
    built = hb.hybrid_deterministic_equivalent(
        ex.problem, fitted, z, w, incumbent, acquisition=hb.SoftEI(temperature=0.5)
    )
    solution = built.solve(rng)
    return built, solution, fitted[0], dataset


def test_soft_ei_gives_ei_from_its_log():
    """SoftEI minimizes -log SoftEI, so it is exp of minus the objective."""
    built, solution, *_ = solved_iteration()

    ei, _ = hb.plots.acquisition_on_grid(built, solution, solution.u[np.newaxis])

    np.testing.assert_allclose(ei, np.exp(-solution.value), rtol=1e-6)


def test_iteration_draws_hybrid_bos_panels():
    """The GP, f under the scenarios and EI, whose peak is at the trial point."""
    built, solution, fitted, dataset = solved_iteration()
    axes = matplotlib.figure.Figure().subplots(3)

    hb.plots.iteration(built, solution, fitted, dataset, axes)

    u, ei = (np.asarray(data) for data in axes[2].get_lines()[0].get_data())
    assert abs(u[np.nanargmax(ei)] - solution.u[0]) <= u[1] - u[0]
    for ax in axes:
        assert ax.get_lines()[-1].get_label() == "trial point"


def test_iteration_draws_standard_bos_panels():
    """The GP of f and EI, on two axes."""
    rng = np.random.default_rng(2)
    dataset = training_data()
    fitted = ex.gp.fit(dataset.u, dataset.f, rng)
    incumbent = float(hb.incumbents(dataset)[-1])
    built = hb.standard_acquisition(ex.problem, fitted, incumbent)
    solution = built.solve(rng)
    axes = matplotlib.figure.Figure().subplots(2)

    hb.plots.iteration(built, solution, fitted, dataset, axes)

    assert axes[1].get_ylabel() == "acquisition"


def test_iteration_needs_one_axes_per_panel():
    """Hybrid BO draws three panels, so two axes are refused."""
    built, solution, fitted, dataset = solved_iteration()
    axes = matplotlib.figure.Figure().subplots(2)

    with pytest.raises(ValueError, match="panels"):
        hb.plots.iteration(built, solution, fitted, dataset, axes)


def test_fits_and_solves_are_logged(caplog):
    """At `INFO`, the log says which unknown state is fitted, what it was fitted to, and
    how many starts of each solve succeeded.

    `caplog` is a pytest fixture that captures log records during the test.
    """
    with caplog.at_level(logging.INFO, logger="hybrid_bo"):
        hb.fit_surrogate(
            ex.system, training_data(), gp=ex.gp, rng=np.random.default_rng(0)
        )

    assert "Fitting the GP of y" in caplog.text
    assert "to 5 points" in caplog.text
    assert "negative_log_marginal_likelihood: 10 of 10 starts solved" in caplog.text


def test_the_library_style_turns_the_grid_on():
    """`plt.style.use("hybrid_bo.default")` loads the style shipped in the package."""
    with matplotlib.style.context("hybrid_bo.default"):
        assert matplotlib.rcParams["axes.grid"]
