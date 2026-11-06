"""The loop a script writes, with a measurement that has no oracle, and against the
methods it replaces."""

import illustrative_example as ex
import numpy as np
import pytest
from plain_loop import evaluate_points, hybrid_loop, standard_loop

import hybrid_bo as hb

N_INITIAL, N_ITERATIONS = 5, 2


def measure_sin(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A real system to the library: it returns the measured y = sin(u) and no more."""
    return np.sin(u)


def check_the_dataset_holds_what_was_measured(dataset: hb.Dataset) -> None:
    """Every row holds the measured y, the x that solves g, and f there."""
    assert dataset.u.shape == (N_INITIAL + N_ITERATIONS, 1)
    assert hb.dataset.n_iterations(dataset) == N_ITERATIONS
    np.testing.assert_array_equal(dataset.y, np.sin(dataset.u))
    np.testing.assert_allclose(
        dataset.x + np.exp(dataset.x) - dataset.y, 0.0, atol=1e-9
    )
    expected_f = [
        ex.problem.f(*row).full().item()
        for row in zip(dataset.x, dataset.y, dataset.u, strict=True)
    ]
    np.testing.assert_allclose(dataset.f, expected_f)


def test_hybrid_bo_runs_with_a_measurement_that_has_no_oracle():
    """A measurement that returns an array goes through hybrid BO's optimization loop, and each
    trial point joins the dataset."""
    # Setup: the initial points and the scenarios, drawn once
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    scenarios = hb.saa_scenarios(5, ex.system.n_y, rng)
    initial = evaluate_points(ex.problem, measure_sin, points, rng)

    # Act
    dataset, solutions = hybrid_loop(
        ex.problem,
        measure_sin,
        initial,
        scenarios,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
        approximation=hb.SAA(n_scenarios=5),
    )

    # Check
    check_the_dataset_holds_what_was_measured(dataset)
    trial_points = np.array([solution.u for solution in solutions])
    np.testing.assert_array_equal(dataset.u[N_INITIAL:], trial_points)


def test_standard_bo_runs_with_a_measurement_that_has_no_oracle():
    """A measurement that returns an array goes through standard BO's optimization loop, and each
    trial point joins the dataset."""
    # Setup: the initial points
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    initial = evaluate_points(ex.problem, measure_sin, points, rng)

    # Act
    dataset, solutions = standard_loop(
        ex.problem,
        measure_sin,
        initial,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
    )

    # Check
    check_the_dataset_holds_what_was_measured(dataset)
    trial_points = np.array([solution.u for solution in solutions])
    np.testing.assert_array_equal(dataset.u[N_INITIAL:], trial_points)


def latin_hypercube_design(rng: np.random.Generator) -> np.ndarray:
    """The design `hb.latin_hypercube_sampling` draws after the initial points."""
    return hb.initial_points(ex.system.u, N_ITERATIONS, rng)


def uniform_design(rng: np.random.Generator) -> np.ndarray:
    """The design `hb.uniform_sampling` draws after the initial points."""
    lower, upper = (np.array([bound]) for bound in ex.u.start)
    return hb.sampling.uniform(lower, upper, N_ITERATIONS, rng)


# `parametrize` runs the test once per entry, with the entry as the argument
@pytest.mark.parametrize(
    "approximation",
    [
        hb.SAA(n_scenarios=5),
        hb.SoftSAA(temperature=50e-3, n_scenarios=5),
        hb.Linearization(),
    ],
)
def test_the_hybrid_loop_is_hybrid_bo(approximation):
    """From the same seed, the loop takes the same points as `hybrid_bo`: the regression
    tests run the loop, so this keeps the method tied to it while it exists."""
    # The method
    run = hb.hybrid_bo(
        ex.problem,
        ex.measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(0),
        gp=ex.gp,
        acquisition=hb.EI(),
        approximation=approximation,
    )

    # The loop, drawing what the method draws in the same order
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    if isinstance(approximation, hb.Linearization):
        scenarios = hb.nominal_scenario(ex.system.n_y)
    else:
        scenarios = hb.saa_scenarios(approximation.n_scenarios, ex.system.n_y, rng)
    initial = evaluate_points(ex.problem, ex.measure, points, rng)
    dataset, solutions = hybrid_loop(
        ex.problem,
        ex.measure,
        initial,
        scenarios,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
        approximation=approximation,
    )

    # Check
    np.testing.assert_array_equal(dataset.u, run.dataset.u)
    np.testing.assert_array_equal(dataset.f, run.dataset.f)
    assert [s.value for s in solutions] == [a.value for a in run.acquisitions]


def test_the_standard_loop_is_standard_bo():
    """From the same seed, the loop takes the same points as `standard_bo`, as
    `test_the_hybrid_loop_is_hybrid_bo` says."""
    # The method
    run = hb.standard_bo(
        ex.problem,
        ex.measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(0),
        gp=ex.gp,
        acquisition=hb.EI(),
    )

    # The loop, drawing what the method draws in the same order
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    initial = evaluate_points(ex.problem, ex.measure, points, rng)
    dataset, solutions = standard_loop(
        ex.problem,
        ex.measure,
        initial,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
    )

    # Check
    np.testing.assert_array_equal(dataset.u, run.dataset.u)
    np.testing.assert_array_equal(dataset.f, run.dataset.f)
    assert [s.value for s in solutions] == [a.value for a in run.acquisitions]


@pytest.mark.parametrize(
    ("method", "draw_design"),
    [
        (hb.latin_hypercube_sampling, latin_hypercube_design),
        (hb.uniform_sampling, uniform_design),
    ],
)
def test_the_points_of_a_baseline_are_the_method(method, draw_design):
    """From the same seed, evaluating the initial points and a design one by one gives
    the method's dataset, as `test_the_hybrid_loop_is_hybrid_bo` says."""
    # The method
    run = method(
        ex.problem,
        ex.measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(0),
    )

    # The script, drawing what the method draws in the same order
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    all_points = np.concatenate([points, draw_design(rng)])
    dataset = evaluate_points(ex.problem, ex.measure, all_points, rng)

    # Check
    np.testing.assert_array_equal(dataset.u, run.dataset.u)
    np.testing.assert_array_equal(dataset.f, run.dataset.f)
