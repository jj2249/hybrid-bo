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


def measure_without_oracle_values(
    u: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The illustrative example's oracle simulation, reading only the measured
    variables, so its draws are the methods'."""
    return ex.measure(u, rng).measured


def check_the_dataset_holds_what_was_measured(dataset: hb.Dataset) -> None:
    """Every row holds the measured y, the x that solves g, and f there, and no oracle
    values."""
    assert dataset.u.shape == (N_INITIAL + N_ITERATIONS, 1)
    assert dataset.x_oracle is None and dataset.f_oracle is None
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
    """A measurement that returns an array goes through the hybrid BO loop, and each
    trial point joins the dataset."""
    # Setup: the initial points and the scenarios, drawn once
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    scenarios = hb.saa_scenarios(5, ex.system.n_y, rng)
    start = evaluate_points(ex.problem, measure_sin, points, rng)

    # Act
    dataset, solutions = hybrid_loop(
        ex.problem,
        measure_sin,
        start,
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
    """A measurement that returns an array goes through the standard BO loop, and each
    trial point joins the dataset."""
    # Setup: the initial points
    rng = np.random.default_rng(0)
    points = hb.initial_points(ex.system.u, N_INITIAL, rng)
    start = evaluate_points(ex.problem, measure_sin, points, rng)

    # Act
    dataset, solutions = standard_loop(
        ex.problem,
        measure_sin,
        start,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
    )

    # Check
    check_the_dataset_holds_what_was_measured(dataset)
    trial_points = np.array([solution.u for solution in solutions])
    np.testing.assert_array_equal(dataset.u[N_INITIAL:], trial_points)


def test_the_hybrid_loop_is_hybrid_bo():
    """From the same seed, the loop takes the same points as `hybrid_bo`: the regression
    tests run the loop, so this keeps the method tied to it while it exists."""
    # The method
    approximation = hb.SAA(n_scenarios=5)
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
    scenarios = hb.saa_scenarios(5, ex.system.n_y, rng)
    start = evaluate_points(ex.problem, measure_without_oracle_values, points, rng)
    dataset, solutions = hybrid_loop(
        ex.problem,
        measure_without_oracle_values,
        start,
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
    start = evaluate_points(ex.problem, measure_without_oracle_values, points, rng)
    dataset, solutions = standard_loop(
        ex.problem,
        measure_without_oracle_values,
        start,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
    )

    # Check
    np.testing.assert_array_equal(dataset.u, run.dataset.u)
    np.testing.assert_array_equal(dataset.f, run.dataset.f)
    assert [s.value for s in solutions] == pytest.approx(
        [a.value for a in run.acquisitions], rel=0
    )
