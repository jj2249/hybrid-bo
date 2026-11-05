"""Regression tests: each method of the illustrative example vs stored results.

These lock in the *current* behavior. If a PR intentionally changes behavior,
regenerate with `pixi run -e dev test --update-baselines` and review the JSON diff.

Each method runs as the plain loop a script writes (`plain_loop.py`), one point at a
time, drawing what the method it replaces drew in the same order. The baselines were
recorded by the pre-refactor code, so each result is mapped to the keys of its result
files (`ResultsBO`) to compare. The acquisition problems' solutions (#53) came later and
have keys of their own.
"""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import illustrative_example as ex
import numpy as np
import pytest
from plain_loop import evaluate_points, hybrid_loop, standard_loop

import hybrid_bo as hb
from hybrid_bo.acquisition import AcquisitionSolution
from hybrid_bo.oracle import OracleMeasurement

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8

# Small, seeded runs so the test is fast and deterministic
SEED, N_ITERATIONS = 0, 2

# What a run of a method gives: the dataset, and each iteration's solution
Result = tuple[hb.Dataset, list[AcquisitionSolution]]


def run_hybrid_bo(
    approximation: hb.SAA | hb.SoftSAA | hb.Linearization,
    measure: hb.Measure,
    rng: np.random.Generator,
) -> Result:
    """Hybrid BO: the initial points, the scenarios once, then the loop."""
    points = hb.initial_points(ex.system.u, ex.N_INITIAL, rng)
    if isinstance(approximation, hb.Linearization):
        scenarios = hb.nominal_scenario(ex.system.n_y)
    else:
        scenarios = hb.saa_scenarios(approximation.n_scenarios, ex.system.n_y, rng)
    return hybrid_loop(
        ex.problem,
        measure,
        evaluate_points(ex.problem, measure, points, rng),
        scenarios,
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
        approximation=approximation,
    )


def run_standard_bo(measure: hb.Measure, rng: np.random.Generator) -> Result:
    """Standard BO: the initial points, then the loop."""
    points = hb.initial_points(ex.system.u, ex.N_INITIAL, rng)
    return standard_loop(
        ex.problem,
        measure,
        evaluate_points(ex.problem, measure, points, rng),
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=hb.EI(),
    )


def run_latin_hypercube_sampling(
    measure: hb.Measure, rng: np.random.Generator
) -> Result:
    """Latin hypercube sampling: the initial points and the design, then every point."""
    points = hb.initial_points(ex.system.u, ex.N_INITIAL, rng)
    design = hb.initial_points(ex.system.u, N_ITERATIONS, rng)
    all_points = np.concatenate([points, design])
    return evaluate_points(ex.problem, measure, all_points, rng), []


def run_uniform_sampling(measure: hb.Measure, rng: np.random.Generator) -> Result:
    """Uniform sampling: the initial points and the design, then every point."""
    points = hb.initial_points(ex.system.u, ex.N_INITIAL, rng)
    lower, upper = (np.array([bound]) for bound in ex.u.start)
    design = hb.sampling.uniform(lower, upper, N_ITERATIONS, rng)
    all_points = np.concatenate([points, design])
    return evaluate_points(ex.problem, measure, all_points, rng), []


# The illustrative example's methods, with their labels in the stored results
_SUBSTITUTED = {"unknown_states_as_variables": False}
METHODS = {
    "hybrid_bo": partial(run_hybrid_bo, hb.SAA(n_scenarios=25, **_SUBSTITUTED)),
    "hybrid_bo_lin": partial(run_hybrid_bo, hb.Linearization(**_SUBSTITUTED)),
    "hybrid_bo_soft": partial(
        run_hybrid_bo,
        hb.SoftSAA(temperature=50e-3, n_scenarios=25, **_SUBSTITUTED),
    ),
    "standard_bo": run_standard_bo,
    "latin_hypercube_sampling": run_latin_hypercube_sampling,
    "uniform_sampling": run_uniform_sampling,
}


def with_oracle_values(
    dataset: hb.Dataset, measurements: list[OracleMeasurement]
) -> hb.Dataset:
    """The dataset with the oracle values the measurements gave, one per row."""
    x_oracle = np.array([measurement.x for measurement in measurements])
    y_oracle = np.array([measurement.y for measurement in measurements])
    f_oracle = np.array(
        [
            ex.problem.f(*row).full().item()
            for row in zip(x_oracle, y_oracle, dataset.u, strict=True)
        ]
    )
    return replace(dataset, x_oracle=x_oracle, y_oracle=y_oracle, f_oracle=f_oracle)


def as_baseline_columns(
    dataset: hb.Dataset, solutions: list[AcquisitionSolution]
) -> dict[str, np.ndarray]:
    """The run under the `ResultsBO` keys: they store y inside x, f as a column, and
    split every array into the initial points and the rest."""
    assert dataset.x_oracle is not None and dataset.y_oracle is not None
    assert dataset.f_oracle is not None
    columns = {
        "u": dataset.u,
        "x": np.column_stack([dataset.x, dataset.y]),
        "f": dataset.f[:, np.newaxis],
        "incumbents": hb.incumbents(dataset)[:, np.newaxis],
        "x_no_noise": np.column_stack([dataset.x_oracle, dataset.y_oracle]),
        "f_no_noise": dataset.f_oracle[:, np.newaxis],
        "incumbents_no_noise": hb.oracle_incumbents(dataset)[:, np.newaxis],
    }
    n = ex.N_INITIAL
    baseline = {
        f"{name}_{part}": value[:n] if part == "initial" else value[n:]
        for part in ("initial", "bo")
        for name, value in columns.items()
    }

    # The BO methods' acquisition solutions: the value per iteration, and hybrid BO's
    # scenario states (iteration, scenario, state)
    if solutions:
        baseline["acquisition_values"] = np.array([s.value for s in solutions])
    x = [s.x for s in solutions if s.x is not None]
    y = [s.y for s in solutions if s.y is not None]
    if x and y:
        baseline["acquisition_x"] = np.stack(x)
        baseline["acquisition_y"] = np.stack(y)
    return baseline


def run_method(method: str) -> dict[str, np.ndarray]:
    """Runs one run of a method as the experiment does and returns it under the
    `ResultsBO` keys."""
    # The loop sees the measured variables alone, as with a real system; the oracle
    # values are kept here, for the columns that score the run
    measurements: list[OracleMeasurement] = []

    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The oracle simulation, noting what it gave."""
        measurement = ex.measure(u, rng)
        measurements.append(measurement)
        return measurement.measured

    # As in the experiment: one generator per run, initial points drawn from it
    rng = np.random.default_rng(SEED).spawn(1)[0]
    dataset, solutions = METHODS[method](measure, rng)
    return as_baseline_columns(with_oracle_values(dataset, measurements), solutions)


# `parametrize` runs the test once per method, each as its own test case. The other
# argument is a fixture from `conftest.py`, filled in by pytest
@pytest.mark.parametrize("method", list(METHODS))
def test_illustrative_example(method, update_baselines):
    """The method reproduces its stored results on the illustrative example."""
    actual = run_method(method)

    # With --update-baselines, store the results instead of comparing. `pytest.skip`
    # ends the test and reports it as skipped
    baseline_path = BASELINE_DIR / f"illustrative_{method}.json"
    if update_baselines:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: np.asarray(v).tolist() for k, v in actual.items()}
        baseline_path.write_text(json.dumps(data, indent=1) + "\n")
        pytest.skip(f"updated {baseline_path.name}")

    # Every array matches its stored value
    expected = json.loads(baseline_path.read_text())
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(expected[name]),
            rtol=RTOL,
            atol=ATOL,
            err_msg=name,
            strict=True,
        )
