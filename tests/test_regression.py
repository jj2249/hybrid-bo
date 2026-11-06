"""Regression tests: each method of the illustrative example vs stored results.

These lock in the *current* behavior. If a PR intentionally changes behavior,
regenerate with `pixi run -e dev test --update-baselines` and review the JSON diff.

Each method runs as the plain loop a script writes (`plain_loop.py`), one point at a
time, and each result is mapped to the keys of the stored JSON to compare.
"""

import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import illustrative_example as ex
import numpy as np
import pytest
from oracle_helper import oracle_incumbents, oracle_states
from plain_loop import evaluate_points, hybrid_loop, standard_loop

import hybrid_bo as hb

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8

# Small, seeded runs so the test is fast and deterministic
SEED, N_ITERATIONS = 0, 2

# What a run of a method gives: the dataset, and each iteration's solution
Result = tuple[hb.Dataset, Sequence[tuple]]


def run_hybrid_bo(
    acquisition: Callable[[float], hb.EI | hb.SoftEI],
    measure: hb.Measure,
    rng: np.random.Generator,
) -> Result:
    """Hybrid BO: the initial points, the scenarios once, then the loop."""
    points = hb.lhs_points(ex.system.u, ex.N_INITIAL, rng)
    z = rng.standard_normal((25, ex.system.n_y))
    return hybrid_loop(
        ex.problem,
        measure,
        evaluate_points(ex.problem, measure, points, rng),
        z,
        np.full(25, 1 / 25),
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
        acquisition=acquisition,
        unknown_states_as_variables=False,
    )


def run_standard_bo(measure: hb.Measure, rng: np.random.Generator) -> Result:
    """Standard BO: the initial points, then the loop."""
    points = hb.lhs_points(ex.system.u, ex.N_INITIAL, rng)
    return standard_loop(
        ex.problem,
        measure,
        evaluate_points(ex.problem, measure, points, rng),
        rng,
        n_iterations=N_ITERATIONS,
        gp=ex.gp,
    )


def run_latin_hypercube_sampling(
    measure: hb.Measure, rng: np.random.Generator
) -> Result:
    """Latin hypercube sampling: the initial points and the design, then every point."""
    points = hb.lhs_points(ex.system.u, ex.N_INITIAL, rng)
    design = hb.lhs_points(ex.system.u, N_ITERATIONS, rng)
    all_points = np.concatenate([points, design])
    return evaluate_points(ex.problem, measure, all_points, rng), []


def run_uniform_sampling(measure: hb.Measure, rng: np.random.Generator) -> Result:
    """Uniform sampling: the initial points and the design, then every point."""
    points = hb.lhs_points(ex.system.u, ex.N_INITIAL, rng)
    design = hb.uniform_points(ex.system.u, N_ITERATIONS, rng)
    all_points = np.concatenate([points, design])
    return evaluate_points(ex.problem, measure, all_points, rng), []


# The illustrative example's methods, with their labels in the stored results
METHODS = {
    "hybrid_bo": partial(run_hybrid_bo, hb.EI),
    "hybrid_bo_soft": partial(
        run_hybrid_bo,
        partial(hb.SoftEI, temperature=50e-3),
    ),
    "standard_bo": run_standard_bo,
    "latin_hypercube_sampling": run_latin_hypercube_sampling,
    "uniform_sampling": run_uniform_sampling,
}


def as_baseline_columns(
    dataset: hb.Dataset,
    states: np.ndarray,
    solutions: Sequence[tuple],
) -> dict[str, np.ndarray]:
    """The run under the `ResultsBO` keys: they store y inside x, f as a column, and
    split every array into the initial points and the rest."""
    f_oracle = np.array(
        [
            ex.problem.f(row[:1], row[1:], u).full().item()
            for row, u in zip(states, dataset.u, strict=True)
        ]
    )
    columns = {
        "u": dataset.u,
        "x": np.column_stack([dataset.x, dataset.y]),
        "f": dataset.f[:, np.newaxis],
        "incumbents": hb.incumbents(dataset)[:, np.newaxis],
        "x_no_noise": states,
        "f_no_noise": f_oracle[:, np.newaxis],
        "incumbents_no_noise": oracle_incumbents(dataset.f, f_oracle)[:, np.newaxis],
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
        baseline["acquisition_values"] = np.array([s[-1] for s in solutions])
    scenario_solutions = [s for s in solutions if len(s) == 4]
    if scenario_solutions:
        baseline["acquisition_x"] = np.stack([s[1] for s in scenario_solutions])
        baseline["acquisition_y"] = np.stack([s[2] for s in scenario_solutions])
    return baseline


def run_method(method: str) -> dict[str, np.ndarray]:
    """Runs one run of a method as the experiment does and returns it under the
    `ResultsBO` keys."""
    # As in the experiment: one generator per run, initial points drawn from it
    rng = np.random.default_rng(SEED).spawn(1)[0]
    dataset, solutions = METHODS[method](ex.measure, rng)

    # The oracle values for the columns that score the run, from a generator of their own
    states = oracle_states(
        ex.problem, ex.oracle, dataset.u, np.random.default_rng(SEED)
    )
    return as_baseline_columns(dataset, states, solutions)


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
