"""Regression tests: each method on the illustrative example vs stored results.

If a PR intentionally changes behavior, regenerate with
`pixi run -e dev test --update-baselines` and review the JSON diff.
"""

import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import illustrative_example as ex
import numpy as np
import pytest
from plain_loop import hybrid_loop, measure_points, standard_loop

import hybrid_bo as hb

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8

# Small, seeded runs so the test is fast and deterministic
SEED, N_ITERATIONS = 0, 2

# What a run of a method gives: the dataset, and each iteration's solution
Result = tuple[hb.Dataset, Sequence[hb.Solution]]


def run_hybrid_bo(
    objective: Callable[[float], hb.obj.EI | hb.obj.SoftEI], rng: np.random.Generator
) -> Result:
    """Hybrid BO: the initial points, the scenarios once, then the loop."""
    points = hb.lhs_points([ex.u], ex.N_INITIAL, rng)
    saa = hb.SAA(rng.standard_normal((25, 1)))
    return hybrid_loop(
        measure_points(points, rng),
        saa,
        rng,
        n_iterations=N_ITERATIONS,
        objective=objective,
    )


def run_standard_bo(rng: np.random.Generator) -> Result:
    """Standard BO: the initial points, then the loop."""
    points = hb.lhs_points([ex.u], ex.N_INITIAL, rng)
    return standard_loop(measure_points(points, rng), rng, n_iterations=N_ITERATIONS)


def run_latin_hypercube_sampling(rng: np.random.Generator) -> Result:
    """Latin hypercube sampling: the initial points and the design, then every point."""
    points = hb.lhs_points([ex.u], ex.N_INITIAL, rng)
    design = hb.lhs_points([ex.u], N_ITERATIONS, rng)
    return measure_points(np.concatenate([points, design]), rng), []


METHODS = {
    "hybrid_bo": partial(run_hybrid_bo, hb.obj.EI),
    "hybrid_bo_soft": partial(run_hybrid_bo, partial(hb.obj.SoftEI, temperature=50e-3)),
    "standard_bo": run_standard_bo,
    "latin_hypercube_sampling": run_latin_hypercube_sampling,
}


def as_baseline_columns(
    dataset: hb.Dataset, solutions: Sequence[hb.Solution]
) -> dict[str, np.ndarray]:
    """The run under the stored keys: y inside x, f as a column, and every array split
    into the initial points and the rest."""
    columns = {
        "u": dataset.u,
        "x": np.column_stack([dataset.x, dataset.y]),
        "f": dataset.f[:, np.newaxis],
        "incumbents": dataset.incumbents[:, np.newaxis],
    }
    n = ex.N_INITIAL
    baseline = {
        f"{name}_{part}": value[:n] if part == "initial" else value[n:]
        for part in ("initial", "bo")
        for name, value in columns.items()
    }

    # The BO methods' solutions: the objective per iteration, and hybrid BO's scenario
    # states (iteration, scenario, state)
    if solutions:
        baseline["acquisition_values"] = np.array([s.value for s in solutions])
    if solutions and solutions[0].x.shape[1]:
        baseline["acquisition_x"] = np.stack([s.x for s in solutions])
        baseline["acquisition_y"] = np.stack([s.y for s in solutions])
    return baseline


def run_method(method: str) -> dict[str, np.ndarray]:
    """One seeded run of a method, under the stored keys."""
    rng = np.random.default_rng(SEED).spawn(1)[0]
    dataset, solutions = METHODS[method](rng)
    return as_baseline_columns(dataset, solutions)


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
