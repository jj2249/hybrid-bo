"""Regression tests: each method of the illustrative example vs stored results.

These lock in the *current* behavior. If a PR intentionally changes behavior,
regenerate with `pixi run -e dev test --update-baselines` and review the JSON diff.

The baselines were recorded by the pre-refactor code, so each `Run` is mapped to the
keys of its result files (`ResultsBO`) to compare. The acquisition problems' solutions
(#53) came later and have keys of their own.
"""

import json
from pathlib import Path

import illustrative_example as ex
import numpy as np
import pytest

from hybrid_bo import Run, incumbents, initial_points, oracle_incumbents

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8

# Small, seeded runs so the test is fast and deterministic
SEED, N_ITERATIONS = 0, 2


def as_baseline_columns(run: Run) -> dict[str, np.ndarray]:
    """The run under the `ResultsBO` keys: they store y inside x, f as a column, and
    split every array into the initial points and the rest."""
    history = run.history
    assert history.x_oracle is not None and history.y_oracle is not None
    assert history.f_oracle is not None
    columns = {
        "u": history.u,
        "x": np.column_stack([history.x, history.y]),
        "f": history.f[:, np.newaxis],
        "incumbents": incumbents(history)[:, np.newaxis],
        "x_no_noise": np.column_stack([history.x_oracle, history.y_oracle]),
        "f_no_noise": history.f_oracle[:, np.newaxis],
        "incumbents_no_noise": oracle_incumbents(history)[:, np.newaxis],
    }
    n = history.n_initial
    baseline = {
        f"{name}_{part}": value[:n] if part == "initial" else value[n:]
        for part in ("initial", "bo")
        for name, value in columns.items()
    }

    # The BO methods' acquisition solutions: the value per iteration, and hybrid BO's
    # scenario states (iteration, scenario, state)
    if run.acquisitions:
        baseline["acquisition_values"] = np.array([a.value for a in run.acquisitions])
    x = [a.x for a in run.acquisitions if a.x is not None]
    y = [a.y for a in run.acquisitions if a.y is not None]
    if x and y:
        baseline["acquisition_x"] = np.stack(x)
        baseline["acquisition_y"] = np.stack(y)
    return baseline


def run_method(method: str) -> dict[str, np.ndarray]:
    """Runs one run of a method as the experiment does and returns it under the
    `ResultsBO` keys."""
    # As in the experiment: one generator per run, initial points drawn from it
    rng = np.random.default_rng(SEED).spawn(1)[0]
    u = initial_points(ex.system.u, ex.configurations[method].n_initial, rng)
    run = ex.methods[method](ex.problem, ex.measure, u, N_ITERATIONS, rng=rng)
    return as_baseline_columns(run)


# `parametrize` runs the test once per method, each as its own test case. The other
# argument is a fixture from `conftest.py`, filled in by pytest
@pytest.mark.parametrize("method", list(ex.methods))
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
