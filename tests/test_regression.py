"""Regression tests: each method on the illustrative example vs stored results.

These lock in the *current* behavior. If a PR intentionally changes behavior,
regenerate with `pixi run -e dev test --update-baselines` and review the JSON diff.

Every method runs on the new API; its `Run` is mapped to the legacy `ResultsBO` keys,
so the baselines keep the format they were recorded in until #30.
"""

import functools
import json
from pathlib import Path

import casadi as cas
import numpy as np
import pytest

from hybrid_bo import (
    HybridBOSettings,
    Run,
    StandardBOSettings,
    hybrid_bo,
    incumbents,
    initial_points,
    latin_hypercube_sampling,
    oracle_experiment,
    standard_bo,
    uniform_sampling,
)

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8
METHODS = {
    # The legacy settings of the illustrative example are the defaults (EI, 25
    # scenarios, 10 starts for the acquisition problem and the GP fit, the default
    # GP), except that the unknown states are substituted, not variables
    "hybrid_bo": functools.partial(
        hybrid_bo, settings=HybridBOSettings(unknown_states_as_variables=False)
    ),
    "standard_bo": functools.partial(standard_bo, settings=StandardBOSettings()),
    "latin_hypercube_sampling": latin_hypercube_sampling,
    "uniform_sampling": uniform_sampling,
}


def as_legacy_results(run: Run) -> dict[str, np.ndarray]:
    """The run under the `ResultsBO` keys: legacy stores y inside x, f as a column,
    and splits every array into the initial points and the rest."""
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
        "incumbents_no_noise": np.minimum.accumulate(history.f_oracle)[:, np.newaxis],
    }
    n = history.n_initial
    return {
        f"{name}_{part}": value[:n] if part == "initial" else value[n:]
        for part in ("initial", "bo")
        for name, value in columns.items()
    }


def run_method(method, config, problem):
    """Runs a method as the legacy routine did and returns its results under the
    `ResultsBO` keys."""
    # As in examples/main.py: one generator per run, initial points drawn from it
    rng = np.random.default_rng(config.seed).spawn(1)[0]
    u = initial_points(problem.system.u, 2, rng)
    experiment = oracle_experiment(
        problem.system,
        oracle=lambda v: [cas.sin(v.u)],
        noise_std=config.std_measurement_noise,
    )
    run = METHODS[method](problem, experiment, u, config.n_iterations_bo, rng=rng)
    return as_legacy_results(run)


# `parametrize` runs the test once per method, each as its own test case. The other
# arguments are fixtures from `conftest.py`, filled in by pytest
@pytest.mark.parametrize("method", list(METHODS))
def test_illustrative_example(
    method,
    illustrative_config,
    illustrative_problem,
    update_baselines,
):
    """The method reproduces its stored results on the illustrative example."""
    actual = run_method(method, illustrative_config, illustrative_problem)

    baseline_path = BASELINE_DIR / f"illustrative_{method}.json"
    if update_baselines:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: np.asarray(v).tolist() for k, v in actual.items()}
        baseline_path.write_text(json.dumps(data, indent=1) + "\n")
        pytest.skip(f"updated {baseline_path.name}")

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
