"""Regression tests: each routine on the illustrative example vs stored results.

These lock in the *current* behavior, including the suspected bound bug in
`setup_acq_problem` in `legacy/routines/hybrid_bo.py` (see ROADMAP Known issues). If a PR
intentionally changes behavior, regenerate with
`pixi run -e dev test --update-baselines` and review the JSON diff.

Standard BO and the baselines run on the new API; their `Run` is mapped to the
legacy `ResultsBO` keys, so all four routines share one baseline format until #30.
"""

import functools
import json
import pickle
from dataclasses import fields
from pathlib import Path

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest

from hybrid_bo import (
    Run,
    StandardBOSettings,
    incumbents,
    initial_points,
    latin_hypercube_sampling,
    oracle_experiment,
    standard_bo,
    uniform_sampling,
)
from hybrid_bo.legacy.results import ResultsBO
from hybrid_bo.legacy.routines import do_hybrid_bo

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8
NEW_API = {
    "latin_hypercube_sampling": latin_hypercube_sampling,
    "uniform_sampling": uniform_sampling,
    # The legacy settings of the illustrative example are the defaults: EI, 10 starts
    # for the acquisition problem and the GP fit, and the default GP
    "standard_bo": functools.partial(standard_bo, settings=StandardBOSettings()),
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


def run_new_api(routine, config, problem):
    """Runs a method on the new API, as the legacy routine would, and returns its
    results under the `ResultsBO` keys."""
    # As in examples/main.py: one generator per run, initial points drawn from it
    rng = np.random.default_rng(config.seed).spawn(1)[0]
    u = initial_points(problem.system.u, 2, rng)
    experiment = oracle_experiment(
        problem.system,
        oracle=lambda v: [cas.sin(v.u)],
        noise_std=config.std_measurement_noise,
    )
    run = NEW_API[routine](problem, experiment, u, config.n_iterations_bo, rng=rng)
    return as_legacy_results(run)


def run_legacy(routine, config, results_dir):
    """Runs a legacy routine and returns the first run's `ResultsBO` fields."""
    run_routine(routine, config, results_dir)

    # Routines catch and print exceptions, so a crash shows up as missing results
    # or as NaN rows (result arrays are NaN-initialized and saved regardless)
    results_path = results_dir / "results_bo.pkl"
    assert results_path.exists(), "routine did not save results (check stderr)"
    with results_path.open("rb") as f:
        results = pickle.load(f)
    assert len(results) == config.n_runs_bo
    return {fld.name: getattr(results[0], fld.name) for fld in fields(ResultsBO)}


def run_routine(routine, config, results_dir):
    problem = ex.get_custom_problem(config)
    # As in examples/main.py: one generator per run, initial points drawn from it
    rngs = config.rng.spawn(config.n_runs_bo)
    u_initial = [ex.get_u_initial(problem, rng) for rng in rngs]
    match routine:
        case "hybrid_bo":
            gp = ex.get_gp_hybrid_bo(problem.n_u)
            do_hybrid_bo(
                problem,
                config,
                gp,
                u_initial,
                rngs,
                ex.get_optimizer_hybrid_bo(),
                results_dir,
                None,
            )


@pytest.mark.parametrize(
    "routine",
    ["hybrid_bo", "standard_bo", "latin_hypercube_sampling", "uniform_sampling"],
)
def test_illustrative_example(
    routine,
    illustrative_config,
    illustrative_problem,
    tmp_path,
    update_baselines,
):
    config = illustrative_config
    if routine in NEW_API:
        actual = run_new_api(routine, config, illustrative_problem)
    else:
        actual = run_legacy(routine, config, tmp_path)
    for name, value in actual.items():
        assert not np.isnan(value).any(), f"{name} has NaNs: routine failed mid-run"

    baseline_path = BASELINE_DIR / f"illustrative_{routine}.json"
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
