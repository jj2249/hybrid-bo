"""Regression tests: each routine on the illustrative example vs stored results.

These lock in the *current* behavior, including the suspected bound bug in
`setup_acq_problem` in `legacy/routines/hybrid_bo.py` (see ROADMAP Known issues). If a PR
intentionally changes behavior, regenerate with
`pixi run -e dev test --update-baselines` and review the JSON diff.
"""

import json
import pickle
from dataclasses import fields
from pathlib import Path

import illustrative_example as ex
import numpy as np
import pytest

from hybrid_bo.legacy.results import ResultsBO
from hybrid_bo.legacy.routines import (
    do_hybrid_bo,
    do_latin_hypercube_sampling,
    do_standard_bo,
    do_uniform_sampling,
)

BASELINE_DIR = Path(__file__).parent / "data" / "regression"
RTOL, ATOL = 1e-6, 1e-8


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
        case "standard_bo":
            gp = ex.get_gp_standard_bo(problem.n_u)
            do_standard_bo(
                problem,
                config,
                gp,
                u_initial,
                rngs,
                ex.get_optimizer_standard_bo(),
                results_dir,
                None,
            )
        case "latin_hypercube_sampling":
            do_latin_hypercube_sampling(
                problem, config, u_initial, rngs, results_dir, None
            )
        case "uniform_sampling":
            do_uniform_sampling(problem, config, u_initial, rngs, results_dir, None)


@pytest.mark.parametrize(
    "routine",
    ["hybrid_bo", "standard_bo", "latin_hypercube_sampling", "uniform_sampling"],
)
def test_illustrative_example(routine, illustrative_config, tmp_path, update_baselines):
    config = illustrative_config
    run_routine(routine, config, tmp_path)

    # Routines catch and print exceptions, so a crash shows up as missing results
    # or as NaN rows (result arrays are NaN-initialised and saved regardless)
    results_path = tmp_path / "results_bo.pkl"
    assert results_path.exists(), "routine did not save results (check stderr)"
    with results_path.open("rb") as f:
        results = pickle.load(f)
    assert len(results) == config.n_runs_bo
    actual = {fld.name: getattr(results[0], fld.name) for fld in fields(ResultsBO)}
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
