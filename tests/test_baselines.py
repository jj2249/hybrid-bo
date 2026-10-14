"""The Latin hypercube and uniform baselines, and the Run they return."""

import importlib.metadata
import io
import pickle

import casadi as cas
import numpy as np
import pytest

import hybrid_bo
from hybrid_bo import (
    RootFindingError,
    Run,
    initial_points,
    latin_hypercube_sampling,
    oracle_experiment,
    uniform_sampling,
)

METHODS = [latin_hypercube_sampling, uniform_sampling]


def illustrative_run(problem, method, rng=None) -> Run:
    rng = np.random.default_rng(0) if rng is None else rng
    experiment = oracle_experiment(problem.system, oracle=lambda v: [cas.sin(v.u)])
    u = initial_points(problem.system.u, 2, rng)
    return method(problem, experiment, u, 3, rng)


class _DataOnlyUnpickler(pickle.Unpickler):
    """Loads only the library's data classes, numpy and builtins: no user code."""

    allowed = ("hybrid_bo.run", "hybrid_bo.history", "numpy", "builtins")

    def find_class(self, module, name):
        if not module.startswith(self.allowed):
            msg = f"the pickle needs {module}.{name}"
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)


@pytest.mark.parametrize("method", METHODS)
def test_initial_points_come_first(illustrative_problem, method):
    rng = np.random.default_rng(0)
    u = initial_points(illustrative_problem.system.u, 2, rng)
    experiment = oracle_experiment(
        illustrative_problem.system, oracle=lambda v: [cas.sin(v.u)]
    )

    run = method(illustrative_problem, experiment, u, 3, rng)

    assert run.history.n_initial == 2
    assert run.history.u.shape == (5, 1)
    np.testing.assert_array_equal(run.history.u[:2], u)
    assert np.all((-2.0 <= run.history.u) & (run.history.u <= 2.0))


@pytest.mark.parametrize("method", METHODS)
def test_run_pickles_without_the_code_that_produced_it(illustrative_problem, method):
    run = illustrative_run(illustrative_problem, method)

    loaded = _DataOnlyUnpickler(io.BytesIO(pickle.dumps(run))).load()

    assert loaded.record == run.record
    np.testing.assert_array_equal(loaded.history.u, run.history.u)
    np.testing.assert_array_equal(loaded.history.f, run.history.f)
    assert loaded.history.n_initial == run.history.n_initial


def test_record_describes_the_problem_method_and_versions(illustrative_problem):
    run = illustrative_run(illustrative_problem, latin_hypercube_sampling)
    record = run.record

    assert all(isinstance(value, str) for value in record.values())
    assert "exp(x)" in record["g"]
    assert "sin(((12*x)-4))" in record["f"]
    assert "sin(u)" in record["experiment"]
    assert "lower=-2.0" in record["u"]
    assert record["h_inputs"] == "u"
    assert record["measured"] == "y"
    assert record["method"] == "latin_hypercube_sampling"
    assert record["settings"] == "n_points=3"
    for package in ("casadi", "numpy", "scipy"):
        assert record[package] == importlib.metadata.version(package)
    assert record["hybrid_bo"] == importlib.metadata.version("hybrid-bo")
    assert record["git"]


def test_record_holds_the_seed_and_spawn_key(illustrative_problem):
    rng = np.random.default_rng(7).spawn(3)[2]

    record = illustrative_run(illustrative_problem, uniform_sampling, rng).record

    assert record["seed"] == "7"
    assert record["spawn_key"] == "(2,)"


def test_custom_experiment_is_recorded_by_docstring_or_name(illustrative_problem):
    def plant(u, rng):
        return np.sin(u)

    def described_plant(u, rng):
        """Pilot plant, sensor on y."""
        return np.sin(u)

    records = [
        latin_hypercube_sampling(
            illustrative_problem,
            experiment,
            np.array([[0.5]]),
            1,
            np.random.default_rng(0),
        ).record
        for experiment in (plant, described_plant)
    ]

    assert records[0]["experiment"].endswith("plant")
    assert records[1]["experiment"] == "Pilot plant, sensor on y."


@pytest.mark.parametrize("method", METHODS)
def test_failed_evaluation_carries_the_partial_run(illustrative_problem, method):
    # The third evaluation (the first design point) measures NaN, so the
    # root-finding problem has no root
    calls = []

    def experiment(u, rng):
        calls.append(u)
        return np.array([0.5]) if len(calls) < 3 else np.array([np.nan])

    with pytest.raises(RootFindingError) as info:
        method(
            illustrative_problem,
            experiment,
            np.array([[-0.5], [0.5]]),
            2,
            np.random.default_rng(0),
        )

    run = info.value.run
    assert isinstance(run, hybrid_bo.Run)
    assert run.history.u.shape == (2, 1)
    assert run.history.n_initial == 2
    assert run.record["method"] == method.__name__
