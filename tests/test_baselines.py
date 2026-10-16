"""The Latin hypercube and uniform baselines, and the Run they return.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

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
    oracle_measure,
    uniform_sampling,
)

METHODS = [latin_hypercube_sampling, uniform_sampling]


def illustrative_run(problem, method, rng=None) -> Run:
    """A run of `method` on the illustrative example: 2 initial points, then 3."""
    rng = np.random.default_rng(0) if rng is None else rng
    measure = oracle_measure(problem.system, oracle=lambda v: [cas.sin(v.u)])
    u = initial_points(problem.system.u, 2, rng)
    return method(problem, measure, u, 3, rng)


class _DataOnlyUnpickler(pickle.Unpickler):
    """Loads only the library's data classes, numpy and builtins: no user code."""

    allowed = ("hybrid_bo.run", "hybrid_bo.history", "numpy", "builtins")

    def find_class(self, module, name):
        """The class `module.name`, if the module is allowed."""
        if not module.startswith(self.allowed):
            msg = f"the pickle needs {module}.{name}"
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize("method", METHODS)
def test_initial_points_come_first(illustrative_problem, method):
    """The history starts with the given initial points, followed by the method's
    points, all inside the bounds of u."""
    rng = np.random.default_rng(0)
    u = initial_points(illustrative_problem.system.u, 2, rng)
    measure = oracle_measure(
        illustrative_problem.system, oracle=lambda v: [cas.sin(v.u)]
    )

    run = method(illustrative_problem, measure, u, 3, rng)

    assert run.history.n_initial == 2
    assert run.history.u.shape == (5, 1)
    np.testing.assert_array_equal(run.history.u[:2], u)
    assert np.all((-2.0 <= run.history.u) & (run.history.u <= 2.0))


@pytest.mark.parametrize("method", METHODS)
def test_run_pickles_without_the_code_that_produced_it(illustrative_problem, method):
    """A pickled run loads with only the library's data classes, numpy and builtins,
    and keeps its record and history."""
    run = illustrative_run(illustrative_problem, method)

    loaded = _DataOnlyUnpickler(io.BytesIO(pickle.dumps(run))).load()

    assert loaded.record == run.record
    assert loaded.surrogates == []
    np.testing.assert_array_equal(loaded.history.u, run.history.u)
    np.testing.assert_array_equal(loaded.history.f, run.history.f)
    assert loaded.history.n_initial == run.history.n_initial


def test_record_describes_the_problem_method_and_versions(illustrative_problem):
    """The run's record holds, as strings, the equations, the measurement, the
    variables, the method and its settings, and the package versions."""
    run = illustrative_run(illustrative_problem, latin_hypercube_sampling)
    record = run.record

    # The problem and the measurement
    assert all(isinstance(value, str) for value in record.values())
    assert "exp(x)" in record["g"]
    assert "sin(((12*x)-4))" in record["f"]
    assert "sin(u)" in record["measure"]
    assert "lower=-2.0" in record["u"]
    assert record["h_inputs"] == "u"
    assert record["measured"] == "y"

    # The method and its settings
    assert record["method"] == "latin_hypercube_sampling"
    assert record["settings"] == "n_points=3"

    # The versions of the code that produced it
    for package in ("casadi", "numpy", "scipy"):
        assert record[package] == importlib.metadata.version(package)
    assert record["hybrid_bo"] == importlib.metadata.version("hybrid-bo")
    assert record["git"]


def test_record_holds_the_seed_and_spawn_key(illustrative_problem):
    """The record holds the seed and spawn key of the run's generator, so the run can
    be reproduced."""
    rng = np.random.default_rng(7).spawn(3)[2]

    record = illustrative_run(illustrative_problem, uniform_sampling, rng).record

    assert record["seed"] == "7"
    assert record["spawn_key"] == "(2,)"


def test_custom_measure_is_recorded_by_docstring_or_name(illustrative_problem):
    """A custom measurement is recorded by its docstring, or by its name when it has
    none."""

    # Two measurements: `plant` has no docstring on purpose
    def plant(u, rng):
        return np.sin(u)

    def described_plant(u, rng):
        """Pilot plant, sensor on y."""
        return np.sin(u)

    # One run with each
    records = [
        latin_hypercube_sampling(
            illustrative_problem,
            measure,
            np.array([[0.5]]),
            1,
            np.random.default_rng(0),
        ).record
        for measure in (plant, described_plant)
    ]

    assert records[0]["measure"].endswith("plant")
    assert records[1]["measure"] == "Pilot plant, sensor on y."


@pytest.mark.parametrize("method", METHODS)
def test_failed_evaluation_carries_the_partial_run(illustrative_problem, method):
    """When an evaluation fails, the error carries the run up to the point before."""
    # The third evaluation (the first design point) measures NaN, so the
    # root-finding problem has no root
    calls = []

    def measure(u, rng):
        calls.append(u)
        return np.array([0.5]) if len(calls) < 3 else np.array([np.nan])

    # `pytest.raises` fails the test unless the block raises that exception; `info`
    # holds the exception
    with pytest.raises(RootFindingError) as info:
        method(
            illustrative_problem,
            measure,
            np.array([[-0.5], [0.5]]),
            2,
            np.random.default_rng(0),
        )

    # The run holds the two initial points
    run = info.value.run
    assert isinstance(run, hybrid_bo.Run)
    assert run.history.u.shape == (2, 1)
    assert run.history.n_initial == 2
    assert run.record["method"] == method.__name__
