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
    and keeps its artifact and history."""
    run = illustrative_run(illustrative_problem, method)

    loaded = _DataOnlyUnpickler(io.BytesIO(pickle.dumps(run))).load()

    assert loaded.artifact == run.artifact
    assert loaded.surrogates == []
    np.testing.assert_array_equal(loaded.history.u, run.history.u)
    np.testing.assert_array_equal(loaded.history.f, run.history.f)
    assert loaded.history.n_initial == run.history.n_initial


def test_artifact_describes_the_problem_method_and_versions(illustrative_problem):
    """The run's artifact holds, as strings, the equations, the measurement, the
    variables, the method and its settings, and the package versions."""
    run = illustrative_run(illustrative_problem, latin_hypercube_sampling)
    artifact = run.artifact

    # The problem and the measurement
    assert all(isinstance(value, str) for value in artifact.values())
    assert "exp(x)" in artifact["g"]
    assert "sin(((12*x)-4))" in artifact["f"]
    assert "sin(u)" in artifact["measure"]
    assert "lower=-2.0" in artifact["u"]
    assert artifact["h_inputs"] == "u"
    assert artifact["measured"] == "y"

    # The method and its settings
    assert artifact["method"] == "latin_hypercube_sampling"
    assert artifact["settings"] == "n_points=3"

    # The versions of the code that produced it
    for package in ("casadi", "numpy", "scipy"):
        assert artifact[package] == importlib.metadata.version(package)
    assert artifact["hybrid_bo"] == importlib.metadata.version("hybrid-bo")
    assert artifact["git"]


def test_artifact_holds_the_seed_and_spawn_key(illustrative_problem):
    """The artifact holds the seed and spawn key of the run's generator, so the run can
    be reproduced."""
    rng = np.random.default_rng(7).spawn(3)[2]

    artifact = illustrative_run(illustrative_problem, uniform_sampling, rng).artifact

    assert artifact["seed"] == "7"
    assert artifact["spawn_key"] == "(2,)"


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
    artifacts = [
        latin_hypercube_sampling(
            illustrative_problem,
            measure,
            np.array([[0.5]]),
            1,
            np.random.default_rng(0),
        ).artifact
        for measure in (plant, described_plant)
    ]

    assert artifacts[0]["measure"].endswith("plant")
    assert artifacts[1]["measure"] == "Pilot plant, sensor on y."


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
    assert run.artifact["method"] == method.__name__
