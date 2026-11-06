"""The Latin hypercube and uniform baselines, and the Run they return.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

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
    measure = oracle_measure(
        problem.system, oracle=lambda v: [cas.sin(v.u)], noise_std=0.0
    )
    u = initial_points(problem.system.u, 2, rng)
    return method(problem, measure, initial_points=u, n_iterations=3, rng=rng)


class _DataOnlyUnpickler(pickle.Unpickler):
    """Loads only the library's data classes, numpy and builtins: no user code."""

    allowed = ("hybrid_bo.run", "hybrid_bo.dataset", "numpy", "builtins")

    def find_class(self, module, name):
        """The class `module.name`, if the module is allowed."""
        if not module.startswith(self.allowed):
            msg = f"the pickle needs {module}.{name}"
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize("method", METHODS)
def test_initial_points_come_first(illustrative_problem, method):
    """The dataset starts with the given initial points, followed by the method's
    points, all inside the bounds of u."""
    rng = np.random.default_rng(0)
    u = initial_points(illustrative_problem.system.u, 2, rng)
    measure = oracle_measure(
        illustrative_problem.system,
        oracle=lambda v: [cas.sin(v.u)],
        noise_std=0.0,
    )

    run = method(
        illustrative_problem, measure, initial_points=u, n_iterations=3, rng=rng
    )

    assert run.dataset.n_initial == 2
    assert run.dataset.u.shape == (5, 1)
    np.testing.assert_array_equal(run.dataset.u[:2], u)
    assert np.all((-2.0 <= run.dataset.u) & (run.dataset.u <= 2.0))


@pytest.mark.parametrize("method", METHODS)
def test_run_pickles_without_the_code_that_produced_it(illustrative_problem, method):
    """A pickled run loads with only the library's data classes, numpy and builtins,
    and keeps its method and dataset."""
    run = illustrative_run(illustrative_problem, method)

    loaded = _DataOnlyUnpickler(io.BytesIO(pickle.dumps(run))).load()

    assert loaded.method == run.method
    assert loaded.surrogates == []
    np.testing.assert_array_equal(loaded.dataset.u, run.dataset.u)
    np.testing.assert_array_equal(loaded.dataset.f, run.dataset.f)
    assert loaded.dataset.n_initial == run.dataset.n_initial


@pytest.mark.parametrize("method", METHODS)
def test_failed_evaluation_carries_the_partial_run(illustrative_problem, method):
    """When an evaluation fails, the error carries the run up to the point before."""
    # The third evaluation (the first design point) measures NaN, which is rejected
    # before the root-finding problem
    calls = []

    def measure(u, rng):
        calls.append(u)
        return np.array([0.5]) if len(calls) < 3 else np.array([np.nan])

    # `pytest.raises` fails the test unless the block raises that exception; `info`
    # holds the exception
    with pytest.raises(RootFindingError, match="isn't finite") as info:
        method(
            illustrative_problem,
            measure,
            initial_points=np.array([[-0.5], [0.5]]),
            n_iterations=2,
            rng=np.random.default_rng(0),
        )

    # The run holds the two initial points
    run = info.value.run
    assert isinstance(run, hybrid_bo.Run)
    assert run.dataset.u.shape == (2, 1)
    assert run.dataset.n_initial == 2
    assert run.method == method.__name__
