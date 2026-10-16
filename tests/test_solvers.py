"""NLPs and the multistart solver."""

import logging

import casadi as cas
import numpy as np
import pytest

from hybrid_bo import NoSolutionError
from hybrid_bo.solvers import NLP, Ipopt, multistart

w = cas.SX.sym("w")
# Two local minima, near -1 (the global one) and near +1
DOUBLE_WELL = NLP(
    cas.Function("f", [w], [(w**2 - 1) ** 2 + 0.1 * w]),
    lower=np.array([-2.0]),
    upper=np.array([2.0]),
)


def test_best_start_wins():
    """With one start in each well, multistart returns the global minimum and its
    objective value."""
    solution = multistart(DOUBLE_WELL, np.array([[1.5], [-1.5]]), Ipopt())
    assert solution.w.shape == (1,)
    assert solution.w[0] < 0
    np.testing.assert_allclose(solution.f, DOUBLE_WELL.f(solution.w).full().item())


def test_one_start_finds_its_local_minimum():
    """A single start in the right well stays there, at the local minimum."""
    solution = multistart(DOUBLE_WELL, np.array([[1.5]]), Ipopt())
    assert solution.w[0] > 0


def test_bounds_and_constraints():
    """The solution respects both the variable bounds and the equality constraint."""
    # Minimize |v|^2 subject to v0 >= 0.7 and v0 + v1 = 1
    v = cas.SX.sym("v", 2)
    nlp = NLP(
        cas.Function("f", [v], [cas.sumsqr(v)]),
        lower=np.array([0.7, -np.inf]),
        upper=np.array([np.inf, np.inf]),
        g=cas.Function("g", [v], [v[0] + v[1] - 1]),
    )

    solution = multistart(nlp, np.array([[1.0, 1.0]]), Ipopt())

    np.testing.assert_allclose(solution.w, [0.7, 0.3], atol=1e-6)


def test_every_start_failing_raises_without_tracebacks(caplog):
    """When no start converges, multistart raises NoSolutionError and logs each
    failure without a traceback.

    `caplog` is a pytest fixture that captures log records during the test.
    """
    # w**2 + 1 = 0 has no real solution
    infeasible = NLP(
        DOUBLE_WELL.f,
        DOUBLE_WELL.lower,
        DOUBLE_WELL.upper,
        g=cas.Function("g", [w], [w**2 + 1]),
    )

    # `pytest.raises` fails the test unless the block raises that exception
    with caplog.at_level(logging.DEBUG), pytest.raises(NoSolutionError):
        multistart(infeasible, np.array([[0.5], [-0.5]]), Ipopt())

    assert caplog.records
    assert all(record.exc_info is None for record in caplog.records)
