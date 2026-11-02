"""The S2MPJ benchmarks (`benchmarks/s2mpj.py`): the traced functions against S2MPJ's
float evaluation, the rules that turn a problem into a hybrid one, and the problems
the rules refuse.

Most tests read the S2MPJ checkout that `pixi run benchmark` fetches into
`benchmarks/.s2mpj/`, and are skipped without it, so the test suite passes without
S2MPJ.
"""

import functools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import s2mpj
from fetch_s2mpj import CHECKOUT

from hybrid_bo.evaluation import evaluate
from hybrid_bo.sampling import lhs

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"

# `skipif` skips the tests it marks when its condition is true, giving the reason
needs_checkout = pytest.mark.skipif(
    not (CHECKOUT / "s2mpjlib.py").exists(),
    reason="no S2MPJ checkout: run pixi run -e benchmarks fetch-s2mpj",
)


# The traced functions


# `parametrize` runs the test once per value in the list, each as its own test case.
# HS26 and HS46 write into float arrays, so they trace only with the patch
@needs_checkout
@pytest.mark.parametrize("name", ["HS27", "HS6", "HS26", "HS39", "HS46", "HS56", "BT9"])
def test_traced_functions_match_the_float_evaluation(name):
    """f and c traced with CasADi symbols equal S2MPJ's own float evaluation at random
    points around S2MPJ's start."""
    # Trace, and draw points within 2 of the start
    problem = s2mpj.load(name)
    f, c = s2mpj.traced(problem)
    start = problem.data.x0.ravel()
    points = lhs(start - 2, start + 2, 10, np.random.default_rng(0))

    # Both functions agree with S2MPJ's at every point
    for point in points:
        np.testing.assert_allclose(
            float(f(point)), problem.data.fx(point.copy()), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            c(point).full().ravel(),
            np.ravel(problem.data.cx(point.copy())),
            rtol=1e-10,
            atol=1e-10,
        )


@needs_checkout
def test_the_trace_check_refuses_a_wrong_trace():
    """`check_trace` raises when a traced function differs from S2MPJ's evaluation,
    as an unpatched trace would, with NaN where a symbol was converted to a float."""
    problem = s2mpj.load("HS27")
    f, c = s2mpj.traced(problem)
    wrong_f = type(f)("f", *_nan_plus(f))

    # `raises` passes only if the block raises the exception given
    with pytest.raises(s2mpj.UnsupportedProblemError) as error:
        s2mpj.check_trace(problem, wrong_f, c, problem.data.x0.reshape(1, -1))
    assert error.value.reason == s2mpj.Reason.UNTRACEABLE


def _nan_plus(function):
    """The inputs and outputs of `function` plus NaN, as `cas.Function` takes them."""
    import casadi as cas

    z = cas.SX.sym("z", function.size1_in(0))
    return [z], [function(z) + np.nan]


# The rules


@needs_checkout
def test_hs27_splits_as_the_issue_works_it_through():
    """HS27, min 0.01(z1 - 1)^2 + (z2 - z1^2)^2 s.t. z1 + z3^2 + 1 = 0: u = (z2, z3),
    x = z1, y = z3^2 (CON1's element), g = z1 + 1 + y, h_inputs = z3, the box
    z2 in [0, 3], z3 in [-1, 3], and f* = 0.04, which IPOPT agrees with."""
    rules = s2mpj.split(s2mpj.load("HS27"))

    assert rules.decisions == (1, 2)
    assert rules.states == (0,)
    assert rules.hidden_name == "CON1"
    assert rules.h_inputs == (2,)
    np.testing.assert_array_equal(rules.lower[1:], [0.0, -1.0])
    np.testing.assert_array_equal(rules.upper[1:], [3.0, 3.0])
    assert rules.optimum == 0.04
    assert rules.ipopt_optimum == pytest.approx(0.04, abs=1e-8)
    np.testing.assert_allclose(rules.solution, [-1.0, 1.0, 0.0], atol=1e-6)


@needs_checkout
def test_the_hs27_benchmark_is_hs27():
    """The HS27 benchmark's equations, measured at the solution's u = (1, 0), give
    y = 0, x = -1 and f = f* = 0.04; its budget is 10d = 20 evaluations, 2d + 1 = 5
    of them initial, with 25d = 50 scenarios and a temperature of 1e-3 f*."""
    # The experiment, and one evaluation at the solution's decision variables
    benchmark = s2mpj.benchmark_experiment("HS27")
    history = evaluate(
        benchmark.problem,
        benchmark.measure,
        np.array([[1.0, 0.0]]),
        np.random.default_rng(0),
    )

    # The oracle's root is HS27's solution
    np.testing.assert_allclose(history.y, [[0.0]], atol=1e-8)
    np.testing.assert_allclose(history.x, [[-1.0]], atol=1e-8)
    np.testing.assert_allclose(history.f, [0.04], atol=1e-8)

    # The budget and the settings the rules set, and what the notes record
    assert set(benchmark.configurations) == {
        "hybrid_bo",
        "hybrid_bo_soft",
        "standard_bo",
        "latin_hypercube_sampling",
    }
    for configuration in benchmark.configurations.values():
        assert (configuration.n_evaluations, configuration.n_initial) == (20, 5)
    method = benchmark.configurations["hybrid_bo_soft"].method
    assert isinstance(method, functools.partial)
    settings = method.keywords["settings"]
    assert settings.approximation.n_scenarios == 50
    assert settings.approximation.temperature == pytest.approx(4e-5)
    assert benchmark.optimum == 0.04
    assert benchmark.notes["problem"] == "HS27"
    assert benchmark.notes["hidden_constraint"] == "CON1"
    assert benchmark.notes["s2mpj_commit"] == s2mpj.COMMIT


@needs_checkout
@pytest.mark.parametrize("name", ["HS39", "HS61", "BT9"])
def test_the_stated_optimum_agrees_with_ipopt(name):
    """Every problem the rules accept states an f* that IPOPT reaches from S2MPJ's
    start."""
    rules = s2mpj.split(s2mpj.load(name))

    assert rules.ipopt_optimum == pytest.approx(rules.optimum, rel=1e-5, abs=1e-6)


def test_states_are_picked_by_pivoted_qr_with_ties_outside_the_hidden_term():
    """The states are the columns a column-pivoted QR picks first, and of two equal
    columns the one outside the hidden term."""
    # z2's column is largest, so it is picked; z1 and z3 are equal
    jacobian = np.array([[1.0, 3.0, 1.0]])
    assert s2mpj.choose_states(jacobian, hidden=()) == (1,)
    jacobian = np.array([[1.0, 0.0, 1.0]])
    assert s2mpj.choose_states(jacobian, hidden=(0,)) == (2,)
    assert s2mpj.choose_states(jacobian, hidden=(2,)) == (0,)


def test_the_box_widens_the_hull_of_start_and_solution():
    """Each infinite bound is the hull's edge widened by max(1, half its width);
    finite bounds are kept."""
    lower, upper = s2mpj.box(
        np.array([-np.inf, -np.inf, 0.5]),
        np.array([np.inf, np.inf, np.inf]),
        start=np.array([2.0, 10.0, 2.0]),
        solution=np.array([1.0, 0.0, 1.0]),
    )

    np.testing.assert_array_equal(lower, [0.0, -5.0, 0.5])
    np.testing.assert_array_equal(upper, [3.0, 15.0, 3.0])


def test_the_hidden_constraint_is_the_same_in_every_process():
    """The pick is seeded from sha256 of the name, so new Python processes with
    different string hash seeds pick what this one picks."""
    code = "import s2mpj; print(s2mpj.pick_hidden('HS27', list(range(1000))))"
    picks = {
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=BENCHMARKS_DIR,
            env=os.environ | {"PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for hash_seed in ("1", "2", "3")
    }

    assert picks == {str(s2mpj.pick_hidden("HS27", list(range(1000))))}


# Unsupported problems


@needs_checkout
@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("HS21", s2mpj.Reason.INEQUALITIES),
        ("HS8", s2mpj.Reason.NO_DEGREES_OF_FREEDOM),
        ("HS28", s2mpj.Reason.ONLY_LINEAR_CONSTRAINTS),
        ("TRYmB", s2mpj.Reason.NO_SOLUTION),
        ("MARATOS", s2mpj.Reason.IPOPT_DISAGREES),
        ("HS7", s2mpj.Reason.ILL_CONDITIONED),
        ("HS87", s2mpj.Reason.UNTRACEABLE),
    ],
)
def test_an_unsupported_problem_raises_naming_the_reason(name, reason):
    """A problem outside the rules raises `UnsupportedProblemError` with the reason,
    which its message names."""
    with pytest.raises(s2mpj.UnsupportedProblemError) as error:
        s2mpj.split(s2mpj.load(name))

    assert error.value.reason == reason
    assert str(reason) in str(error.value)


@needs_checkout
def test_an_unknown_problem_is_a_value_error():
    """A name S2MPJ has no problem for, or that isn't a plain name, raises
    `ValueError`."""
    for name in ("NOSUCHPROBLEM", "../s2mpjlib"):
        with pytest.raises(ValueError, match="S2MPJ has no problem"):
            s2mpj.load(name)
