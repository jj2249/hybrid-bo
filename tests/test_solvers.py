"""NLPs and the multistart solver."""

import logging

import casadi as cas
import numpy as np
import pytest

from hybrid_bo import NoSolutionError, RootFindingError
from hybrid_bo.solvers import (
    NLP,
    Ipopt,
    RootProblem,
)

w = cas.SX.sym("w")
# Two local minima, near -1 (the global one) and near +1
DOUBLE_WELL = NLP(
    cas.Function("f", [w], [(w**2 - 1) ** 2 + 0.1 * w]),
    lower=np.array([-2.0]),
    upper=np.array([2.0]),
)


def test_best_start_wins():
    """With one start in each well, `NLP.solve_from` returns the global minimum
    and its objective value."""
    w, f = DOUBLE_WELL.solve_from(np.array([[1.5], [-1.5]]), Ipopt())
    assert w.shape == (1,)
    assert w[0] < 0
    np.testing.assert_allclose(f, DOUBLE_WELL.f(w).full().item())


def test_a_tie_keeps_the_earlier_start():
    """When two starts reach minima whose objectives differ only by rounding,
    `NLP.solve_from` keeps the earlier start, so the result doesn't depend on the
    last bits of the platform's arithmetic (the illustrative example's standard BO has
    such a tie, which macOS and Windows broke differently)."""
    # Two wells, at -1 and +1, the later one lower by 2e-14
    tied = NLP(
        cas.Function("f", [w], [(w**2 - 1) ** 2 - 1e-14 * w]),
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )

    best, _ = tied.solve_from(np.array([[-1.5], [1.5]]), Ipopt())

    assert best[0] < 0


def test_one_start_finds_its_local_minimum():
    """A single start in the right well stays there, at the local minimum."""
    w, _ = DOUBLE_WELL.solve_from(np.array([[1.5]]), Ipopt())
    assert w[0] > 0


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

    w, _ = nlp.solve_from(np.array([[1.0, 1.0]]), Ipopt())

    np.testing.assert_allclose(w, [0.7, 0.3], atol=1e-6)


def test_an_empty_constraint_is_no_constraint():
    """An equality constraint of no rows (the known equations of a system without
    mechanistic states) leaves the problem unconstrained: the double well's global
    minimum."""
    nlp = NLP(
        DOUBLE_WELL.f,
        DOUBLE_WELL.lower,
        DOUBLE_WELL.upper,
        g=cas.Function("g", [w], [cas.SX(0, 1)]),
    )

    best, _ = nlp.solve_from(np.array([[-1.5]]), Ipopt())

    reference, _ = DOUBLE_WELL.solve_from(np.array([[-1.5]]), Ipopt())
    np.testing.assert_allclose(best, reference, atol=1e-8)


def test_every_start_failing_raises():
    """When no start converges, `NLP.solve_from` raises NoSolutionError."""
    # w**2 + 1 = 0 has no real solution
    infeasible = NLP(
        DOUBLE_WELL.f,
        DOUBLE_WELL.lower,
        DOUBLE_WELL.upper,
        g=cas.Function("g", [w], [w**2 + 1]),
    )

    # `pytest.raises` fails the test unless the block raises that exception
    with pytest.raises(NoSolutionError):
        infeasible.solve_from(np.array([[0.5], [-0.5]]), Ipopt())


def test_starts_at_the_iteration_limit_are_warned_about(caplog):
    """When starts run IPOPT to its iteration limit, `NLP.solve_from` logs one
    warning giving how many of the starts did and the name of the NLP's objective, even
    though another start succeeded."""
    # The Rosenbrock function, which five iterations solve only from its minimum
    v = cas.SX.sym("v", 2)
    rosenbrock = NLP(
        cas.Function(
            "rosenbrock", [v], [(1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2]
        ),
        lower=np.array([-5.0, -5.0]),
        upper=np.array([5.0, 5.0]),
    )
    five_iterations = Ipopt(solver_options={"print_level": 0, "max_iter": 5})

    starts = np.array([[-1.5, 2.0], [-1.2, 1.0], [1.0, 1.0]])

    # Solve, recording every log message
    with caplog.at_level(logging.DEBUG):
        rosenbrock.solve_from(starts, five_iterations)

    # One warning, counting the two starts away from the minimum
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "2 of 3 starts" in warnings[0].getMessage()
    assert "rosenbrock" in warnings[0].getMessage()


def test_other_failures_are_not_warned_about(caplog):
    """A start that fails for a reason other than the iteration limit (here an
    infeasible constraint) logs nothing."""
    infeasible = NLP(
        DOUBLE_WELL.f,
        DOUBLE_WELL.lower,
        DOUBLE_WELL.upper,
        g=cas.Function("g", [w], [w**2 + 1]),
    )

    with pytest.raises(NoSolutionError):
        infeasible.solve_from(np.array([[0.5], [-0.5]]), Ipopt())

    assert not caplog.records


# x^2 = p, with x a variable and p a parameter
ROOT_W, ROOT_P = cas.SX.sym("x"), cas.SX.sym("p")
SQUARE = cas.Function("square", [ROOT_W, ROOT_P], [ROOT_W**2 - ROOT_P])


def test_root_problem_gives_the_root_at_the_parameters():
    """The same problem is solved at each parameter, from the start's side."""
    problem = RootProblem(SQUARE, np.array([-10.0]), np.array([10.0]))

    np.testing.assert_allclose(
        problem.solve(np.array([4.0]), np.array([[1.0]]), 1e-10), [2.0]
    )
    np.testing.assert_allclose(
        problem.solve(np.array([9.0]), np.array([[-1.0]]), 1e-10), [-3.0]
    )


def test_root_problem_skips_a_root_outside_the_bounds():
    """A start that reaches a root outside the bounds fails, and the next start is used."""
    problem = RootProblem(SQUARE, np.array([0.0]), np.array([10.0]))
    starts = np.array([[-1.0], [1.0]])

    root = problem.solve(np.array([4.0]), starts, 1e-10)

    np.testing.assert_allclose(root, [2.0])


def test_root_problem_raises_when_no_start_gives_a_root():
    """x^2 = -1 has no real root."""
    problem = RootProblem(SQUARE, np.array([-10.0]), np.array([10.0]))

    with pytest.raises(RootFindingError, match="from 2 starts"):
        problem.solve(np.array([-1.0]), np.array([[1.0], [2.0]]), 1e-10)


def test_root_problem_with_no_variables_is_empty():
    """With no unknowns nothing is solved."""
    problem = RootProblem(SQUARE, np.zeros(0), np.zeros(0))

    assert problem.solve(np.array([1.0]), np.zeros((1, 0)), 1e-10).shape == (0,)


def test_root_problem_is_quiet_when_every_start_diverges(capfd):
    """A start that fails is expected, so it prints nothing, even through infinity."""
    w, p = cas.SX.sym("w"), cas.SX.sym("p")
    residual = cas.Function("residual", [w, p], [cas.exp(w) - p])
    problem = RootProblem(residual, np.array([-np.inf]), np.array([np.inf]))

    with pytest.raises(RootFindingError):
        problem.solve(np.array([-1.0]), np.array([[0.0], [2.0]]), 1e-10)

    assert capfd.readouterr().err == ""


def test_root_problem_refuses_equations_that_do_not_determine_the_variables():
    """CasADi refuses a structurally singular system, and that is a `RootFindingError`
    like any other failure to find a root."""
    w, p = cas.SX.sym("w"), cas.SX.sym("p")
    problem = RootProblem(
        cas.Function("residual", [w, p], [p]), np.array([-1.0]), np.array([1.0])
    )

    with pytest.raises(RootFindingError, match="singular"):
        problem.solve(np.array([0.0]), np.array([[0.0]]), 1e-10)
