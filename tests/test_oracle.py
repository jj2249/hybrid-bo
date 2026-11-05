"""The oracle: the oracle optimum, solved through the hybrid model with the oracle as the
unknown equations, against values recorded from the NLP it used to build by hand and
against closed forms, including an oracle that reads the unknown states.

pytest runs every function whose name starts with `test_`.
"""

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest
from test_no_known_equations import composite_problem
from test_two_unknown_states import benchmark_problem

import hybrid_bo as hb


def implicit_system(known_equations: bool) -> hb.System:
    """A test system over u in [-1, 1] with one unknown state y, measured, and with
    `known_equations`, one mechanistic state x = y."""
    u = hb.Var("u", -1.0, 1.0)
    y = hb.Var("y", start=(-3.0, 3.0))
    if not known_equations:
        return hb.System(u=[u], y=[y], h_inputs=[u], measured=[y])
    x = hb.Var("x", start=(-3.0, 3.0))
    return hb.System(
        u=[u], x=[x], y=[y], g=lambda v: [v.x - v.y], h_inputs=[u], measured=[y]
    )


def implicit_oracle(v):
    """An unknown equation that reads y: y = 0.5 y + sin(u), so y = 2 sin(u)."""
    return [0.5 * v.y + cas.sin(v.u)]


def test_the_oracle_optimum_of_the_illustrative_example_keeps_its_value():
    """The illustrative example's oracle optimum is the value the hand-built NLP gave
    from seed 0, which is the optimum the example states."""
    optimum = hb.oracle_optimum(
        ex.problem, ex.oracle, np.random.default_rng(0), n_starts=30
    )

    # `pytest.approx` compares with a tolerance, since IPOPT differs by machine
    assert optimum == pytest.approx(-50.4608044704652, abs=1e-8)
    assert optimum == pytest.approx(ex.optimum, abs=1e-8)


def test_the_oracle_optimum_without_known_equations_keeps_its_value():
    """With no mechanistic states, the oracle optimum is the value the hand-built NLP
    gave from seed 0."""
    problem, _ = composite_problem(1)

    optimum = hb.oracle_optimum(
        problem, lambda v: [cas.sin(v.u1) + v.u1 / 2], np.random.default_rng(0)
    )

    assert optimum == pytest.approx(0.0038600001548442595, abs=1e-8)


def test_the_oracle_optimum_with_two_unknown_states_keeps_its_value():
    """With P = 2, the oracle's rows line up with the unknown states: the oracle optimum
    is the value the hand-built NLP gave from seed 0."""
    problem, _ = benchmark_problem(1)

    optimum = hb.oracle_optimum(
        problem,
        lambda v: [cas.sin(v.u1), cas.cos(v.u1)],
        np.random.default_rng(0),
    )

    assert optimum == pytest.approx(0.03871480730686, abs=1e-8)


# `parametrize` runs the test once per value, here with and without known equations
@pytest.mark.parametrize("known_equations", [True, False])
def test_the_oracle_optimum_of_an_oracle_that_reads_y(known_equations):
    """An oracle may read y: with y = 0.5 y + sin(u), so y = 2 sin(u), the lowest y over
    u in [-1, 1] is -2 sin(1), where the explicit y = sin(u) would give -sin(1)."""
    problem = hb.Problem(implicit_system(known_equations), f=lambda v: v.y)

    optimum = hb.oracle_optimum(problem, implicit_oracle, np.random.default_rng(0))

    assert optimum == pytest.approx(-2 * np.sin(1.0), abs=1e-6)


def test_the_oracle_simulation_of_an_oracle_that_reads_y():
    """The oracle simulation solves an oracle that reads y for y = 2 sin(u) too."""
    measure = hb.oracle_measure(implicit_system(True), implicit_oracle, noise_std=0.0)

    measurement = measure(np.array([0.5]), np.random.default_rng(0))

    np.testing.assert_allclose(measurement.y, [2 * np.sin(0.5)], atol=1e-8)
    np.testing.assert_allclose(measurement.x, [2 * np.sin(0.5)], atol=1e-8)
