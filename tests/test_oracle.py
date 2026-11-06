"""The oracle: the oracle optimum, a deterministic equivalent with the unknown equations
tied to the oracle, against a value recorded from the NLP it used to build by hand and
against closed forms, including oracles that read the unknown states.

pytest runs every function whose name starts with `test_`.
"""

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest

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


# `parametrize` runs the test once per value, here with and without known equations
@pytest.mark.parametrize("known_equations", [True, False])
def test_the_oracle_optimum_of_an_oracle_that_reads_y(known_equations):
    """An oracle may read y: with y = 0.5 y + sin(u), so y = 2 sin(u), the lowest y over
    u in [-1, 1] is -2 sin(1), where the explicit y = sin(u) would give -sin(1)."""
    problem = hb.Problem(implicit_system(known_equations), f=lambda v: v.y)

    optimum = hb.oracle_optimum(problem, implicit_oracle, np.random.default_rng(0))

    assert optimum == pytest.approx(-2 * np.sin(1.0), abs=1e-6)


def test_the_oracle_optimum_of_two_unknown_states_that_read_each_other():
    """With y1 = 0.5 y2 + sin(u) and y2 = 0.25 y1 + cos(u) + 1, solving the pair gives
    y1 = (sin(u) + 0.5 cos(u) + 0.5) / 0.875, which rises over u in [-1, 1], so the
    lowest y1 is at u = -1."""
    u = hb.Var("u", -1.0, 1.0)
    y1, y2 = hb.Var("y1", start=(-4.0, 4.0)), hb.Var("y2", start=(-4.0, 4.0))
    system = hb.System(u=[u], y=[y1, y2], h_inputs=[u], measured=[y1, y2])
    problem = hb.Problem(system, f=lambda v: v.y1)

    optimum = hb.oracle_optimum(
        problem,
        lambda v: [0.5 * v.y2 + cas.sin(v.u), 0.25 * v.y1 + cas.cos(v.u) + 1.0],
        np.random.default_rng(0),
    )

    expected = (np.sin(-1.0) + 0.5 * np.cos(-1.0) + 0.5) / 0.875
    assert optimum == pytest.approx(expected, abs=1e-6)


def test_the_oracle_optimum_raises_when_every_start_fails():
    """Known equations with no real root, x^2 + 1 = 0, leave every start infeasible."""
    u = hb.Var("u", -1.0, 1.0)
    x = hb.Var("x", start=(-1.0, 1.0))
    y = hb.Var("y", start=(-1.0, 1.0))
    system = hb.System(
        u=[u], x=[x], y=[y], g=lambda v: [v.x**2 + 1.0], h_inputs=[u], measured=[y]
    )
    problem = hb.Problem(system, f=lambda v: v.y)

    # `pytest.raises` passes only if the block raises this exception
    with pytest.raises(hb.NoSolutionError):
        hb.oracle_optimum(
            problem, lambda v: [cas.sin(v.u)], np.random.default_rng(0), n_starts=2
        )


def test_the_oracle_simulation_of_an_oracle_that_reads_y():
    """The oracle simulation solves an oracle that reads y for y = 2 sin(u) too."""
    measure = hb.oracle_measure(implicit_system(True), implicit_oracle, noise_std=0.0)

    measurement = measure(np.array([0.5]), np.random.default_rng(0))

    np.testing.assert_allclose(measurement.y, [2 * np.sin(0.5)], atol=1e-8)
    np.testing.assert_allclose(measurement.x, [2 * np.sin(0.5)], atol=1e-8)
