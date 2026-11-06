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

# The illustrative example's minimizer, found by a bounded search of the closed form
U_AT_THE_OPTIMUM = np.array([-0.44633082071941])


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
    problem = hb.Problem(implicit_system(True), f=lambda v: v.y)
    measure = hb.measure_oracle(problem, implicit_oracle, noise_std=0.0)

    measured = measure(np.array([0.5]), np.random.default_rng(0))

    np.testing.assert_allclose(measured, [2 * np.sin(0.5)], atol=1e-8)


def test_a_measurement_below_the_stated_optimum_warns():
    """A point whose oracle objective is below `optimum`, beyond the tolerance, can't
    leave it the optimum, so the measurement warns."""
    measure = hb.measure_oracle(
        ex.problem, ex.oracle, noise_std=0.0, optimum=ex.optimum + 1e-3
    )

    # `pytest.warns` fails unless the block warns with the category, matching `match`
    with pytest.warns(hb.OracleOptimumWarning, match="below the stated"):
        measure(U_AT_THE_OPTIMUM, np.random.default_rng(0))


# `filterwarnings("error")` turns every warning into a failure of this test
@pytest.mark.filterwarnings("error")
def test_a_measurement_at_the_stated_optimum_does_not_warn():
    """The point that reaches the stated optimum doesn't warn, nor does one above it."""
    # Setup: check that the point does reach the optimum
    values = hb.oracle_values(
        ex.problem,
        ex.oracle,
        U_AT_THE_OPTIMUM[np.newaxis],
        np.random.default_rng(0),
    )
    assert values.f[0] == pytest.approx(ex.optimum, abs=1e-8)

    # Act and check: warnings are failures here
    measure = hb.measure_oracle(
        ex.problem, ex.oracle, noise_std=0.0, optimum=ex.optimum
    )
    for u in (U_AT_THE_OPTIMUM, np.array([1.5])):
        measure(u, np.random.default_rng(0))


def test_the_optimum_check_draws_nothing():
    """A measurement draws the same with and without `optimum`, since the check only
    reads the states it already solved."""
    plain = hb.measure_oracle(ex.problem, ex.oracle, noise_std=0.1)
    checked = hb.measure_oracle(
        ex.problem, ex.oracle, noise_std=0.1, optimum=ex.optimum
    )

    first = plain(np.array([0.5]), np.random.default_rng(0))
    second = checked(np.array([0.5]), np.random.default_rng(0))

    np.testing.assert_array_equal(first, second)


def test_oracle_values_solve_every_state_at_each_point():
    """The oracle values at any u are the states and objective of the closed form, one
    row per point."""
    u = np.array([[0.5], [-0.25]])

    values = hb.oracle_values(
        hb.Problem(implicit_system(True), f=lambda v: v.y + 1.0),
        implicit_oracle,
        u,
        np.random.default_rng(0),
    )

    np.testing.assert_allclose(values.y, 2 * np.sin(u), atol=1e-8)
    np.testing.assert_allclose(values.x, 2 * np.sin(u), atol=1e-8)
    np.testing.assert_allclose(values.f, 2 * np.sin(u[:, 0]) + 1.0, atol=1e-8)


@pytest.mark.filterwarnings("error")
def test_a_reconstruction_agrees_with_the_values_of_the_oracle():
    """With one root, the states reconstructed from noise-free measurements are the ones
    `oracle_values` gives, so `check_reconstruction` doesn't warn."""
    rng = np.random.default_rng(0)
    u = np.array([[-1.0], [0.5], [1.5]])
    dataset = hb.reconstruct_states(ex.problem, u=u, measured=np.sin(u), rng=rng)

    values = hb.oracle_values(ex.problem, ex.oracle, u, np.random.default_rng(1))

    hb.check_reconstruction(ex.system, dataset, values)
