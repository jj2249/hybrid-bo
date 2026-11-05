"""Optimizing the hybrid model with the GP's mean, without Bayesian optimization: given
measurements, the surrogate model fitted to them, and the nominal optimum it gives.

pytest runs every function whose name starts with `test_`.
"""

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest

import hybrid_bo as hb
from hybrid_bo.evaluation import evaluate
from hybrid_bo.gp import predict


def data_of(problem: hb.Problem, measure: hb.Measure, n: int) -> hb.History:
    """n evaluated points over the start bounds of u, as training data."""
    rng = np.random.default_rng(0)
    return evaluate(problem, measure, hb.initial_points(problem.system.u, n, rng), rng)


def test_a_gp_of_the_unknown_equation_gives_the_oracle_optimum():
    """With enough data the GP reproduces y = sin(u), so the nominal optimum of the
    illustrative example is `hb.oracle_optimum`, at u = -0.45."""
    # Plain arrays of u and the measured y, as data from outside a run would be
    rng = np.random.default_rng(1)
    u = np.linspace(-2.0, 2.0, 30)[:, np.newaxis]
    history = hb.reconstruct(ex.problem, u=u, measured=np.sin(u), rng=rng)

    # The surrogate model, and the hybrid model's solution with it
    fitted = hb.fit_surrogate(ex.system, history, gp=ex.gp, rng=rng)
    solution = hb.optimize_hybrid_model(ex.problem, fitted, rng, n_starts=30)

    # The same optimum as with the oracle
    optimum = hb.oracle_optimum(ex.problem, ex.oracle, rng, n_starts=30)
    np.testing.assert_allclose(solution.f, optimum, rtol=1e-4)
    np.testing.assert_allclose(solution.u, [-0.45], atol=5e-3)


def test_the_solution_solves_the_hybrid_model():
    """The solution satisfies g = 0 and its y is the GP's mean there, also with few
    data, where the mean is not the oracle."""
    # A GP of six points
    data = data_of(ex.problem, ex.measure, 6)
    rng = np.random.default_rng(1)
    fitted = hb.fit_surrogate(ex.system, data, gp=ex.gp, rng=rng)

    solution = hb.optimize_hybrid_model(ex.problem, fitted, rng)

    # The known equation, the tie to the mean, and f at those states
    mean, _ = predict(fitted[0], solution.u[np.newaxis])
    residual = solution.x + np.exp(solution.x) - solution.y
    objective = (6 * solution.x - 2) ** 2 * np.sin(12 * solution.x - 4)
    np.testing.assert_allclose(residual, 0.0, atol=1e-7)
    np.testing.assert_allclose(solution.y, mean, atol=1e-7)
    np.testing.assert_allclose(solution.f, objective.item(), rtol=1e-9)


def test_the_solution_stays_within_the_bounds_of_the_unknown_states():
    """With y = sin(u) bounded by 0.2, f = (y - 0.5)^2 is lowest at the bound, not at
    y = 0.5."""
    # A system with bounds on y, and data that leave them
    u_var, y_var = hb.Var("u", -2.0, 2.0), hb.Var("y", -0.2, 0.2)
    system = hb.System(u=[u_var], y=[y_var], h_inputs=[u_var], measured=[y_var])
    problem = hb.Problem(system, f=lambda v: (v.y - 0.5) ** 2)
    u = np.linspace(-2.0, 2.0, 25)[:, np.newaxis]

    # `reconstruct` warns of those data and keeps them
    rng = np.random.default_rng(0)
    # `warns` passes only if the block gives that warning
    with pytest.warns(hb.OutOfBoundsWarning):
        history = hb.reconstruct(problem, u=u, measured=np.sin(u), rng=rng)

    # The solution stops at the bound
    fitted = hb.fit_surrogate(system, history, gp=ex.gp, rng=rng)
    solution = hb.optimize_hybrid_model(problem, fitted, rng)

    np.testing.assert_allclose(solution.y, [0.2], atol=1e-6)


def test_a_surrogate_model_can_take_a_mechanistic_state():
    """With x among the arguments of h, the GP is fitted to the reconstructed x: here
    y = x exactly, and x comes only from the root-finding problem."""
    # h takes x, and x is not measured: g says x = y
    u_var, x_var = hb.Var("u", -2.0, 2.0), hb.Var("x", -2.0, 2.0)
    y_var = hb.Var("y", -2.0, 2.0)
    system = hb.System(
        u=[u_var],
        x=[x_var],
        y=[y_var],
        g=lambda v: [v.x - v.y],
        h_inputs=[x_var],
        measured=[y_var],
    )
    problem = hb.Problem(system, f=lambda v: v.x)
    u = np.linspace(-1.0, 1.0, 9)[:, np.newaxis]

    # The fit reads the reconstructed x
    rng = np.random.default_rng(0)
    history = hb.reconstruct(problem, u=u, measured=u, rng=rng)
    fitted = hb.fit_surrogate(system, history, gp=ex.gp, rng=rng)

    # The GP's mean follows y = x
    mean, _ = predict(fitted[0], np.array([[0.3]]))
    np.testing.assert_allclose(mean, [0.3], atol=1e-2)


def test_optimize_hybrid_model_works_with_no_known_equations():
    """With M = 0 the training data have no x, and f(y, u) = (y - 0.5)^2 with
    y = sin(u) is lowest at u = pi / 6."""
    # Plain arrays, as measured outside the library; nothing is left to solve for
    u_var, y_var = hb.Var("u", -2.0, 2.0), hb.Var("y", start=(-1.0, 1.0))
    system = hb.System(u=[u_var], y=[y_var], h_inputs=[u_var], measured=[y_var])
    problem = hb.Problem(system, f=lambda v: (v.y - 0.5) ** 2)
    u = np.linspace(-2.0, 2.0, 25)[:, np.newaxis]

    # The pipeline
    rng = np.random.default_rng(0)
    history = hb.reconstruct(problem, u=u, measured=np.sin(u), rng=rng)
    fitted = hb.fit_surrogate(system, history, gp=ex.gp, rng=rng)
    solution = hb.optimize_hybrid_model(problem, fitted, rng)

    # No mechanistic states, and the optimum
    assert solution.x.shape == (0,)
    np.testing.assert_allclose(solution.u, [np.pi / 6], atol=1e-3)
    np.testing.assert_allclose(solution.y, [0.5], atol=1e-4)


def test_optimize_hybrid_model_works_with_two_unknown_states():
    """With P = 2 there is one GP per unknown state, in the order of y: with
    x + exp(x) = y1 + y2 and f = x^2 + (y1 - y2)^2, f is 0 at x = 0, y1 = y2 = 0.5."""
    # The system, and training data of y1 = sin(u1), y2 = cos(u2)
    u1, u2 = hb.Var("u1", -2.0, 2.0), hb.Var("u2", -2.0, 2.0)
    x = hb.Var("x", start=(-2.5, 1.0))
    y1, y2 = hb.Var("y1", start=(-1.0, 1.0)), hb.Var("y2", start=(-1.0, 1.0))
    system = hb.System(
        u=[u1, u2],
        x=[x],
        y=[y1, y2],
        g=lambda v: [v.x + cas.exp(v.x) - v.y1 - v.y2],
        h_inputs=[u1, u2],
        measured=[y1, y2],
    )
    problem = hb.Problem(system, f=lambda v: v.x**2 + (v.y1 - v.y2) ** 2)
    measure = hb.oracle_measure(
        system, lambda v: [cas.sin(v.u1), cas.cos(v.u2)], noise_std=0.0
    )
    data = data_of(problem, measure, 60)

    rng = np.random.default_rng(1)
    fitted = hb.fit_surrogate(system, data, gp=ex.gp, rng=rng)
    solution = hb.optimize_hybrid_model(problem, fitted, rng, n_starts=20)

    # Each GP predicts its own unknown state
    point = np.array([[0.3, 1.1]])
    means = [predict(gp, point)[0].item() for gp in fitted]
    np.testing.assert_allclose(means, [np.sin(0.3), np.cos(1.1)], atol=1e-2)

    # The optimum
    np.testing.assert_allclose(solution.f, 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.y, [0.5, 0.5], atol=5e-3)
    np.testing.assert_allclose(solution.x, [0.0], atol=5e-3)


def test_optimize_hybrid_model_needs_one_gp_per_unknown_state():
    """A surrogate model of the wrong size raises `ValueError`."""
    # `raises` passes only if the block raises that exception
    with pytest.raises(ValueError, match="one fitted GP per unknown state"):
        hb.optimize_hybrid_model(ex.problem, (), np.random.default_rng(0))
