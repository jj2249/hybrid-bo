"""Illustrative problem evaluation vs its closed-form solution.

h_unknown: x1 = sin(u)
h_known:   x0 + exp(x0) = x1  =>  x0 = x1 - W(exp(x1))  (Lambert W)
f:         (6 x0 - 2)^2 sin(12 x0 - 4)
"""

import illustrative_example as ex
import numpy as np
from scipy.special import lambertw

from hybrid_bo import lhs

U = np.array([[-2.0], [-0.5], [0.0], [1.0], [2.0]])


def closed_form(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.sin(u)
    x0 = x1 - lambertw(np.exp(x1)).real
    return np.hstack([x0, x1]), (6 * x0 - 2) ** 2 * np.sin(12 * x0 - 4)


def test_evaluate_with_simulation(illustrative_config):
    problem = ex.get_custom_problem(illustrative_config)

    f, x = problem.evaluate_with_simulation(U, rng=np.random.default_rng(0))

    x_ref, f_ref = closed_form(U)
    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(f, f_ref, rtol=1e-8, atol=1e-10)


def test_noise_free_reconstruction_matches_oracle(illustrative_config):
    # The root-finding problem always runs; without noise it recovers the oracle states
    problem = ex.get_custom_problem(illustrative_config)

    f, x, f_oracle, x_oracle = problem.evaluate_with_noisy_simulation(
        U, 0.0, illustrative_config.indices_x_measured, rng=np.random.default_rng(0)
    )

    x_ref, f_ref = closed_form(U)
    np.testing.assert_allclose(x_oracle, x_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(f_oracle, f_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(f, f_ref, rtol=1e-8, atol=1e-10)


def test_noise_is_the_first_stream_draw(illustrative_config):
    # The fsolve guesses come from LatinHypercube, which spawns a child generator
    # instead of consuming the stream, so the noise is the evaluation's first stream
    # draw. The measured state is the oracle state plus exactly that noise.
    problem = ex.get_custom_problem(illustrative_config)
    measured = illustrative_config.indices_x_measured
    noise_std = 0.1

    _, x, _, x_oracle = problem.single_evaluate_with_noisy_simulation(
        U[[1]].T, noise_std, measured, 100, np.random.default_rng(0)
    )

    noise = np.random.default_rng(0).normal(0.0, noise_std, size=(len(measured), 1))
    np.testing.assert_allclose(x[measured] - x_oracle[measured], noise, rtol=1e-12)


def test_lhs_spawns_instead_of_consuming_the_stream():
    # Pins the scipy behaviour the draw order relies on (DESIGN.md)
    rng = np.random.default_rng(0)
    lhs(np.zeros(1), np.ones(1), 3, rng)
    second = lhs(np.zeros(1), np.ones(1), 3, rng)

    # Each call takes the next SeedSequence child, sharing the counter with spawn
    after_one_spawn = np.random.default_rng(0)
    after_one_spawn.spawn(1)
    np.testing.assert_array_equal(
        second, lhs(np.zeros(1), np.ones(1), 3, after_one_spawn)
    )
    # ... and leaves the stream untouched
    assert rng.random() == np.random.default_rng(0).random()
