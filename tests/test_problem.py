"""Illustrative problem evaluation vs its closed-form solution.

h_unknown: x1 = sin(u)
h_known:   x0 + exp(x0) = x1  =>  x0 = x1 - W(exp(x1))  (Lambert W)
f:         (6 x0 - 2)^2 sin(12 x0 - 4)
"""

import illustrative_example as ex
import numpy as np
from scipy.special import lambertw


def test_evaluate_with_simulation(illustrative_config):
    problem = ex.get_custom_problem(illustrative_config)
    u = np.array([[-2.0], [-0.5], [0.0], [1.0], [2.0]])

    f, x = problem.evaluate_with_simulation(u, rng=np.random.default_rng(0))

    x1 = np.sin(u)
    x0 = x1 - lambertw(np.exp(x1)).real
    np.testing.assert_allclose(x, np.hstack([x0, x1]), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        f, (6 * x0 - 2) ** 2 * np.sin(12 * x0 - 4), rtol=1e-8, atol=1e-10
    )
