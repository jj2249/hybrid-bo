"""lhs and uniform vs the scipy/numpy calls they wrap.

The regression baselines depend on these drawing exactly as the code they replaced.
"""

import numpy as np
import pytest
from scipy.stats import qmc

from hybrid_bo import lhs, uniform

LOWER = np.array([-2.0, 0.0, 10.0])
UPPER = np.array([2.0, 1.0, 10.5])


def test_lhs_matches_scipy():
    points = lhs(LOWER, UPPER, 7, np.random.default_rng(0))

    samples = qmc.LatinHypercube(3, rng=np.random.default_rng(0)).random(7)
    np.testing.assert_array_equal(points, LOWER + (UPPER - LOWER) * samples)


def test_uniform_matches_numpy():
    points = uniform(LOWER, UPPER, 7, np.random.default_rng(0))

    samples = np.random.default_rng(0).uniform(size=(7, 3))
    np.testing.assert_array_equal(points, LOWER + (UPPER - LOWER) * samples)


@pytest.mark.parametrize("sample", [lhs, uniform])
def test_shape_and_bounds(sample):
    points = sample(LOWER, UPPER, 50, np.random.default_rng(1))

    assert points.shape == (50, 3)
    assert (points >= LOWER).all()
    assert (points <= UPPER).all()
