from dataclasses import replace

import illustrative_example as ex
import numpy as np
import pytest
from scipy.stats import qmc

from hybrid_bo import Bounds, lhs, uniform

BOUNDS = Bounds([-2.0, 0.0], [2.0, 10.0])


def test_bounds_converts_to_read_only_1d_float():
    bounds = Bounds([0, 1], [2, 3])
    assert bounds.lower.dtype == float
    assert bounds.lower.shape == (2,)
    with pytest.raises(ValueError, match="read-only"):
        bounds.lower[0] = 5.0


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([0.0, 0.0], [1.0]),  # shapes differ
        ([1.0], [0.0]),  # lower > upper
        ([[0.0], [1.0]], [[1.0], [2.0]]),  # column vectors, not 1-D
    ],
)
def test_bounds_rejects_invalid(lower, upper):
    with pytest.raises(ValueError):
        Bounds(lower, upper)


def test_unbounded():
    bounds = Bounds.unbounded(3)
    assert (bounds.lower == -np.inf).all()
    assert (bounds.upper == np.inf).all()


def test_lhs_matches_scaled_latin_hypercube():
    # Same draws and arithmetic as the formula it replaced, so results are bit-identical
    samples = qmc.LatinHypercube(2, rng=np.random.default_rng(0)).random(5)
    expected = BOUNDS.lower + (BOUNDS.upper - BOUNDS.lower) * samples
    np.testing.assert_array_equal(lhs(BOUNDS, 5, np.random.default_rng(0)), expected)


def test_lhs_has_one_point_per_stratum():
    n = 7
    points = lhs(BOUNDS, n, np.random.default_rng(1))
    strata = np.floor((points - BOUNDS.lower) / (BOUNDS.upper - BOUNDS.lower) * n)
    for column in strata.T:
        assert sorted(column) == list(range(n))


def test_uniform_within_bounds():
    points = uniform(BOUNDS, 100, np.random.default_rng(2))
    assert points.shape == (100, 2)
    assert ((points >= BOUNDS.lower) & (points <= BOUNDS.upper)).all()


def test_problem_rejects_infinite_start_bounds(illustrative_config):
    problem = ex.get_custom_problem(illustrative_config)
    with pytest.raises(ValueError, match="x_start_bounds must be finite"):
        replace(problem, x_start_bounds=Bounds.unbounded(problem.n_x))


def test_problem_rejects_wrong_bound_size(illustrative_config):
    problem = ex.get_custom_problem(illustrative_config)
    with pytest.raises(ValueError, match="u_bounds must have size 1"):
        replace(problem, u_bounds=Bounds([0.0, 0.0], [1.0, 1.0]))
