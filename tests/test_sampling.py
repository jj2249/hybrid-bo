"""lhs and uniform vs the scipy/numpy calls they wrap.

The regression baselines depend on these drawing exactly as the code they replaced.
"""

import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import qmc

from hybrid_bo.sampling import lhs, uniform

LOWER = np.array([-2.0, 0.0, 10.0])
UPPER = np.array([2.0, 1.0, 10.5])


def test_lhs_matches_scipy():
    """lhs is scipy's Latin hypercube from the same seed, stretched to the bounds."""
    points = lhs(LOWER, UPPER, 7, np.random.default_rng(0))

    samples = qmc.LatinHypercube(3, rng=np.random.default_rng(0)).random(7)
    np.testing.assert_array_equal(points, LOWER + (UPPER - LOWER) * samples)


def test_uniform_matches_numpy():
    """uniform is numpy's uniform draw from the same seed, stretched to the bounds."""
    points = uniform(LOWER, UPPER, 7, np.random.default_rng(0))

    samples = np.random.default_rng(0).uniform(size=(7, 3))
    np.testing.assert_array_equal(points, LOWER + (UPPER - LOWER) * samples)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize("sample", [lhs, uniform])
def test_shape_and_bounds(sample):
    """Both samplers return one row per point, inside the bounds."""
    points = sample(LOWER, UPPER, 50, np.random.default_rng(1))

    assert points.shape == (50, 3)
    assert (points >= LOWER).all()
    assert (points <= UPPER).all()


def test_lhs_spawns_instead_of_consuming_the_stream():
    """lhs takes the generator's next spawned child and leaves its stream untouched,
    the scipy behavior the draw order relies on (DESIGN.md)."""
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


def test_lhs_does_not_import_scipy_stats():
    """lhs draws without importing `scipy.stats`, which takes about a second, so a
    study task that only draws its initial points starts quickly. Checked in a new
    process, as this one has imported it."""
    code = (
        "import sys; import numpy as np; from hybrid_bo.sampling import lhs; "
        "lhs(np.zeros(2), np.ones(2), 3, np.random.default_rng(0)); "
        "print('scipy.stats' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
