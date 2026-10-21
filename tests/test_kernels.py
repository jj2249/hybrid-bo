"""Kernel covariances vs closed-form numpy references."""

import casadi as cas
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from hybrid_bo.gp import (
    Matern12,
    Matern32,
    Matern52,
    Param,
    Scale,
    SquaredExponential,
    covariance,
)

RTOL, ATOL = 1e-8, 1e-10

rng = np.random.default_rng(0)
X1 = rng.normal(size=(5, 2))
X2 = rng.normal(size=(4, 2))


def evaluate(kernel):
    """The kernel's covariance between the numeric points X1 and X2, as numpy."""
    return cas.evalf(covariance(kernel, cas.DM(X1), cas.DM(X2))).full()


def scaled_distance(lengthscale):
    """The Euclidean distance between each row of X1 and of X2, each dimension
    divided by its lengthscale, shape (5, 4)."""
    return cdist(X1 / lengthscale, X2 / lengthscale)


# `parametrize` runs the test once per entry of the list, each as its own test case.
# Two of them stacked run every combination: each kernel, with each lengthscale
@pytest.mark.parametrize(
    ("kernel", "reference"),
    [
        (Matern12, lambda r: np.exp(-r)),
        (Matern32, lambda r: (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)),
        (
            Matern52,
            lambda r: (1 + np.sqrt(5) * r + 5 / 3 * r**2) * np.exp(-np.sqrt(5) * r),
        ),
        (SquaredExponential, lambda r: np.exp(-0.5 * r**2)),
    ],
)
@pytest.mark.parametrize(
    "lengthscale", [(0.7, 1.3), 0.8], ids=["one per input", "one number"]
)
def test_stationary(kernel, reference, lengthscale):
    """Each stationary kernel is its closed form of the scaled distance, with one
    lengthscale per input (ARD) or one number for every input."""
    np.testing.assert_allclose(
        evaluate(kernel(lengthscale=lengthscale)),
        reference(scaled_distance(np.array(lengthscale))),
        rtol=RTOL,
        atol=ATOL,
    )


def test_scale_multiplies_unsquared():
    """`Scale` multiplies its kernel by the output scale itself, not its square."""
    kernel = Matern52(lengthscale=0.8)
    np.testing.assert_allclose(
        evaluate(Scale(kernel, outputscale=2.5)),
        2.5 * evaluate(kernel),
        rtol=RTOL,
        atol=ATOL,
    )


def test_param_checks():
    """`Param` and the kernels reject invalid arguments."""
    # `pytest.raises` fails the test unless the block raises that exception, with a
    # message matching `match`
    with pytest.raises(ValueError, match="positive"):
        Param(0.0, (-1.0, 1.0))
    with pytest.raises(ValueError, match="inside"):
        Param(20.0, (0.1, 10.0))
    with pytest.raises(ValueError, match="ard"):
        Matern52(lengthscale=(0.5, 0.5), ard=False)
