"""Kernels vs closed-form numpy references."""

import casadi as cas
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from hybrid_bo import kernels

RTOL, ATOL = 1e-8, 1e-10

rng = np.random.default_rng(0)
X1 = rng.normal(size=(5, 2))
X2 = rng.normal(size=(4, 2))
LENGTHSCALE = np.array([[0.7], [1.3]])


def evaluate(kernel, *args):
    """The kernel at the numeric points X1, X2 with numeric arguments, as numpy."""
    return cas.evalf(kernel(cas.DM(X1), cas.DM(X2), *args)).full()


def scaled_distance(lengthscale):
    """The Euclidean distance between each row of X1 and of X2, each dimension
    divided by its lengthscale, shape (5, 4)."""
    return cdist(X1 / lengthscale.T, X2 / lengthscale.T)


# `parametrize` runs the test once per entry of the list, each as its own test case.
# Two of them stacked run every combination: each kernel, with each lengthscale
@pytest.mark.parametrize(
    ("kernel", "reference"),
    [
        (kernels.matern12, lambda r: np.exp(-r)),
        (kernels.matern32, lambda r: (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)),
        (
            kernels.matern52,
            lambda r: (1 + np.sqrt(5) * r + 5 / 3 * r**2) * np.exp(-np.sqrt(5) * r),
        ),
        (kernels.squared_exponential, lambda r: np.exp(-0.5 * r**2)),
    ],
)
@pytest.mark.parametrize(
    "lengthscale", [LENGTHSCALE, np.array([[0.8]])], ids=["ARD", "shared"]
)
def test_stationary(kernel, reference, lengthscale):
    """Each stationary kernel is its closed form of the scaled distance, with one
    lengthscale per dimension (ARD) or one shared by all."""
    np.testing.assert_allclose(
        evaluate(kernel, cas.DM(lengthscale)),
        reference(scaled_distance(lengthscale)),
        rtol=RTOL,
        atol=ATOL,
    )


def test_constant():
    """The constant kernel is 1 between every pair of points."""
    np.testing.assert_allclose(
        evaluate(kernels.constant), np.ones((5, 4)), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (np.array([[2.0]]), 2.0 * X1 @ X2.T),
        (np.array([[2.0], [0.5]]), X1 @ np.diag([2.0, 0.5]) @ X2.T),
    ],
    ids=["shared", "ARD"],
)
def test_linear(scale, expected):
    """The linear kernel is the dot product of the points, weighted by one scale per
    dimension (ARD) or one shared by all."""
    np.testing.assert_allclose(
        evaluate(kernels.linear, cas.DM(scale)), expected, rtol=RTOL, atol=ATOL
    )


def test_linear_ard_scale_is_trainable():
    """The linear kernel depends on its symbolic ARD scale, so a fit can tune it."""
    scale = cas.SX.sym("scale", 2)
    X = cas.SX.sym("X", 3, 2)
    assert cas.depends_on(kernels.linear(X, X, scale), scale)
