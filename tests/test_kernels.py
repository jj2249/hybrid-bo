"""Kernels vs closed-form numpy references."""

import casadi as cas
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from hybrid_bo.legacy.kernels import (
    ConstantKernel,
    LinearKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    MaternInfKernel,
    OutputScaleKernel,
)
from hybrid_bo.legacy.parameterization import Parameter

TOL = {"rtol": 1e-8, "atol": 1e-10}

rng = np.random.default_rng(0)
X1 = rng.normal(size=(5, 2))
X2 = rng.normal(size=(4, 2))
LENGTHSCALE = np.array([[0.7], [1.3]])


def scaled_distance(X1, X2):
    return cdist(X1 / LENGTHSCALE.T, X2 / LENGTHSCALE.T)


@pytest.mark.parametrize(
    ("kernel_cls", "reference"),
    [
        (Matern12Kernel, lambda r: np.exp(-r)),
        (Matern32Kernel, lambda r: (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)),
        (
            Matern52Kernel,
            lambda r: (1 + np.sqrt(5) * r + 5 / 3 * r**2) * np.exp(-np.sqrt(5) * r),
        ),
        (MaternInfKernel, lambda r: np.exp(-0.5 * r**2)),
    ],
)
def test_matern(kernel_cls, reference):
    kernel = kernel_cls(2, lengthscale=Parameter(n_rows=2, value=LENGTHSCALE))
    np.testing.assert_allclose(
        kernel.k(X1, X2), reference(scaled_distance(X1, X2)), **TOL
    )


def test_outputscale_multiplies_base_kernel():
    base = Matern52Kernel(2, lengthscale=Parameter(n_rows=2, value=LENGTHSCALE))
    kernel = OutputScaleKernel(base, outputscale=Parameter(value=2.5))
    np.testing.assert_allclose(kernel.k(X1, X2), 2.5 * base.k(X1, X2), **TOL)


def test_constant():
    kernel = ConstantKernel(2, constant=Parameter(value=0.3))
    np.testing.assert_allclose(kernel.k(X1, X2), np.full((5, 4), 0.3), **TOL)


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (np.array([[2.0]]), 2.0 * X1 @ X2.T),
        (np.array([[2.0], [0.5]]), X1 @ np.diag([2.0, 0.5]) @ X2.T),
    ],
    ids=["shared", "ARD"],
)
def test_linear(scale, expected):
    kernel = LinearKernel(2, scale=Parameter(n_rows=scale.shape[0], value=scale))
    np.testing.assert_allclose(kernel.k(X1, X2), expected, **TOL)


@pytest.mark.xfail(
    strict=True, reason="ARD k_variable uses scale.value, not the symbol"
)
def test_linear_ard_scale_is_trainable():
    kernel = LinearKernel(2, ARD=True)
    X = cas.SX.sym("X", 3, 2)
    assert cas.depends_on(kernel.k_variable(X, X), kernel.scale.symbol)
