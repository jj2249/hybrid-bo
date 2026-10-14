"""GP posterior and log marginal likelihood vs numpy/scipy references."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hybrid_bo.legacy import GP
from hybrid_bo.legacy.kernels import Matern52Kernel, OutputScaleKernel
from hybrid_bo.legacy.means import ConstantMean, ZeroMean
from hybrid_bo.legacy.parameterization import Parameter

TOL = {"rtol": 1e-8, "atol": 1e-10}
NOISE_STD = 0.1
JITTER = 1e-10  # GP always adds this to the diagonal

rng = np.random.default_rng(0)
X_train = rng.uniform(-1, 1, size=(6, 2))
Y_train = np.sin(3 * X_train[:, [0]]) + X_train[:, [1]] ** 2
X_test = rng.uniform(-1, 1, size=(4, 2))


@pytest.fixture(params=["zero_mean", "constant_mean"])
def gp(request):
    kernel = OutputScaleKernel(
        Matern52Kernel(
            2, lengthscale=Parameter(n_rows=2, value=np.array([[0.6], [0.9]]))
        ),
        outputscale=Parameter(value=1.7),
    )
    mean = (
        ZeroMean(2)
        if request.param == "zero_mean"
        else ConstantMean(2, constant=Parameter(value=0.4))
    )
    gp = GP(2, kernel, mean)
    gp.noise_std.fix(NOISE_STD)
    gp.set_XY_train(X_train, Y_train)
    return gp


def prior(gp):
    K = gp.kernel.k(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    return gp.mean.m(X_train), K


def test_predict(gp):
    m_train, K = prior(gp)
    K_star = gp.kernel.k(X_train, X_test)
    mean_ref = gp.mean.m(X_test) + K_star.T @ np.linalg.solve(K, Y_train - m_train)
    cov_ref = gp.kernel.k(X_test, X_test) - K_star.T @ np.linalg.solve(K, K_star)

    mean, cov = gp.predict(X_test)
    np.testing.assert_allclose(mean, mean_ref, **TOL)
    np.testing.assert_allclose(cov, cov_ref, **TOL)


def test_log_mll(gp):
    m_train, K = prior(gp)
    ref = multivariate_normal(m_train.ravel(), K).logpdf(Y_train.ravel())
    np.testing.assert_allclose(gp.log_mll(), [[ref]], **TOL)
