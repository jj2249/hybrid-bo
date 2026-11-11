"""The GP vs numpy/scipy references, and what a fit finds."""

import casadi as cas
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

import hybrid_bo as hb
from hybrid_bo.gp import GP, FittedGP, Matern52, Zero
from hybrid_bo.utils import Scaler

RTOL, ATOL = 1e-8, 1e-10
NOISE_STD = 0.1
JITTER = 1e-6

rng = np.random.default_rng(0)
X_train = rng.uniform(-1, 1, size=(6, 2))
Y_train = np.sin(3 * X_train[:, 0]) + X_train[:, 1] ** 2
X_test = rng.uniform(-1, 1, size=(4, 2))

A, B, OUT = hb.Var("a", -1, 1), hb.Var("b", -1, 1), hb.Var("out", -5, 5)
DATA = hb.Dataset(
    X_train, np.zeros((6, 0)), Y_train[:, None], Y_train, ("a", "b", "out")
)


def matern52(X1, X2):
    """The reference kernel: 1.7 * Matern-5/2 with lengthscales (0.6, 0.9)."""
    r = cdist(X1 / [0.6, 0.9], X2 / [0.6, 0.9])
    return 1.7 * (1 + np.sqrt(5) * r + 5 / 3 * r**2) * np.exp(-np.sqrt(5) * r)


@pytest.fixture
def fitted():
    """A fitted GP with known hyperparameters and identity scalers."""
    gp = GP(
        kernel=Matern52(lengthscale=(0.6, 0.9), outputscale=1.7),
        mean=Zero(),
        noise_std=NOISE_STD,
    )
    identity = Scaler(np.ones(2), np.zeros(2))
    return FittedGP(
        gp, (A, B), OUT, X_train, Y_train, identity, Scaler(np.ones(1), np.zeros(1))
    )


def test_kernel_is_its_closed_form():
    """Matern52's covariance is the closed form of the scaled distance."""
    kernel = Matern52(lengthscale=(0.6, 0.9), outputscale=1.7)
    covariance = cas.evalf(kernel.covariance(cas.DM(X_train), cas.DM(X_test))).full()
    np.testing.assert_allclose(
        covariance, matern52(X_train, X_test), rtol=RTOL, atol=ATOL
    )


def test_predict(fitted):
    """The mean and std are the closed-form conditional mean and std, numeric or
    symbolic."""
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    K_star = matern52(X_train, X_test)
    mean_ref = K_star.T @ np.linalg.solve(K, Y_train)
    cov_ref = matern52(X_test, X_test) - K_star.T @ np.linalg.solve(K, K_star)

    mean, std = fitted.predict(X_test)
    np.testing.assert_allclose(mean, mean_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, np.sqrt(np.diag(cov_ref)), rtol=RTOL, atol=ATOL)

    X = cas.SX.sym("X", *X_test.shape)
    symbolic_mean, _ = cas.Function("predict", [X], list(fitted.predict(X)))(X_test)
    np.testing.assert_allclose(symbolic_mean.full().ravel(), mean, rtol=RTOL)


def test_log_marginal_likelihood(fitted):
    """The log marginal likelihood is scipy's normal log density of the data."""
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    ref = multivariate_normal(np.zeros(len(X_train)), K).logpdf(Y_train)
    np.testing.assert_allclose(fitted.log_marginal_likelihood(), ref, rtol=RTOL)


def test_fit_learns_within_bounds_and_keeps_fixed_values():
    """A fit learns each None within its bounds, one lengthscale per input with ARD,
    keeps fixed values, and records what it models."""
    gp = GP(kernel=Matern52(outputscale=2.0), mean=Zero(), noise_std=0.05)

    fitted = gp.fit(DATA, inputs=[A, B], output=OUT, rng=np.random.default_rng(0))

    kernel = fitted.gp.kernel
    assert kernel.outputscale == 2.0
    assert isinstance(kernel.lengthscale, tuple) and len(kernel.lengthscale) == 2
    assert all(1e-3 <= value <= 1e2 for value in kernel.lengthscale)
    assert (fitted.inputs, fitted.output) == ((A, B), OUT)
    assert gp.kernel == Matern52(outputscale=2.0)


def test_fit_maximizes_the_log_marginal_likelihood():
    """The fit is at least as likely as any point of a grid over the bounds."""
    data = hb.Dataset(
        X_train[:, :1], np.zeros((6, 0)), Y_train[:, None], Y_train, ("a", "out")
    )
    gp = GP(kernel=Matern52(), mean=Zero(), noise_std=0.0)

    fitted = gp.fit(data, inputs=[A], output=OUT, rng=np.random.default_rng(1))

    grid = [
        FittedGP(
            GP(kernel=Matern52((ls,), outputscale=os), mean=Zero(), noise_std=0.0),
            *(fitted.inputs, fitted.output, fitted.X, fitted.Y),
            *(fitted.x_scaler, fitted.y_scaler),
        ).log_marginal_likelihood()
        for os in np.geomspace(0.1, 10.0, 15)
        for ls in np.geomspace(1e-3, 1e2, 15)
    ]
    assert fitted.log_marginal_likelihood() >= max(grid) - 1e-8


def test_fit_with_lengthscale_prior_maximizes_the_log_posterior():
    """With the prior, the fit is at least as probable as any lengthscale of a grid
    over the bounds."""
    data = hb.Dataset(
        X_train[:, :1], np.zeros((6, 0)), Y_train[:, None], Y_train, ("a", "out")
    )
    kernel = Matern52(outputscale=1.0, lengthscale_prior=True)
    gp = GP(kernel=kernel, mean=Zero(), noise_std=0.0)

    fitted = gp.fit(data, inputs=[A], output=OUT, rng=np.random.default_rng(1))

    def log_posterior(ls):
        """The log marginal likelihood plus the log prior at the lengthscale ls."""
        fit = FittedGP(
            GP(kernel=Matern52((ls,), outputscale=1.0), mean=Zero(), noise_std=0.0),
            *(fitted.inputs, fitted.output, fitted.X, fitted.Y),
            *(fitted.x_scaler, fitted.y_scaler),
        )
        return fit.log_marginal_likelihood() + float(
            kernel.log_prior(cas.DM([np.log(ls)]), 1)
        )

    grid = [log_posterior(ls) for ls in np.geomspace(1e-3, 1e2, 200)]
    lengthscale = fitted.gp.kernel.lengthscale
    assert isinstance(lengthscale, tuple)
    assert log_posterior(lengthscale[0]) >= max(grid) - 1e-8


def test_gp_checks():
    """A negative noise, data that aren't finite and inputs that don't match raise."""
    with pytest.raises(ValueError, match="noise_std"):
        GP(kernel=Matern52(), mean=Zero(), noise_std=-0.1)
    bad = hb.Dataset(X_train, DATA.x, DATA.y * np.nan, DATA.f, DATA.names)
    gp = GP(kernel=Matern52(), mean=Zero(), noise_std=0.0)
    with pytest.raises(ValueError, match="finite"):
        gp.fit(bad, inputs=[A, B], output=OUT, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="X has 1 inputs and the GP has 2"):
        FittedGP(
            GP(kernel=Matern52(0.5, outputscale=1.0), mean=Zero(), noise_std=0.0),
            *((A, B), OUT, X_train, Y_train),
            *(Scaler(np.ones(2), np.zeros(2)), Scaler(np.ones(1), np.zeros(1))),
        ).predict(np.zeros((3, 1)))
