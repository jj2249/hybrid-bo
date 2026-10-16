"""Fitted GP prediction and log marginal likelihood vs numpy/scipy references, and
what a fit finds and draws."""

from dataclasses import replace

import casadi as cas
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hybrid_bo import kernels
from hybrid_bo.gp import (
    FittedGP,
    Hyper,
    PriorGP,
    Scaler,
    default_gp,
    fit_gp,
    log_marginal_likelihood,
    predict,
)
from hybrid_bo.solvers import Ipopt

RTOL, ATOL = 1e-8, 1e-10
NOISE_STD = 0.1
JITTER = 1e-10  # always added to the diagonal

rng = np.random.default_rng(0)
X_train = rng.uniform(-1, 1, size=(6, 2))
Y_train = np.sin(3 * X_train[:, 0]) + X_train[:, 1] ** 2
X_test = rng.uniform(-1, 1, size=(4, 2))

NOISE = Hyper("noise_std", 1e-2, bounds=(1e-6, 1.0))


def matern52(X1, X2):
    """The reference kernel: 1.7 * Matern-5/2 with lengthscales (0.6, 0.9)."""
    r = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / [0.6, 0.9], axis=-1)
    return 1.7 * (1 + np.sqrt(5) * r + 5 / 3 * r**2) * np.exp(-np.sqrt(5) * r)


# A fixture is a function marked `@pytest.fixture`: a test with an argument of the
# same name gets its return value. With `params`, each test that uses it runs once per
# entry, which the fixture reads from `request.param`
@pytest.fixture(params=["zero_mean", "constant_mean"])
def model(request):
    """A prior GP and a fitted GP with known hyperparameters and identity scalers, so
    predictions are in the units of the reference; and the prior mean as a function
    of X."""
    hyperparameters = {
        "noise_std": np.array([NOISE_STD]),
        "outputscale": np.array([1.7]),
        "lengthscale": np.array([0.6, 0.9]),
    }
    # The default GP (noise-free) plus a noise hyperparameter
    noise_free = default_gp(2)
    gp = PriorGP(
        (NOISE, *noise_free.hyperparameters),
        noise_free.kernel,
    )
    # The constant-mean variant adds an offset hyperparameter
    prior_mean = np.zeros
    if request.param == "constant_mean":
        gp = PriorGP(
            (*gp.hyperparameters, Hyper("offset", 0.0, (-1.0, 1.0), log=False)),
            gp.kernel,
            mean=lambda h, X: h.offset * cas.SX.ones(X.shape[0], 1),
        )
        hyperparameters["offset"] = np.array([0.4])

        def prior_mean(n):
            return np.full(n, 0.4)

    identity = Scaler(np.ones(2), np.zeros(2))
    fitted = FittedGP(
        hyperparameters,
        X_train,
        Y_train,
        identity,
        Scaler(np.ones(1), np.zeros(1)),
    )
    return gp, fitted, prior_mean


def test_predict(model):
    """The fitted GP's mean and std match the closed-form conditional mean and std of
    the noise-free output in numpy."""
    # Reference: the conditional mean and covariance from the kernel matrices
    gp, fitted, prior_mean = model
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    K_star = matern52(X_train, X_test)
    residual = Y_train - prior_mean(len(X_train))
    mean_ref = prior_mean(len(X_test)) + K_star.T @ np.linalg.solve(K, residual)
    cov_ref = matern52(X_test, X_test) - K_star.T @ np.linalg.solve(K, K_star)

    # Prediction: one mean and std per row of X_test
    mean, std = predict(gp, fitted, X_test)
    assert mean.shape == std.shape == (len(X_test),)
    np.testing.assert_allclose(mean, mean_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, np.sqrt(np.diag(cov_ref)), rtol=RTOL, atol=ATOL)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize("symbol_type", [cas.SX, cas.MX])
def test_predict_symbolic_matches_numeric(model, symbol_type):
    """Predicting at CasADi symbols gives the same values as predicting at the
    numbers. The MX case checks only that a fitted GP can be evaluated at MX inputs:
    the library math itself is SX only."""
    gp, fitted, _ = model
    X = symbol_type.sym("X", *X_test.shape)
    function = cas.Function("predict", [X], list(predict(gp, fitted, X)))
    mean, std = function(X_test)
    expected_mean, expected_std = predict(gp, fitted, X_test)
    np.testing.assert_allclose(mean.full().ravel(), expected_mean, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std.full().ravel(), expected_std, rtol=RTOL, atol=ATOL)


def test_predict_undoes_the_scalers(model):
    """With non-identity scalers, predict scales its inputs and unscales its outputs:
    the mean and std are in the original units."""
    # The same GP with scalers, and the test points in original units
    gp, fitted, _ = model
    x_scaler = Scaler(np.array([2.0, 0.5]), np.array([0.1, -0.3]))
    y_scaler = Scaler(np.array([4.0]), np.array([1.0]))
    scaled = FittedGP(fitted.hyperparameters, fitted.X, fitted.Y, x_scaler, y_scaler)
    X_original = (X_test - x_scaler.intercept) / x_scaler.slope

    # The same prediction as the unscaled GP, unscaled
    mean, std = predict(gp, scaled, X_original)
    expected_mean, expected_std = predict(gp, fitted, X_test)
    np.testing.assert_allclose(mean, (expected_mean - 1.0) / 4.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, expected_std / 4.0, rtol=RTOL, atol=ATOL)


def test_log_marginal_likelihood(model):
    """The log marginal likelihood is scipy's multivariate normal log density of the
    training data."""
    gp, fitted, prior_mean = model
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    ref = multivariate_normal(prior_mean(len(X_train)), K).logpdf(Y_train)
    np.testing.assert_allclose(
        log_marginal_likelihood(gp, fitted), ref, rtol=RTOL, atol=ATOL
    )


def test_hyper_checks():
    """Hyper and PriorGP reject invalid arguments."""
    # `pytest.raises` fails the test unless the block raises that exception, with a
    # message matching `match`
    with pytest.raises(ValueError, match="positive"):
        Hyper("offset", 0.0, (-1.0, 1.0))
    with pytest.raises(ValueError, match="inside"):
        Hyper("scale", 20.0, (0.1, 10.0))
    with pytest.raises(ValueError, match="identifier"):
        Hyper("noise std", 0.1, (0.01, 1.0))
    with pytest.raises(ValueError, match="repeat"):
        PriorGP((NOISE, NOISE), lambda h, X1, X2: kernels.constant(X1, X2))


# Fitting: the regression baselines pin the fitted hyperparameters of the
# illustrative example; these check what a fit is and what it draws

FIT_DATA = {
    "1d": (np.linspace(-2.0, 2.0, 4)[:, None], np.sin(np.linspace(-2.0, 2.0, 4))),
    "2d": (X_train, Y_train),
}


def test_fit_maximizes_the_log_marginal_likelihood():
    """The fitted hyperparameters are at least as likely as any point of a log-spaced
    grid over the bounds of the default GP."""
    X, Y = FIT_DATA["1d"]
    gp = default_gp(1)

    fitted = fit_gp(gp, X, Y, np.random.default_rng(1))

    # The likelihood at every grid point, on the same scaled training data
    grid = [
        log_marginal_likelihood(
            gp,
            replace(
                fitted,
                hyperparameters={
                    "outputscale": np.array([outputscale]),
                    "lengthscale": np.array([lengthscale]),
                },
            ),
        )
        for outputscale in np.geomspace(0.1, 10.0, 15)
        for lengthscale in np.geomspace(1e-3, 1e2, 15)
    ]
    assert log_marginal_likelihood(gp, fitted) >= max(grid) - 1e-8


@pytest.mark.parametrize("data", FIT_DATA, ids=list(FIT_DATA))
def test_fit_draws_one_lhs_and_nothing_from_the_stream(data):
    """A fit's only draw is its Latin hypercube of starts, which takes the next
    spawned child and leaves the stream untouched (DESIGN.md, order of draws)."""
    X, Y = FIT_DATA[data]
    rng = np.random.default_rng(1)

    fit_gp(default_gp(X.shape[1]), X, Y, rng)

    # The stream is where a fresh generator's is, and the next child is the second
    reference = np.random.default_rng(1)
    assert rng.bit_generator.state == reference.bit_generator.state
    reference.spawn(1)
    assert rng.spawn(1)[0].random() == reference.spawn(1)[0].random()


def test_failed_fit_keeps_initial_values(caplog):
    """A fit whose solver fails keeps the initial hyperparameters and logs that it
    did.

    `caplog` is a pytest fixture that captures log records during the test.
    """
    X, Y = FIT_DATA["1d"]
    no_iterations = Ipopt(solver_options={"print_level": 0, "max_iter": 0})
    fitted = fit_gp(default_gp(1), X, Y, np.random.default_rng(0), solver=no_iterations)

    # `pytest.approx` compares numbers with a tolerance
    assert fitted.hyperparameters == pytest.approx(
        {"outputscale": np.array([1.0]), "lengthscale": np.array([1.0])}
    )
    assert "initial values" in caplog.text
