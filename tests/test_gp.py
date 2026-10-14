"""GP prediction and log marginal likelihood vs numpy/scipy references, and fitting
vs the legacy GP."""

import casadi as cas
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hybrid_bo import (
    GP,
    FittedGP,
    Hyper,
    Ipopt,
    default_gp,
    fit_gp,
    kernels,
    log_likelihood,
    predict,
)
from hybrid_bo.gp import Scaler
from hybrid_bo.legacy import GP as LegacyGP
from hybrid_bo.legacy.affine_transformers import MinMaxTransformer, StandardTransformer
from hybrid_bo.legacy.kernels import Matern52Kernel, OutputScaleKernel
from hybrid_bo.legacy.means import ZeroMean
from hybrid_bo.legacy.optimizers import CasadiOptimizer, MultiStartOptimizer

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


@pytest.fixture(params=["zero_mean", "constant_mean"])
def model(request):
    """A GP with known hyperparameters and identity scalers, so predictions are in
    the units of the reference; and its prior mean as a function of X."""
    hyperparameters = {
        "noise_std": np.array([NOISE_STD]),
        "outputscale": np.array([1.7]),
        "lengthscale": np.array([0.6, 0.9]),
    }
    spec = default_gp(2)
    gp = GP(
        (NOISE, *spec.hyperparameters),
        spec.kernel,
    )
    prior_mean = np.zeros
    if request.param == "constant_mean":
        gp = GP(
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
    gp, fitted, prior_mean = model
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    K_star = matern52(X_train, X_test)
    residual = Y_train - prior_mean(len(X_train))
    mean_ref = prior_mean(len(X_test)) + K_star.T @ np.linalg.solve(K, residual)
    cov_ref = matern52(X_test, X_test) - K_star.T @ np.linalg.solve(K, K_star)

    mean, std = predict(gp, fitted, X_test)
    assert mean.shape == std.shape == (len(X_test),)
    np.testing.assert_allclose(mean, mean_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, np.sqrt(np.diag(cov_ref)), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("symbol_type", [cas.SX, cas.MX])
def test_predict_symbolic_matches_numeric(model, symbol_type):
    gp, fitted, _ = model
    X = symbol_type.sym("X", *X_test.shape)
    function = cas.Function("predict", [X], list(predict(gp, fitted, X)))
    mean, std = function(X_test)
    expected_mean, expected_std = predict(gp, fitted, X_test)
    np.testing.assert_allclose(mean.full().ravel(), expected_mean, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std.full().ravel(), expected_std, rtol=RTOL, atol=ATOL)


def test_predict_undoes_the_scalers(model):
    gp, fitted, _ = model
    x_scaler = Scaler(np.array([2.0, 0.5]), np.array([0.1, -0.3]))
    y_scaler = Scaler(np.array([4.0]), np.array([1.0]))
    scaled = FittedGP(fitted.hyperparameters, fitted.X, fitted.Y, x_scaler, y_scaler)
    X_original = (X_test - x_scaler.intercept) / x_scaler.slope

    mean, std = predict(gp, scaled, X_original)
    expected_mean, expected_std = predict(gp, fitted, X_test)
    np.testing.assert_allclose(mean, (expected_mean - 1.0) / 4.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, expected_std / 4.0, rtol=RTOL, atol=ATOL)


def test_log_likelihood(model):
    gp, fitted, prior_mean = model
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    ref = multivariate_normal(prior_mean(len(X_train)), K).logpdf(Y_train)
    np.testing.assert_allclose(log_likelihood(gp, fitted), ref, rtol=RTOL, atol=ATOL)


def test_hyper_checks():
    with pytest.raises(ValueError, match="positive"):
        Hyper("offset", 0.0, (-1.0, 1.0))
    with pytest.raises(ValueError, match="inside"):
        Hyper("scale", 20.0, (0.1, 10.0))
    with pytest.raises(ValueError, match="identifier"):
        Hyper("noise std", 0.1, (0.01, 1.0))
    with pytest.raises(ValueError, match="repeat"):
        GP((NOISE, NOISE), lambda h, X1, X2: kernels.constant(X1, X2))


# Fitting vs the legacy GP: same data, same generator, so the same starts

FIT_DATA = {
    "1d": (np.linspace(-2.0, 2.0, 4)[:, None], np.sin(np.linspace(-2.0, 2.0, 4))),
    "2d": (X_train, Y_train),
}


def fit_legacy(X, Y, learn_noise, rng):
    """The legacy GP of the illustrative example, fitted as its routines do."""
    d = X.shape[1]
    base_kernel = Matern52Kernel(d)
    kernel = OutputScaleKernel(base_kernel)
    optimizer = MultiStartOptimizer(
        CasadiOptimizer("ipopt", "nonlinear", {"print_time": 0}, {"print_level": 0})
    )
    gp = LegacyGP(d, kernel, ZeroMean(d), optimizer=optimizer)
    if not learn_noise:
        gp.noise_std.fix(0.0)
    x_transformer, y_transformer = MinMaxTransformer(), StandardTransformer()
    gp.set_XY_train(
        x_transformer.fit_transform(X), y_transformer.fit_transform(Y[:, None])
    )
    gp.create_training_problem()
    optimizer.lower_bounds_starting_points = optimizer.lower_bounds
    optimizer.upper_bounds_starting_points = optimizer.upper_bounds
    optimizer.X0 = optimizer.create_lhs_samples(10, rng)
    gp.solve_training_problem()

    hyperparameters = {
        "outputscale": kernel.outputscale.value.ravel(),
        "lengthscale": base_kernel.lengthscale.value.ravel(),
    }
    if learn_noise:
        hyperparameters["noise_std"] = gp.noise_std.value.ravel()
    mean, cov = gp.predict(x_transformer.transform(X_test[:, : X.shape[1]]))
    prediction = (
        y_transformer.inverse_transform(mean).ravel(),
        np.sqrt(np.diag(cov)) / y_transformer.slope.item(),
    )
    return hyperparameters, prediction


@pytest.mark.parametrize("learn_noise", [False, True], ids=["noise_free", "noise"])
@pytest.mark.parametrize("data", FIT_DATA, ids=list(FIT_DATA))
def test_fit_matches_legacy(data, learn_noise):
    X, Y = FIT_DATA[data]
    gp = default_gp(X.shape[1])
    if learn_noise:  # legacy order: noise first, then the kernel's
        gp = GP((NOISE, *gp.hyperparameters), gp.kernel)
    rng, rng_legacy = np.random.default_rng(1), np.random.default_rng(1)

    fitted = fit_gp(gp, X, Y, rng)
    hyperparameters, (mean_legacy, std_legacy) = fit_legacy(
        X, Y, learn_noise, rng_legacy
    )

    assert fitted.hyperparameters.keys() == hyperparameters.keys()
    for name, value in hyperparameters.items():
        np.testing.assert_allclose(
            fitted.hyperparameters[name], value, rtol=1e-6, err_msg=name
        )
    mean, std = predict(gp, fitted, X_test[:, : X.shape[1]])
    np.testing.assert_allclose(mean, mean_legacy, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(std, std_legacy, rtol=1e-6, atol=1e-8)

    # Same draws in the same order: the streams are in the same state, and the next
    # child (what the next lhs call gets) is the same
    assert rng.bit_generator.state == rng_legacy.bit_generator.state
    assert rng.spawn(1)[0].random() == rng_legacy.spawn(1)[0].random()


def test_failed_fit_keeps_initial_values(caplog):
    X, Y = FIT_DATA["1d"]
    no_iterations = Ipopt(solver_options={"print_level": 0, "max_iter": 0})
    fitted = fit_gp(default_gp(1), X, Y, np.random.default_rng(0), solver=no_iterations)
    assert fitted.hyperparameters == pytest.approx(
        {"outputscale": np.array([1.0]), "lengthscale": np.array([1.0])}
    )
    assert "initial values" in caplog.text
