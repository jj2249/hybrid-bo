"""Fitted GP prediction and log marginal likelihood vs numpy/scipy references, and
what a fit finds and draws."""

import dataclasses
import logging

import casadi as cas
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hybrid_bo.errors import NoSolutionError
from hybrid_bo.gp import (
    GP,
    FittedGP,
    Matern52,
    Param,
    Scale,
    Zero,
)
from hybrid_bo.solvers import Ipopt
from hybrid_bo.utils import Scaler

RTOL, ATOL = 1e-8, 1e-10
NOISE_STD = 0.1  # in measured units
JITTER = 1e-6  # always added to the diagonal

rng = np.random.default_rng(0)
X_train = rng.uniform(-1, 1, size=(6, 2))
Y_train = np.sin(3 * X_train[:, 0]) + X_train[:, 1] ** 2
X_test = rng.uniform(-1, 1, size=(4, 2))

# The default kernel: an output scale times Matern-5/2, both fitted
DEFAULT = GP(mean=Zero(), kernel=Scale(Matern52()), noise_std=0.0)


def matern52(X1, X2):
    """The reference kernel: 1.7 * Matern-5/2 with lengthscales (0.6, 0.9)."""
    r = np.linalg.norm((X1[:, None, :] - X2[None, :, :]) / [0.6, 0.9], axis=-1)
    return 1.7 * (1 + np.sqrt(5) * r + 5 / 3 * r**2) * np.exp(-np.sqrt(5) * r)


# A fixture is a function marked `@pytest.fixture`: a test with an argument of the
# same name gets its return value
@pytest.fixture
def fitted():
    """A fitted GP with known parameters, noise and identity scalers, so predictions
    are in the units of the reference."""
    gp = GP(
        mean=Zero(),
        kernel=Scale(Matern52(lengthscale=(0.6, 0.9)), 1.7),
        noise_std=NOISE_STD,
    )
    identity = Scaler(np.ones(2), np.zeros(2))
    return FittedGP(gp, X_train, Y_train, identity, Scaler(np.ones(1), np.zeros(1)))


def test_predict(fitted):
    """The fitted GP's mean and std match the closed-form conditional mean and std of
    the noise-free output in numpy."""
    # Reference: the conditional mean and covariance from the kernel matrices
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    K_star = matern52(X_train, X_test)
    mean_ref = K_star.T @ np.linalg.solve(K, Y_train)
    cov_ref = matern52(X_test, X_test) - K_star.T @ np.linalg.solve(K, K_star)

    # Prediction: one mean and std per row of X_test
    mean, std = fitted.predict(X_test)
    assert mean.shape == std.shape == (len(X_test),)
    np.testing.assert_allclose(mean, mean_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, np.sqrt(np.diag(cov_ref)), rtol=RTOL, atol=ATOL)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize("symbol_type", [cas.SX, cas.MX])
def test_predict_symbolic_matches_numeric(fitted, symbol_type):
    """Predicting at CasADi symbols gives the same values as predicting at the
    numbers. The MX case checks only that a fitted GP can be evaluated at MX inputs:
    the library math itself is SX only."""
    X = symbol_type.sym("X", *X_test.shape)
    function = cas.Function("predict", [X], list(fitted.predict(X)))
    mean, std = function(X_test)
    expected_mean, expected_std = fitted.predict(X_test)
    np.testing.assert_allclose(mean.full().ravel(), expected_mean, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std.full().ravel(), expected_std, rtol=RTOL, atol=ATOL)


def test_predict_undoes_the_scalers(fitted):
    """With non-identity scalers, predict scales its inputs and unscales its outputs:
    the mean and std are in measured units. The noise is in measured units too, so
    the same GP in scaled units has a noise std 4 times larger (the output slope)."""
    # The same GP with scalers, the noise in measured units, and the test points in
    # measured units
    x_scaler = Scaler(np.array([2.0, 0.5]), np.array([0.1, -0.3]))
    y_scaler = Scaler(np.array([4.0]), np.array([1.0]))
    measured = dataclasses.replace(fitted.gp, noise_std=NOISE_STD / 4.0)
    scaled = FittedGP(measured, fitted.X, fitted.Y, x_scaler, y_scaler)
    X_measured = (X_test - x_scaler.intercept) / x_scaler.slope

    # The same prediction as the unscaled GP, unscaled
    mean, std = scaled.predict(X_measured)
    expected_mean, expected_std = fitted.predict(X_test)
    np.testing.assert_allclose(mean, (expected_mean - 1.0) / 4.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(std, expected_std / 4.0, rtol=RTOL, atol=ATOL)


def test_log_marginal_likelihood(fitted):
    """The log marginal likelihood is scipy's multivariate normal log density of the
    training data."""
    K = matern52(X_train, X_train) + (NOISE_STD**2 + JITTER) * np.eye(len(X_train))
    ref = multivariate_normal(np.zeros(len(X_train)), K).logpdf(Y_train)
    assert isinstance(fitted.log_marginal_likelihood(), float)
    np.testing.assert_allclose(
        fitted.log_marginal_likelihood(), ref, rtol=RTOL, atol=ATOL
    )


def test_gp_checks():
    """The GP and the fitted GP reject invalid arguments."""
    # `pytest.raises` fails the test unless the block raises that exception, with a
    # message matching `match`
    with pytest.raises(ValueError, match="noise_std"):
        GP(mean=Zero(), kernel=Matern52(), noise_std=-0.1)
    scaler = Scaler(np.ones(1), np.zeros(1))
    with pytest.raises(ValueError, match=r"got Matern52\(lengthscale=Param"):
        FittedGP(
            GP(mean=Zero(), kernel=Matern52(), noise_std=0.0),
            X_train,
            Y_train,
            scaler,
            scaler,
        )
    with pytest.raises(ValueError, match="lengthscale has 3 entries for 2 inputs"):
        GP(mean=Zero(), kernel=Matern52((0.1, 0.2, 0.3)), noise_std=0.0).fit(
            X_train, Y_train, rng
        )
    with pytest.raises(ValueError, match="lengthscale has 1 entries for 2 inputs"):
        GP(mean=Zero(), kernel=Scale(Scale(Matern52((0.1,)))), noise_std=0.0).fit(
            X_train, Y_train, rng
        )


def test_a_fitted_gp_rejects_a_param_whatever_the_width_of_its_inputs():
    """A lengthscale with ARD is a `Param` left even with no input columns, where it has
    no search entries."""
    scaler = Scaler(np.ones(1), np.zeros(1))
    with pytest.raises(ValueError, match="no Param left"):
        FittedGP(
            GP(mean=Zero(), kernel=Matern52(), noise_std=0.0),
            np.zeros((2, 0)),
            np.zeros(2),
            scaler,
            scaler,
        )


def test_inputs_of_different_widths_raise():
    """A shared lengthscale can't hide inputs of different widths: CasADi would broadcast
    a width that is a multiple of the other's, and return numbers."""
    # Setup: a GP of two inputs with one fixed lengthscale
    gp = GP(mean=Zero(), kernel=Matern52(0.3, ard=False), noise_std=0.0)
    fitted = gp.fit(X_train, Y_train, np.random.default_rng(0))

    # Check: twice the training columns, then half of them
    with pytest.raises(ValueError, match="X1 has 4 inputs and X2 has 2"):
        fitted.predict(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="X1 has 2 inputs and X2 has 1"):
        Matern52(0.3).covariance(cas.DM.zeros(3, 2), cas.DM.zeros(5, 1))


def test_hyperparameters_are_fitted_output_scales_first():
    """The search entries are each output scale, outermost first, then the lengthscale:
    one per input with ARD, one without, and none for a fixed value."""
    # Setup: an output scale at each level, and a lengthscale to fit
    outer = Param((1.0, 3.0))
    inner = Param((0.1, 10.0))
    lengthscale = Param((0.1, 1.0))

    # Check: with ARD, without it, and with a fixed lengthscale
    assert Scale(Scale(Matern52(lengthscale), inner), outer).params(2) == [
        outer,
        inner,
        lengthscale,
        lengthscale,
    ]
    assert Scale(Matern52(lengthscale, ard=False), outer).params(2) == [
        outer,
        lengthscale,
    ]
    assert Scale(Scale(Matern52(0.3), 2.0), outer).params(2) == [outer]


def test_with_values_replaces_each_param_in_fit_order():
    """Each `Param` takes its value in the order of `params`, and fixed values stay."""
    param = Param((0.1, 10.0))

    # An ARD lengthscale takes one value per input, a shared one a single value
    kernel = Scale(Scale(Matern52(param), 1.5), param)
    assert kernel.with_values([2.0, 0.3, 0.4], 2) == Scale(
        Scale(Matern52((0.3, 0.4)), 1.5), 2.0
    )
    shared = Scale(Matern52(param, ard=False), param)
    assert shared.with_values([2.0, 0.3], 2) == Scale(Matern52(0.3, ard=False), 2.0)


# Fitting: the regression baselines pin the fitted parameters of the illustrative
# example; these check what a fit is and what it draws

FIT_DATA = {
    "1d": (np.linspace(-2.0, 2.0, 4)[:, None], np.sin(np.linspace(-2.0, 2.0, 4))),
    "2d": (X_train, Y_train),
}


def test_fit_maximizes_the_log_marginal_likelihood():
    """The fitted parameters are at least as likely as any point of a log-spaced grid
    over the bounds of the default kernel."""
    X, Y = FIT_DATA["1d"]

    fitted = DEFAULT.fit(X, Y, np.random.default_rng(1))

    # The likelihood at every grid point, on the same scaled training data
    grid = [
        dataclasses.replace(
            fitted,
            gp=GP(
                mean=Zero(),
                kernel=Scale(Matern52((lengthscale,)), outputscale),
                noise_std=0.0,
            ),
        ).log_marginal_likelihood()
        for outputscale in np.geomspace(0.1, 10.0, 15)
        for lengthscale in np.geomspace(1e-3, 1e2, 15)
    ]
    assert fitted.log_marginal_likelihood() >= max(grid) - 1e-8


def test_fit_fills_in_params_and_keeps_fixed_values():
    """A fit replaces each `Param` with its fitted value, one per input for an ARD
    lengthscale, and leaves fixed parameters, the noise and the jitter as they were."""
    # A fixed output scale, a fitted lengthscale per input
    gp = GP(
        mean=Zero(),
        kernel=Scale(Matern52(), outputscale=2.0),
        noise_std=0.05,
        jitter=1e-7,
    )

    fitted = gp.fit(*FIT_DATA["2d"], np.random.default_rng(0))

    kernel = fitted.gp.kernel
    assert isinstance(kernel, Scale) and isinstance(kernel.kernel, Matern52)
    assert kernel.outputscale == 2.0
    lengthscale = kernel.kernel.lengthscale
    assert isinstance(lengthscale, tuple) and len(lengthscale) == 2
    assert all(1e-3 <= value <= 1e2 for value in lengthscale)
    assert (fitted.gp.noise_std, fitted.gp.jitter) == (0.05, 1e-7)

    # Without ARD, one lengthscale shared by both inputs
    shared = GP(mean=Zero(), kernel=Matern52(ard=False), noise_std=0.0).fit(
        *FIT_DATA["2d"], np.random.default_rng(0)
    )
    assert isinstance(shared.gp.kernel, Matern52)
    assert isinstance(shared.gp.kernel.lengthscale, float)


def test_noise_is_in_measured_units():
    """A fit with `noise_std` in measured units gives the same fitted GP as fitting
    the noise-free kernel with that noise converted to scaled units by hand: the
    training data's standard deviation divides it."""
    X, Y = FIT_DATA["2d"]
    noise_std = 0.3

    fitted = GP(mean=Zero(), kernel=Scale(Matern52()), noise_std=noise_std).fit(
        X, Y, np.random.default_rng(0)
    )

    # The reference: the fitted parameters, the noise scaled by hand into a GP on
    # data that is already scaled (identity scalers), as the log likelihood sees it
    identity = Scaler(np.ones(1), np.zeros(1))
    scaled_noise = dataclasses.replace(fitted.gp, noise_std=noise_std / Y.std())
    reference = FittedGP(scaled_noise, fitted.X, fitted.Y, fitted.x_scaler, identity)
    np.testing.assert_allclose(
        fitted.log_marginal_likelihood(),
        reference.log_marginal_likelihood(),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("data", FIT_DATA, ids=list(FIT_DATA))
def test_fit_draws_one_lhs_and_nothing_from_the_stream(data):
    """A fit's only draw is its Latin hypercube of starts, which takes the next
    spawned child and leaves the stream untouched."""
    X, Y = FIT_DATA[data]
    rng = np.random.default_rng(1)

    DEFAULT.fit(X, Y, rng)

    # The stream is where a fresh generator's is, and the next child is the second
    reference = np.random.default_rng(1)
    assert rng.bit_generator.state == reference.bit_generator.state
    reference.spawn(1)
    assert rng.spawn(1)[0].random() == reference.spawn(1)[0].random()


def test_fit_without_params_draws_nothing():
    """A GP with every parameter fixed is only conditioned on the data: it comes back
    as it was, and the generator spawns no child."""
    X, Y = FIT_DATA["1d"]
    gp = GP(mean=Zero(), kernel=Scale(Matern52(0.5), 1.5), noise_std=0.0)
    rng = np.random.default_rng(1)

    fitted = gp.fit(X, Y, rng)

    assert fitted.gp == gp
    assert rng.spawn(1)[0].random() == np.random.default_rng(1).spawn(1)[0].random()


def test_failed_fit_raises():
    """A fit whose every start fails raises, and the script decides what to do."""
    X, Y = FIT_DATA["1d"]
    no_iterations = Ipopt(solver_options={"print_level": 0, "max_iter": 0})

    with pytest.raises(NoSolutionError):
        DEFAULT.fit(X, Y, np.random.default_rng(0), solver=no_iterations)


def test_fit_converges_on_clustered_inputs(caplog):
    """Every start of a fit converges when the inputs cluster, as they do once BO
    converges: the jitter keeps the training covariance well conditioned (#107).

    With a jitter of 1e-10 these six points (iteration 5 of the illustrative example,
    seed 0, inputs to four digits, outputs sin(u) as its noise-free oracle gives) gave
    cond(K) of about 1e11, and 7 of 10 starts ran to IPOPT's iteration limit; this
    stands in for timing the fit, which would be flaky.

    `caplog` is a pytest fixture that captures log records during the test.
    """
    X = np.array([-1.8859, 1.3673, -1.9739, -1.591, -1.5762, -1.57])[:, None]

    DEFAULT.fit(X, np.sin(X[:, 0]), np.random.default_rng(0))

    assert "iteration limit" not in caplog.text


def test_nearly_singular_covariance_is_warned_about(caplog):
    """A GP whose training covariance is nearly singular (near-duplicate inputs and a
    tiny jitter) logs a warning giving the condition number and the jitter."""
    # Two inputs 1e-9 apart, with a jitter far below that
    X = np.array([0.0, 0.5, 0.5 + 1e-9, 1.0])[:, None]
    gp = GP(mean=Zero(), kernel=Matern52(0.3), noise_std=0.0, jitter=1e-14)

    gp.fit(X, np.sin(X[:, 0]), np.random.default_rng(0))

    # One warning about the conditioning, whose arguments are the condition number
    # (above the threshold) and the jitter
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    conditioning = [r for r in warnings if "condition number" in r.getMessage()]
    assert len(conditioning) == 1
    condition_number, jitter = conditioning[0].args
    assert condition_number > 1e10
    assert jitter == 1e-14


def test_well_conditioned_fit_is_not_warned_about(caplog):
    """A fit with the default jitter on spread-out inputs logs no warning about the
    condition number."""
    X, Y = FIT_DATA["1d"]

    DEFAULT.fit(X, Y, np.random.default_rng(0))

    assert "condition number" not in caplog.text


def test_param_is_fitted_within_its_bounds():
    """A `Param` with narrow bounds is fitted inside them."""
    gp = GP(mean=Zero(), kernel=Matern52(Param((0.4, 0.6))), noise_std=0.0)

    fitted = gp.fit(*FIT_DATA["2d"], np.random.default_rng(0))

    assert isinstance(fitted.gp.kernel, Matern52)
    lengthscale = fitted.gp.kernel.lengthscale
    assert isinstance(lengthscale, tuple)
    assert all(0.4 - 1e-8 <= value <= 0.6 + 1e-8 for value in lengthscale)


@pytest.mark.parametrize(
    ("X", "Y"),
    [
        (X_train[:, 0], Y_train),
        (X_train, Y_train[:, np.newaxis]),
        (X_train, Y_train[:-1]),
    ],
    ids=["X is 1-D", "Y is a column", "Y is short"],
)
def test_a_fit_rejects_shapes_that_do_not_match(X, Y):
    """A fit names the shapes it needs, rather than failing inside CasADi."""
    with pytest.raises(ValueError, match=r"X of shape \(n, d\) and Y of shape \(n,\)"):
        DEFAULT.fit(X, Y, np.random.default_rng(0))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_fit_rejects_values_that_are_not_finite(bad):
    """NaN or infinity in the data would give NaN predictions, so a fit refuses them."""
    Y = Y_train.copy()
    Y[2] = bad

    with pytest.raises(ValueError, match="finite data"):
        DEFAULT.fit(X_train, Y, np.random.default_rng(0))


@pytest.mark.parametrize(
    "X", [X_test[0], X_test.tolist()], ids=["one point, 1-D", "a list"]
)
def test_predict_needs_a_2d_array(fitted, X):
    """A numeric X must be a 2-D array, and the message names the shape."""
    with pytest.raises(ValueError, match=r"\(n_points, d\)"):
        fitted.predict(X)


def test_fit_returns_a_new_fitted_gp_and_leaves_the_gp_alone():
    """`fit` returns the fitted GP: the prior still has its `Param`s."""
    gp = GP(mean=Zero(), kernel=Matern52(), noise_std=0.0)

    gp.fit(X_train, Y_train, np.random.default_rng(0))

    assert gp.kernel == Matern52()
