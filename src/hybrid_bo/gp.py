"""Gaussian processes: the prior GP, fitting it to data, and predicting.

A prior GP is data: `GP(mean=..., kernel=..., noise_std=...)`, with the
mean and the kernel from a closed set of frozen dataclasses (`Zero`; `Scale`, `Matern12`,
`Matern32`, `Matern52`, `SquaredExponential`). Each kernel declares its hyperparameters as
fields: a number fixes the value, a `Param` is fitted within its bounds. Kernel
hyperparameters are in scaled units (inputs scaled onto [0, 1], outputs standardized);
`noise_std`, the known observation noise, is in measured units.

`fit_gp` scales the training data, fits every `Param` by maximizing the log marginal
likelihood and returns the fitted GP, a `FittedGP`: the prior GP with every `Param`
replaced by its value, held with the scaled data and the scalers. `predict` gives its
noise-free mean and standard deviation in measured units, at numeric or symbolic
points.
"""

import dataclasses
import logging
import typing
from dataclasses import dataclass
from typing import Any, assert_never

import casadi as cas
import numpy as np

from hybrid_bo.problem import Symbolic
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import NLP, Ipopt, solve_with_multistart

_logger = logging.getLogger(__name__)

# Above this condition number of the fitted training covariance, `fit_gp` warns
CONDITION_THRESHOLD = 1e10


@dataclass(frozen=True)
class Param:
    """A hyperparameter to fit: its bounds, the hard limits of the fit and where its
    multistart draws the starts.

    With `log`, fitting searches over its logarithm, so the bounds must be positive.
    Raises `ValueError` if `log` is set with a bound that isn't positive.
    """

    bounds: tuple[float, float]
    log: bool = True

    def __post_init__(self) -> None:
        if self.log and self.bounds[0] <= 0:
            msg = (
                f"A Param fitted in log space needs positive bounds, got "
                f"{self.bounds}; use log=False otherwise"
            )
            raise ValueError(msg)


# A lengthscale: one number for every input, one per input, or to be fitted
Lengthscale = float | tuple[float, ...] | Param


class _Scalable:
    """Gives a kernel `*` with a number or a `Param`, on either side: shorthand for
    `Scale`, which it builds (`Param((0.1, 10.0)) * Matern52()` is
    `Scale(Matern52(), Param((0.1, 10.0)))`). There is no product of kernels."""

    def __mul__(self, outputscale: "float | Param") -> "Scale":
        """The kernel times an output scale, as a `Scale`.

        Raises `TypeError` unless `outputscale` is a number or a `Param`.
        """
        is_number = isinstance(outputscale, int | float) and not isinstance(
            outputscale, bool
        )
        if not (is_number or isinstance(outputscale, Param)):
            msg = (
                f"A kernel can be multiplied only by a number or a Param (an output "
                f"scale), not {type(outputscale).__name__}"
            )
            raise TypeError(msg)
        value = outputscale if isinstance(outputscale, Param) else float(outputscale)
        return Scale(typing.cast("Kernel", self), value)

    def __rmul__(self, outputscale: "float | Param") -> "Scale":
        """`outputscale * kernel`, the same as `kernel * outputscale`."""
        return self.__mul__(outputscale)


def _check_lengthscale(kernel: Any) -> None:
    """Raises `ValueError` if a kernel without ARD has one lengthscale per input."""
    if isinstance(kernel.lengthscale, tuple) and not kernel.ard:
        msg = f"A lengthscale per input needs ard=True, got {kernel!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Matern12(_Scalable):
    """The Matern-1/2 (absolute exponential) kernel.

    With `ard`, a fitted lengthscale has one entry per input; without, one shared by
    every input. Raises `ValueError` for a lengthscale per input without `ard`.
    """

    lengthscale: Lengthscale = Param((1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class Matern32(_Scalable):
    """The Matern-3/2 kernel. Its lengthscale is as `Matern12`'s."""

    lengthscale: Lengthscale = Param((1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class Matern52(_Scalable):
    """The Matern-5/2 kernel. Its lengthscale is as `Matern12`'s."""

    lengthscale: Lengthscale = Param((1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class SquaredExponential(_Scalable):
    """The squared exponential (RBF, Gaussian) kernel. Its lengthscale is as
    `Matern12`'s."""

    lengthscale: Lengthscale = Param((1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class Scale(_Scalable):
    """A kernel times an output scale, unsquared; `outputscale * kernel` builds it."""

    kernel: "Kernel"
    outputscale: float | Param = Param((0.1, 10.0))


@dataclass(frozen=True)
class Zero:
    """The zero prior mean."""


Stationary = Matern12 | Matern32 | Matern52 | SquaredExponential
Kernel = Stationary | Scale
Mean = Zero


@dataclass(frozen=True, kw_only=True)
class GP:
    """A prior GP, before it sees any data: its mean, its kernel with each
    hyperparameter fixed or to be fitted, and its observation noise.

    `noise_std` is the known standard deviation of the observation noise, in
    measured units (those of the training outputs); 0 is noise-free, and the fitted
    GP then nearly interpolates its data. `jitter` is added to the diagonal of the
    scaled training covariance, a nugget that keeps it well conditioned when inputs
    cluster. Raises `ValueError` if `noise_std` is negative.
    """

    mean: Mean
    kernel: Kernel
    noise_std: float
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        if not self.noise_std >= 0:
            msg = f"noise_std must be at least 0, got {self.noise_std}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Scaler:
    """An affine map per column: scaled = slope * value + intercept."""

    slope: np.ndarray  # (d,)
    intercept: np.ndarray  # (d,)


@dataclass(frozen=True)
class FittedGP:
    """A fitted GP: a prior GP with every hyperparameter fitted to its training data, held
    with that data (scaled) and the scalers that produced it. It describes itself, so
    `predict` takes nothing else.

    Raises `ValueError` if the GP has a `Param` left.
    """

    gp: GP
    X: np.ndarray  # (n, d), scaled
    Y: np.ndarray  # (n,), scaled
    x_scaler: Scaler
    y_scaler: Scaler

    def __post_init__(self) -> None:
        if _params(self.gp.kernel, 1):
            msg = f"A fitted GP has no Param left, got {self.gp.kernel!r}"
            raise ValueError(msg)


def _params(kernel: Kernel, n_inputs: int) -> list[Param]:
    """One `Param` per search entry of the kernel, in the order they are fitted: an
    output scale before the kernel it scales, and a lengthscale with ARD once per input."""
    match kernel:
        case Scale(kernel=inner, outputscale=outputscale):
            own = [outputscale] if isinstance(outputscale, Param) else []
            return own + _params(inner, n_inputs)
        case Matern12() | Matern32() | Matern52() | SquaredExponential():
            if not isinstance(kernel.lengthscale, Param):
                return []
            return [kernel.lengthscale] * (n_inputs if kernel.ard else 1)
        case _:
            assert_never(kernel)


def _with_values(kernel: Kernel, values: list[Any], n_inputs: int) -> Kernel:
    """The kernel with each `Param` replaced by its value. `values` has a number or a
    CasADi scalar per entry of `_params`, in that order; a lengthscale with ARD becomes
    a tuple of one per input."""
    match kernel:
        case Scale(kernel=inner, outputscale=Param()):
            return Scale(_with_values(inner, values[1:], n_inputs), values[0])
        case Scale(kernel=inner, outputscale=outputscale):
            return Scale(_with_values(inner, values, n_inputs), outputscale)
        case Matern12() | Matern32() | Matern52() | SquaredExponential():
            if not isinstance(kernel.lengthscale, Param):
                return kernel
            value = tuple(values[:n_inputs]) if kernel.ard else values[0]
            return dataclasses.replace(kernel, lengthscale=value)
        case _:
            assert_never(kernel)


def _squared_distance(X1: Any, X2: Any, lengthscale: Any) -> Any:
    """Squared distances between the rows of X1 and X2, each input divided by its
    lengthscale, shape (n1, n2). `lengthscale` is one value for every input, or a tuple
    of one per input; a value is a number or a CasADi scalar.

    Clamped below at 1e-30, as gpytorch does, so the square root in the Matern kernels
    has a finite derivative at zero distance.

    Raises `ValueError` if X1 and X2 have different columns, or a tuple doesn't have one
    entry per column.
    """
    # One lengthscale per input, as a row repeated for every point
    n_inputs = X1.shape[1]
    if X2.shape[1] != n_inputs:
        msg = f"X1 has {n_inputs} inputs and X2 has {X2.shape[1]}"
        raise ValueError(msg)
    lengthscales = (
        lengthscale if isinstance(lengthscale, tuple) else (lengthscale,) * n_inputs
    )
    if len(lengthscales) != n_inputs:
        msg = f"lengthscale has {len(lengthscales)} entries for {n_inputs} inputs"
        raise ValueError(msg)
    row = cas.horzcat(*lengthscales)

    Z1 = X1 / cas.repmat(row, X1.shape[0], 1)
    Z2 = X2 / cas.repmat(row, X2.shape[0], 1)
    squared_norms = cas.repmat(cas.sum2(Z1**2), 1, Z2.shape[0]) + cas.repmat(
        cas.sum2(Z2**2).T, Z1.shape[0], 1
    )
    return cas.fmax(-2.0 * (Z1 @ Z2.T) + squared_norms, 1e-30)


def covariance(kernel: Kernel, X1: Any, X2: Any) -> Any:
    """The kernel's covariance between the rows of the CasADi matrices X1 (n1, d) and
    X2 (n2, d), shape (n1, n2). Each hyperparameter must be a number or a CasADi scalar, not
    a `Param`; a lengthscale may be a tuple of them, one per input.

    Raises `ValueError` if X1 and X2 have different columns, or that tuple doesn't have
    one entry per input.
    """
    match kernel:
        case Scale(kernel=inner, outputscale=outputscale):
            return outputscale * covariance(inner, X1, X2)
        case Matern12(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, lengthscale)
            return cas.exp(-cas.sqrt(r2))
        case Matern32(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, lengthscale)
            return (1.0 + np.sqrt(3.0) * cas.sqrt(r2)) * cas.exp(-cas.sqrt(3.0 * r2))
        case Matern52(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, lengthscale)
            return (1.0 + np.sqrt(5.0) * cas.sqrt(r2) + 5.0 / 3.0 * r2) * cas.exp(
                -cas.sqrt(5.0 * r2)
            )
        case SquaredExponential(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, lengthscale)
            return cas.exp(-0.5 * r2)
        case _:
            assert_never(kernel)


def prior_mean(mean: Mean, X: Any) -> Any:
    """The prior mean at the rows of the CasADi matrix X (n, d), a (n, 1) column."""
    match mean:
        case Zero():
            return cas.DM.zeros(X.shape[0], 1)
        case _:
            assert_never(mean)


def minmax(values: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> Scaler:
    """The scaler that maps each column of `values` (n, d) onto [lower, upper].

    A constant column is shifted to `lower` instead.
    """
    low, high = values.min(0), values.max(0)
    is_constant = low == high
    span = np.where(is_constant, 1.0, high - low)

    # Evaluated in this order so that [0, 1] gives the pre-refactor values bit for
    # bit, which the regression baselines depend on
    slope = np.where(is_constant, 1.0, (upper - lower) / span)
    intercept = np.where(is_constant, lower - low, lower - low * (upper - lower) / span)
    return Scaler(slope, intercept)


def standard(values: np.ndarray) -> Scaler:
    """The scaler that gives each column of `values` (n, d) zero mean and unit standard
    deviation (`ddof=0`). A constant column is only shifted."""
    mean, std = values.mean(0), values.std(0)
    std = np.where(std == 0.0, 1.0, std)
    return Scaler(1 / std, -mean / std)


def scale(scaler: Scaler, values: Any) -> Any:
    """Scales the rows of `values` (n, d), numeric or CasADi."""
    n = values.shape[0]
    return values * np.tile(scaler.slope, (n, 1)) + np.tile(scaler.intercept, (n, 1))


def unscale(scaler: Scaler, scaled: Any) -> Any:
    """Undoes `scale` on the rows of `scaled` (n, d), numeric or CasADi."""
    n = scaled.shape[0]
    return (scaled - np.tile(scaler.intercept, (n, 1))) / np.tile(scaler.slope, (n, 1))


def _training_covariance(gp: GP, X: Any, y_scaler: Scaler) -> Any:
    """The scaled training covariance: kernel plus noise variance (`noise_std` in
    scaled units) plus jitter on the diagonal."""
    noise_variance = (gp.noise_std * y_scaler.slope[0]) ** 2
    return covariance(gp.kernel, X, X) + (noise_variance + gp.jitter) * Symbolic.eye(
        X.shape[0]
    )


def _log_marginal_likelihood(
    gp: GP, X: np.ndarray, Y: np.ndarray, y_scaler: Scaler
) -> Any:
    """The log marginal likelihood of the scaled data X (n, d), Y (n,) as a CasADi
    scalar in the GP's hyperparameters (numbers or symbols)."""
    X_c, Y_c = Symbolic(X), Symbolic(Y)
    residual = Y_c - prior_mean(gp.mean, X_c)
    # CasADi's `chol` gives the upper factor
    L = cas.chol(_training_covariance(gp, X_c, y_scaler)).T
    alpha = cas.solve(L.T, cas.solve(L, residual))
    return -0.5 * (
        residual.T @ alpha
        + 2 * cas.sum1(cas.log(cas.diag(L)))
        + len(Y) * np.log(2 * np.pi)
    )


def log_marginal_likelihood(fitted: FittedGP) -> float:
    """The log marginal likelihood of the fitted GP on its (scaled) training data: the
    quantity `fit_gp` maximizes."""
    return float(
        cas.evalf(
            _log_marginal_likelihood(fitted.gp, fitted.X, fitted.Y, fitted.y_scaler)
        )
    )


def fit_gp(
    gp: GP,
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> FittedGP:
    """Fits the prior GP to inputs X (n, d) and outputs Y (n,), giving the fitted GP.

    Scales the inputs onto [0, 1] (`minmax`) and the outputs to zero mean and unit
    variance (`standard`), then maximizes the log marginal likelihood over every
    `Param` (log ones in log space) with `solve_with_multistart`, from `n_starts` Latin
    hypercube starts over their bounds; a lengthscale with ARD has one entry per column
    of X. A GP without a `Param` is only conditioned on the data, with no draws. It warns if
    the training covariance at the kept values has a condition number above
    `CONDITION_THRESHOLD`. Logs the time taken at debug level.

    Raises `NoSolutionError` if every start fails, and `ValueError` if a fixed lengthscale
    per input doesn't have one entry per
    column of X.
    """
    # Scale the data, and list what there is to fit
    x_scaler, y_scaler = minmax(X), standard(Y[:, np.newaxis])
    X_scaled, Y_scaled = scale(x_scaler, X), scale(y_scaler, Y[:, np.newaxis]).ravel()
    n_inputs = X.shape[1]
    params = _params(gp.kernel, n_inputs)
    fitted_gp = gp
    if params:
        # The search space: one entry per entry of `params`, log ones as logarithms
        is_log = np.array([param.log for param in params])
        lower = np.array([param.bounds[0] for param in params])
        upper = np.array([param.bounds[1] for param in params])

        def to_search_space(values: np.ndarray) -> np.ndarray:
            """Hyperparameter entries as search-space entries: logarithms for log ones."""
            return np.where(is_log, np.log(np.where(is_log, values, 1.0)), values)

        # The negative log marginal likelihood over the search space
        w = Symbolic.sym("w", len(params))
        symbols = [cas.exp(w[i]) if is_log[i] else w[i] for i in range(len(params))]
        symbolic_gp = dataclasses.replace(
            gp, kernel=_with_values(gp.kernel, symbols, n_inputs)
        )
        objective = -_log_marginal_likelihood(symbolic_gp, X_scaled, Y_scaled, y_scaler)
        nlp = NLP(
            cas.Function("negative_log_marginal_likelihood", [w], [objective]),
            to_search_space(lower),
            to_search_space(upper),
        )

        # Multistart from Latin hypercube starts over the bounds
        starts = lhs(nlp.lower, nlp.upper, n_starts, rng)
        w_best = solve_with_multistart(nlp, starts, solver).w

        # Back from the search space, to the numbers the fitted GP holds
        values = np.where(is_log, np.exp(w_best), w_best).tolist()
        fitted_gp = dataclasses.replace(
            gp, kernel=_with_values(gp.kernel, values, n_inputs)
        )

    # Warn if the training covariance is nearly singular at those values (a
    # non-finite one counts as singular; the SVD behind `cond` would fail on it)
    K = cas.evalf(_training_covariance(fitted_gp, Symbolic(X_scaled), y_scaler))
    K = K.full()
    condition_number = np.linalg.cond(K) if np.isfinite(K).all() else np.inf
    if condition_number > CONDITION_THRESHOLD:
        _logger.warning(
            "The fitted GP's training covariance is nearly singular: condition "
            "number %.3g with jitter %g",
            condition_number,
            gp.jitter,
        )
    return FittedGP(fitted_gp, X_scaled, Y_scaled, x_scaler, y_scaler)


def predict(fitted: FittedGP, X: Any) -> tuple[Any, Any]:
    """The fitted GP's conditional mean and standard deviation at the rows of X (n, d),
    in measured units. They are noise-free: the standard deviation is of the latent
    output, with no observation noise added.

    For a numpy X, returns two arrays (n,). For a symbolic X (SX or MX; the conditional
    mean and variance themselves are built in SX), returns two (n, 1) CasADi columns,
    for building acquisition problems.

    Raises `ValueError` if X doesn't have the columns of the training inputs.
    """
    # The conditional mean and variance at symbols for X, with the hyperparameters and
    # data as constants. A numpy X is predicted one row at a time, mapped over its rows:
    # predicting all rows at once builds their whole covariance matrix
    gp = fitted.gp
    X_train, Y_train = Symbolic(fitted.X), Symbolic(fitted.Y)
    is_numeric = isinstance(X, np.ndarray)
    X_symbols = Symbolic.sym("X", 1 if is_numeric else X.shape[0], X.shape[1])
    X_scaled = scale(fitted.x_scaler, X_symbols)
    # CasADi's `chol` gives the upper factor
    L = cas.chol(_training_covariance(gp, X_train, fitted.y_scaler)).T
    alpha = cas.solve(L, Y_train - prior_mean(gp.mean, X_train))
    beta = cas.solve(L, covariance(gp.kernel, X_scaled, X_train).T)
    mean = prior_mean(gp.mean, X_scaled) + beta.T @ alpha
    variance = cas.diag(covariance(gp.kernel, X_scaled, X_scaled)) - cas.sum1(beta**2).T

    # Back to measured units, evaluated at X
    function = cas.Function(
        "predict",
        [X_symbols],
        [unscale(fitted.y_scaler, mean), cas.sqrt(variance) / fitted.y_scaler.slope[0]],
    )
    if is_numeric:
        mean, std = function.map(len(X))(X.reshape(1, -1))
        return mean.full().ravel(), std.full().ravel()
    return function(X)
