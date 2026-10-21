"""Gaussian processes: the prior GP, fitting it to data, and predicting.

A prior GP is data (ADR 0008): `GP(mean, kernel, noise_std)`, with the mean and the
kernel from a closed set of frozen dataclasses (`Zero`; `Scale`, `Matern12`,
`Matern32`, `Matern52`, `SquaredExponential`). Each kernel declares its parameters as
fields: a number fixes the value, a `Param` is fitted within its bounds. Kernel
parameters are in scaled units (inputs scaled onto [0, 1], outputs standardized);
`noise_std`, the known observation noise, is in measured units.

`fit_gp` scales the training data, fits every `Param` by maximizing the log marginal
likelihood and returns the fitted GP, a `FittedGP`: the prior GP with every `Param`
replaced by its value, held with the scaled data and the scalers. `predict` gives its
noise-free mean and standard deviation in measured units, at numeric or symbolic
points.
"""

import dataclasses
import logging
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, assert_never

import casadi as cas
import numpy as np

from hybrid_bo.errors import NoSolutionError
from hybrid_bo.problem import Symbolic
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import NLP, Ipopt, multistart

_logger = logging.getLogger(__name__)

# Above this condition number of the fitted training covariance, `fit_gp` warns
CONDITION_THRESHOLD = 1e10


@dataclass(frozen=True)
class Param:
    """A parameter to fit: its initial value and its bounds, the hard limits of the
    fit and where its multistart draws the starts.

    With `log`, fitting searches over its logarithm, so the bounds must be positive.
    Raises `ValueError` if the initial value is outside the bounds, or `log` is set
    with a bound that isn't positive.
    """

    init: float
    bounds: tuple[float, float]
    log: bool = True

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not lower <= self.init <= upper:
            msg = f"Param init ({self.init}) must be inside the bounds {self.bounds}"
            raise ValueError(msg)
        if self.log and lower <= 0:
            msg = (
                f"A Param fitted in log space needs positive bounds, got "
                f"{self.bounds}; use log=False otherwise"
            )
            raise ValueError(msg)


# A lengthscale: one number for every input, one per input, or to be fitted
Lengthscale = float | tuple[float, ...] | Param


def _check_lengthscale(kernel: Any) -> None:
    """Raises `ValueError` if a kernel without ARD has one lengthscale per input."""
    if isinstance(kernel.lengthscale, tuple) and not kernel.ard:
        msg = f"A lengthscale per input needs ard=True, got {kernel!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Matern12:
    """The Matern-1/2 (absolute exponential) kernel.

    With `ard`, a fitted lengthscale has one entry per input; without, one shared by
    every input. Raises `ValueError` for a lengthscale per input without `ard`.
    """

    lengthscale: Lengthscale = Param(1.0, (1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class Matern32:
    """The Matern-3/2 kernel. Its lengthscale is as `Matern12`'s."""

    lengthscale: Lengthscale = Param(1.0, (1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class Matern52:
    """The Matern-5/2 kernel. Its lengthscale is as `Matern12`'s."""

    lengthscale: Lengthscale = Param(1.0, (1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class SquaredExponential:
    """The squared exponential (RBF, Gaussian) kernel. Its lengthscale is as
    `Matern12`'s."""

    lengthscale: Lengthscale = Param(1.0, (1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        _check_lengthscale(self)


@dataclass(frozen=True)
class Scale:
    """A kernel times an output scale, unsquared."""

    kernel: "Kernel"
    outputscale: float | Param = Param(1.0, (0.1, 10.0))


@dataclass(frozen=True)
class Zero:
    """The zero prior mean."""


Stationary = Matern12 | Matern32 | Matern52 | SquaredExponential
Kernel = Stationary | Scale
Mean = Zero
# The closed set, by name, for saving and loading
TYPES = (Param, Matern12, Matern32, Matern52, SquaredExponential, Scale, Zero)


@dataclass(frozen=True)
class GP:
    """A prior GP, before it sees any data: its mean, its kernel with each
    parameter fixed or to be fitted, and its observation noise.

    `noise_std` is the known standard deviation of the observation noise, in
    measured units (those of the training outputs); 0 is noise-free, and the fitted
    GP then nearly interpolates its data. `jitter` is added to the diagonal of the
    scaled training covariance, a nugget that keeps it well conditioned when inputs
    cluster (DESIGN.md). Raises `ValueError` if `noise_std` is negative.
    """

    mean: Mean
    kernel: Kernel
    noise_std: float = 0.0
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
    """A fitted GP: a prior GP with every parameter fitted to its training data, held
    with that data (scaled) and the scalers that produced it. It describes itself, so
    `predict` takes nothing else and a run saves it whole (ADR 0008).

    Raises `ValueError` if the GP has a `Param` left.
    """

    gp: GP
    X: np.ndarray  # (n, d), scaled
    Y: np.ndarray  # (n,), scaled
    x_scaler: Scaler
    y_scaler: Scaler

    def __post_init__(self) -> None:
        unfitted = [path for path, value, _ in _parameters(self.gp, 1) if _is(value)]
        if unfitted:
            msg = f"A fitted GP has no Param left; unfitted: {unfitted}"
            raise ValueError(msg)


def _is(value: Any) -> bool:
    """Whether a parameter's value is a `Param`, still to be fitted."""
    return isinstance(value, Param)


def _parameters(node: Any, n_inputs: int) -> list[tuple[str, Any, int | None]]:
    """The parameters of a GP, mean or kernel, in the order they are fitted: each as
    its dotted path from `node`, its value (a `Param`, a number, a tuple or a
    symbol) and its number of entries, None for one number (a lengthscale with ARD
    has `n_inputs`).

    A parameter is a field whose type admits a `Param`. A node's own parameters come
    before those of the means and kernels it holds.
    """
    hints = typing.get_type_hints(type(node))
    own, held = [], []
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if Param in typing.get_args(hints[field.name]):
            is_per_input = isinstance(node, Stationary) and node.ard
            own.append((field.name, value, n_inputs if is_per_input else None))
        elif isinstance(value, TYPES):
            held += [
                (f"{field.name}.{path}", inner, size)
                for path, inner, size in _parameters(value, n_inputs)
            ]
    return own + held


def _with_parameters(node: Any, values: dict[str, Any]) -> Any:
    """`node` with the parameters at the given dotted paths set to the given values."""
    changes = {}
    for field in dataclasses.fields(node):
        if field.name in values:
            changes[field.name] = values[field.name]
            continue
        prefix = f"{field.name}."
        inner = {
            path.removeprefix(prefix): value
            for path, value in values.items()
            if path.startswith(prefix)
        }
        if inner:
            changes[field.name] = _with_parameters(getattr(node, field.name), inner)
    return dataclasses.replace(node, **changes)


def _symbolic(value: Any) -> Any:
    """A parameter's value as a CasADi constant or column (a tuple as a column)."""
    return Symbolic(np.array(value)) if isinstance(value, tuple) else Symbolic(value)


def _per_column(values: Any, n_rows: int) -> Any:
    """`values` as a (n_rows, d) matrix with one row per point, for elementwise use
    with X (n_rows, d). A scalar is returned unchanged."""
    if values.numel() == 1:
        return values
    return cas.repmat(values.T, n_rows, 1)


def _squared_distance(X1: Any, X2: Any, lengthscale: Any) -> Any:
    """Squared distances between the rows of X1 and X2, each input divided by its
    lengthscale, shape (n1, n2).

    Clamped below at 1e-30, as gpytorch does, so the square root in the Matern kernels
    has a finite derivative at zero distance.
    """
    Z1 = X1 / _per_column(lengthscale, X1.shape[0])
    Z2 = X2 / _per_column(lengthscale, X2.shape[0])
    squared_norms = cas.repmat(cas.sum2(Z1**2), 1, Z2.shape[0]) + cas.repmat(
        cas.sum2(Z2**2).T, Z1.shape[0], 1
    )
    return cas.fmax(-2.0 * (Z1 @ Z2.T) + squared_norms, 1e-30)


def covariance(kernel: Kernel, X1: Any, X2: Any) -> Any:
    """The kernel's covariance between the rows of the CasADi matrices X1 (n1, d) and
    X2 (n2, d), shape (n1, n2). Each parameter must be a number or a CasADi symbol, not
    a `Param`."""
    match kernel:
        case Scale(kernel=inner, outputscale=outputscale):
            return _symbolic(outputscale) * covariance(inner, X1, X2)
        case Matern12(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, _symbolic(lengthscale))
            return cas.exp(-cas.sqrt(r2))
        case Matern32(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, _symbolic(lengthscale))
            return (1.0 + np.sqrt(3.0) * cas.sqrt(r2)) * cas.exp(-cas.sqrt(3.0 * r2))
        case Matern52(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, _symbolic(lengthscale))
            return (1.0 + np.sqrt(5.0) * cas.sqrt(r2) + 5.0 / 3.0 * r2) * cas.exp(
                -cas.sqrt(5.0 * r2)
            )
        case SquaredExponential(lengthscale=lengthscale):
            r2 = _squared_distance(X1, X2, _symbolic(lengthscale))
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


def _training_cholesky(gp: GP, X: Any, y_scaler: Scaler) -> Any:
    """The lower Cholesky factor of the training covariance."""
    # CasADi returns the upper factor
    return cas.chol(_training_covariance(gp, X, y_scaler)).T


def _log_marginal_likelihood(
    gp: GP, X: np.ndarray, Y: np.ndarray, y_scaler: Scaler
) -> Any:
    """The log marginal likelihood of the scaled data X (n, d), Y (n,) as a CasADi
    scalar in the GP's parameters (numbers or symbols)."""
    X_c, Y_c = Symbolic(X), Symbolic(Y)
    residual = Y_c - prior_mean(gp.mean, X_c)
    L = _training_cholesky(gp, X_c, y_scaler)
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
    `Param` (log ones in log space) with `multistart`, from `n_starts` Latin hypercube
    starts over their bounds; a lengthscale with ARD has one entry per column of X. A
    GP without a `Param` is only conditioned on the data, with no draws. If every
    start fails, it keeps the initial values and logs a warning. It also warns if the
    training covariance at the kept values has a condition number above
    `CONDITION_THRESHOLD`. Logs the time taken at debug level.

    Raises `ValueError` if a fixed lengthscale per input doesn't have one entry per
    column of X.
    """
    # Scale the data, and find the parameters to fit
    start = time.perf_counter()
    x_scaler, y_scaler = minmax(X), standard(Y[:, np.newaxis])
    X_scaled, Y_scaled = scale(x_scaler, X), scale(y_scaler, Y[:, np.newaxis]).ravel()
    parameters = _parameters(gp, X.shape[1])
    for path, value, _ in parameters:
        if isinstance(value, tuple) and len(value) != X.shape[1]:
            msg = f"{path} has {len(value)} entries for {X.shape[1]} inputs"
            raise ValueError(msg)
    free = [(path, param, size) for path, param, size in parameters if _is(param)]
    fitted_gp = gp
    if free:
        fitted_gp = _fit(gp, free, X_scaled, Y_scaled, y_scaler, rng, n_starts, solver)

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
    _logger.debug("Fitted the GP in %.2fs", time.perf_counter() - start)
    return FittedGP(fitted_gp, X_scaled, Y_scaled, x_scaler, y_scaler)


def _fit(
    gp: GP,
    free: list[tuple[str, Param, int | None]],
    X: np.ndarray,
    Y: np.ndarray,
    y_scaler: Scaler,
    rng: np.random.Generator,
    n_starts: int,
    solver: Ipopt,
) -> GP:
    """The prior GP with each `Param` (`free`, from `_parameters`) set to the value
    that maximizes the log marginal likelihood of the scaled data; see `fit_gp`."""
    # The search space: one entry per parameter entry, log ones as logarithms
    sizes = [1 if size is None else size for _, _, size in free]
    ends = np.cumsum(sizes)
    is_log = np.repeat([param.log for _, param, _ in free], sizes)
    lower = np.repeat([param.bounds[0] for _, param, _ in free], sizes)
    upper = np.repeat([param.bounds[1] for _, param, _ in free], sizes)
    init = np.repeat([param.init for _, param, _ in free], sizes)

    def to_search_space(values: np.ndarray) -> np.ndarray:
        """Parameter entries as search-space entries: logarithms for log ones."""
        return np.where(is_log, np.log(np.where(is_log, values, 1.0)), values)

    def with_entries(entries: Any, as_value: Callable[[Any, int | None], Any]) -> GP:
        """The GP with each free parameter set to its slice of `entries`."""
        return _with_parameters(
            gp,
            {
                path: as_value(entries[end - n : end], size)
                for (path, _, size), n, end in zip(free, sizes, ends, strict=True)
            },
        )

    # The negative log marginal likelihood over the search space
    w = Symbolic.sym("w", len(is_log))
    entries = [cas.exp(w[i]) if is_log[i] else w[i] for i in range(len(is_log))]
    symbolic_gp = with_entries(cas.vertcat(*entries), lambda value, _: value)
    objective = -_log_marginal_likelihood(symbolic_gp, X, Y, y_scaler)
    nlp = NLP(
        cas.Function("negative_log_marginal_likelihood", [w], [objective]),
        to_search_space(lower),
        to_search_space(upper),
    )

    # Multistart from Latin hypercube starts over the bounds
    starts = lhs(nlp.lower, nlp.upper, n_starts, rng)
    try:
        w_best = multistart(nlp, starts, solver).w
    except NoSolutionError:
        _logger.warning("Every GP fit failed; keeping the initial values")
        w_best = to_search_space(init)

    # Numbers: one per input as a tuple, else one float
    def as_number(value: np.ndarray, size: int | None) -> float | tuple[float, ...]:
        """A fitted parameter's entries as its value."""
        return float(value[0]) if size is None else tuple(map(float, value))

    return with_entries(np.where(is_log, np.exp(w_best), w_best), as_number)


def predict(fitted: FittedGP, X: Any) -> tuple[Any, Any]:
    """The fitted GP's conditional mean and standard deviation at the rows of X (n, d),
    in measured units. They are noise-free: the standard deviation is of the latent
    output, with no observation noise added.

    For a numpy X, returns two arrays (n,). For a symbolic X (SX or MX; the conditional
    mean and variance themselves are built in SX), returns two (n, 1) CasADi columns,
    for building acquisition problems.
    """
    # The conditional mean and variance at symbols for X, with the parameters and
    # data as constants
    gp = fitted.gp
    X_train, Y_train = Symbolic(fitted.X), Symbolic(fitted.Y)
    X_symbols = Symbolic.sym("X", *X.shape)
    X_scaled = scale(fitted.x_scaler, X_symbols)
    L = _training_cholesky(gp, X_train, fitted.y_scaler)
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
    mean, std = function(X)
    if isinstance(X, np.ndarray):
        return mean.full().ravel(), std.full().ravel()
    return mean, std
