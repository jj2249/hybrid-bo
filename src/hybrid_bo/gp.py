"""Gaussian processes: the prior GP, fitting it to data, and predicting.

A `PriorGP` is the prior: named hyperparameters (`Hyper`), and a kernel and a mean
written as functions of them. `fit_gp` scales the training data, fits the
hyperparameters by maximizing the log marginal likelihood and returns the fitted GP, a
`FittedGP`, which is data only (ADR 0003). `predict` gives the fitted GP's noise-free
mean and standard deviation in original units, at numeric or symbolic points.

The hyperparameter named `noise_std`, if the prior GP declares one, is the standard
deviation of the observation noise. Without it the fitted GP interpolates the training
data; only `jitter` is added to the diagonal.
"""

import keyword
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo import kernels
from hybrid_bo.errors import NoSolutionError
from hybrid_bo.problem import Symbolic
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import NLP, Ipopt, multistart

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hyper:
    """A named hyperparameter: its initial value, its bounds for fitting, and its
    number of entries (e.g. one lengthscale per input).

    With `log`, fitting searches over its logarithm, so the bounds must be positive.
    The kernel and mean read it as `h.<name>`, a (size, 1) column.

    Raises `ValueError` if the name isn't an identifier, the size isn't positive, the
    initial value is outside the bounds, or `log` is set with a bound that isn't
    positive.
    """

    name: str
    init: float
    bounds: tuple[float, float]
    size: int = 1
    log: bool = True

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not self.name.isidentifier() or keyword.iskeyword(self.name):
            msg = (
                f"Hyper name {self.name!r} must be a Python identifier (h.{self.name})"
            )
            raise ValueError(msg)
        if self.size < 1:
            msg = f"Hyper {self.name!r}: size must be positive, got {self.size}"
            raise ValueError(msg)
        if not lower <= self.init <= upper:
            msg = (
                f"Hyper {self.name!r}: init ({self.init}) must be inside the bounds "
                f"{self.bounds}"
            )
            raise ValueError(msg)
        if self.log and lower <= 0:
            msg = (
                f"Hyper {self.name!r} is fitted in log space, so its bounds "
                f"{self.bounds} must be positive; use log=False otherwise"
            )
            raise ValueError(msg)


def zero_mean(h: Any, X: Any) -> Any:
    """The zero prior mean, a (n, 1) column for the rows of X."""
    return cas.DM.zeros(X.shape[0], 1)


@dataclass(frozen=True)
class PriorGP:
    """A prior GP, before it sees any data: the hyperparameters, in the order they are
    optimized, and the kernel `(h, X1, X2) -> (n1, n2)` and mean `(h, X) -> (n, 1)` as
    functions of them.

    `jitter` is added to the diagonal of the training covariance: a nugget that keeps
    it well conditioned when inputs cluster, as they do once a run converges. Raises `ValueError` if a hyperparameter name repeats.
    """

    hyperparameters: tuple[Hyper, ...]
    kernel: Callable[[Any, Any, Any], Any]
    mean: Callable[[Any, Any], Any] = zero_mean
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "hyperparameters", tuple(self.hyperparameters))
        names = [hyper.name for hyper in self.hyperparameters]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            msg = (
                f"PriorGP hyperparameters must not repeat a name; repeated: {repeated}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class FittedGP:
    """A fitted GP, data only: a prior GP conditioned on its training data, at
    hyperparameters fitted by maximizing the log marginal likelihood.

    Holds the hyperparameter values by name, each of shape (size,), the scaled
    training data, and the scalers that produced it. The prior GP itself isn't stored
    (ADR 0003), so `predict` takes both.
    """

    hyperparameters: dict[str, np.ndarray]
    X: np.ndarray  # (n, d), scaled
    Y: np.ndarray  # (n,), scaled
    x_scaler: "Scaler"
    y_scaler: "Scaler"


def default_gp(n_inputs: int) -> PriorGP:
    """A noise-free prior GP with an output scale times a Matern-5/2 kernel with one
    lengthscale per input, and a zero mean."""
    return PriorGP(
        hyperparameters=(
            Hyper("outputscale", 1.0, bounds=(0.1, 10.0)),
            Hyper("lengthscale", 1.0, bounds=(1e-3, 1e2), size=n_inputs),
        ),
        kernel=lambda h, X1, X2: (
            h.outputscale * kernels.matern52(X1, X2, h.lengthscale)
        ),
    )


@dataclass(frozen=True)
class Scaler:
    """An affine map per column: scaled = slope * value + intercept."""

    slope: np.ndarray  # (d,)
    intercept: np.ndarray  # (d,)


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


def _namespace(values: dict[str, Any]) -> SimpleNamespace:
    """The `h` passed to the kernel and mean, with each value (numeric or symbolic)
    as a CasADi column."""
    return SimpleNamespace(**{name: Symbolic(value) for name, value in values.items()})


def _gp_record(gp: PriorGP, input_names: Sequence[str]) -> dict[str, str]:
    """The prior GP as text, for a run's record: its hyperparameters and jitter, and
    its kernel and mean as expressions in the hyperparameters and the inputs (a
    primed input belongs to the kernel's second point)."""
    h = _namespace(
        {
            hyper.name: Symbolic.sym(hyper.name, hyper.size)
            for hyper in gp.hyperparameters
        }
    )
    X1 = cas.horzcat(*(Symbolic.sym(name) for name in input_names))
    X2 = cas.horzcat(*(Symbolic.sym(f"{name}'") for name in input_names))
    return {
        "gp": f"hyperparameters={gp.hyperparameters!r}, jitter={gp.jitter}",
        "kernel": str(gp.kernel(h, X1, X2)),
        "mean": str(gp.mean(h, X1)),
    }


def _split(hypers: Sequence[Hyper], values: Any) -> dict[str, Any]:
    """The entries of `values`, one per hyperparameter entry in declaration order,
    grouped by hyperparameter name."""
    ends = np.cumsum([hyper.size for hyper in hypers])
    return {
        hyper.name: values[end - hyper.size : end]
        for hyper, end in zip(hypers, ends, strict=True)
    }


def _training_cholesky(gp: PriorGP, h: Any, X: Any) -> Any:
    """The lower Cholesky factor of the training covariance, kernel plus noise
    variance plus jitter on the diagonal."""
    noise_variance = h.noise_std**2 if hasattr(h, "noise_std") else 0.0
    K = gp.kernel(h, X, X) + (noise_variance + gp.jitter) * Symbolic.eye(X.shape[0])
    return cas.chol(K).T  # CasADi returns the upper factor


def _log_marginal_likelihood(gp: PriorGP, h: Any, X: np.ndarray, Y: np.ndarray) -> Any:
    """The log marginal likelihood of the data X (n, d), Y (n,) as a CasADi scalar in
    the hyperparameters `h`."""
    X_c, Y_c = Symbolic(X), Symbolic(Y)
    residual = Y_c - gp.mean(h, X_c)
    L = _training_cholesky(gp, h, X_c)
    alpha = cas.solve(L.T, cas.solve(L, residual))
    return -0.5 * (
        residual.T @ alpha
        + 2 * cas.sum1(cas.log(cas.diag(L)))
        + len(Y) * np.log(2 * np.pi)
    )


def log_marginal_likelihood(gp: PriorGP, fitted: FittedGP) -> float:
    """The log marginal likelihood of the prior GP `gp` at the fitted GP's
    hyperparameters, on its (scaled) training data: the quantity `fit_gp` maximizes."""
    h = _namespace(fitted.hyperparameters)
    return float(cas.evalf(_log_marginal_likelihood(gp, h, fitted.X, fitted.Y)))


def fit_gp(
    gp: PriorGP,
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> FittedGP:
    """Fits the prior GP to inputs X (n, d) and outputs Y (n,), giving the fitted GP.

    Scales the inputs onto [0, 1] (`minmax`) and the outputs to zero mean and unit
    variance (`standard`), then maximizes the log marginal likelihood over the
    hyperparameters (log ones in log space) with `multistart`, from `n_starts` Latin
    hypercube starts over their bounds. If every start fails, it keeps the initial
    values and logs a warning.
    """
    # Scale the data
    x_scaler, y_scaler = minmax(X), standard(Y[:, np.newaxis])
    X_scaled, Y_scaled = scale(x_scaler, X), scale(y_scaler, Y[:, np.newaxis]).ravel()
    hypers = gp.hyperparameters
    if not hypers:
        return FittedGP({}, X_scaled, Y_scaled, x_scaler, y_scaler)

    # The search space: one entry per hyperparameter entry, log ones as logarithms
    sizes = [hyper.size for hyper in hypers]
    is_log = np.repeat([hyper.log for hyper in hypers], sizes)
    lower = np.repeat([hyper.bounds[0] for hyper in hypers], sizes)
    upper = np.repeat([hyper.bounds[1] for hyper in hypers], sizes)
    init = np.repeat([hyper.init for hyper in hypers], sizes)

    def to_search_space(values: np.ndarray) -> np.ndarray:
        """Hyperparameter entries as search-space entries: logarithms for log ones."""
        return np.where(is_log, np.log(np.where(is_log, values, 1.0)), values)

    # The negative log marginal likelihood over the search space
    w = Symbolic.sym("w", len(is_log))
    entries = [cas.exp(w[i]) if is_log[i] else w[i] for i in range(len(is_log))]
    h = _namespace(_split(hypers, cas.vertcat(*entries)))
    objective = -_log_marginal_likelihood(gp, h, X_scaled, Y_scaled)
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
    values = np.where(is_log, np.exp(w_best), w_best)
    return FittedGP(_split(hypers, values), X_scaled, Y_scaled, x_scaler, y_scaler)


def predict(gp: PriorGP, fitted: FittedGP, X: Any) -> tuple[Any, Any]:
    """The fitted GP's conditional mean and standard deviation at the rows of X (n, d),
    in original units. They are noise-free: the standard deviation is of the latent
    output, with no observation noise added.

    `gp` is the prior GP that `fitted` was fitted from.

    For a numpy X, returns two arrays (n,). For a symbolic X (SX or MX; the conditional
    mean and variance themselves are built in SX), returns two (n, 1) CasADi columns,
    for building acquisition problems.
    """
    # The conditional mean and variance at symbols for X, with the hyperparameters and
    # data as constants
    h = _namespace(fitted.hyperparameters)
    X_train, Y_train = Symbolic(fitted.X), Symbolic(fitted.Y)
    X_symbols = Symbolic.sym("X", *X.shape)
    X_scaled = scale(fitted.x_scaler, X_symbols)
    L = _training_cholesky(gp, h, X_train)
    alpha = cas.solve(L, Y_train - gp.mean(h, X_train))
    beta = cas.solve(L, gp.kernel(h, X_scaled, X_train).T)
    mean = gp.mean(h, X_scaled) + beta.T @ alpha
    variance = cas.diag(gp.kernel(h, X_scaled, X_scaled)) - cas.sum1(beta**2).T

    # Back to original units, evaluated at X
    function = cas.Function(
        "predict",
        [X_symbols],
        [unscale(fitted.y_scaler, mean), cas.sqrt(variance) / fitted.y_scaler.slope[0]],
    )
    mean, std = function(X)
    if isinstance(X, np.ndarray):
        return mean.full().ravel(), std.full().ravel()
    return mean, std
