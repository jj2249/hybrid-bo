"""Gaussian processes: the prior GP, fitting it to data, and predicting.

A prior GP is data: `GP(mean=..., kernel=..., noise_std=...)`, with the mean and the kernel
from a closed set of frozen dataclasses (`Zero`; `Scale`, `Matern12`, `Matern32`,
`Matern52`, `SquaredExponential`). Each kernel holds its hyperparameters as fields, where
a number fixes the value and a `Param` is fitted within its bounds, and its covariance as
a method. Kernel hyperparameters are in scaled units (inputs scaled onto [0, 1], outputs
standardized); `noise_std`, the known observation noise, is in measured units.

`gp.fit(X, Y, rng)` scales the training data, fits every `Param` by maximizing the log
marginal likelihood and returns a `FittedGP`: the prior GP with every `Param` replaced by
its value, held with the scaled data and the scalers. `fitted.predict(X)` gives its
noise-free mean and standard deviation in measured units, at numeric or symbolic points.
"""

import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.problem import Symbolic, System
from hybrid_bo.solvers import NLP, Ipopt, solve_with_multistart
from hybrid_bo.utils import Scaler, lhs, minmax, scale, standard, unscale

_logger = logging.getLogger(__name__)

# Above this condition number of the fitted training covariance, `GP.fit` warns
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


@dataclass(frozen=True)
class Stationary(ABC):
    """A kernel that depends only on the distance between points, each input divided by
    its lengthscale. Each kernel below adds its formula as `covariance`.

    With `ard`, a fitted lengthscale has one entry per input; without, one shared by every
    input. Raises `ValueError` for a lengthscale per input without `ard`.
    """

    lengthscale: Lengthscale = Param((1e-3, 1e2))
    ard: bool = True

    def __post_init__(self) -> None:
        """Raises `ValueError` for a lengthscale per input without `ard`."""
        if isinstance(self.lengthscale, tuple) and not self.ard:
            msg = f"A lengthscale per input needs ard=True, got {self!r}"
            raise ValueError(msg)

    @abstractmethod
    def covariance(self, X1: Any, X2: Any) -> Any:
        """The covariance between the rows of the CasADi matrices X1 (n1, d) and X2
        (n2, d), shape (n1, n2). The lengthscale must be numbers or CasADi scalars, not a
        `Param`.

        Raises `ValueError` as `squared_distance` does.
        """

    def squared_distance(self, X1: Any, X2: Any) -> Any:
        """Squared distances between the rows of X1 and X2, each input divided by its
        lengthscale, shape (n1, n2).

        Clamped below at 1e-30, as gpytorch does, so the square root in the Matern kernels
        has a finite derivative at zero distance.

        Raises `ValueError` if X1 and X2 have different columns, or a lengthscale per
        input doesn't have one entry per column.
        """
        # One lengthscale per input, as a row repeated for every point
        n_inputs = X1.shape[1]
        if X2.shape[1] != n_inputs:
            msg = f"X1 has {n_inputs} inputs and X2 has {X2.shape[1]}"
            raise ValueError(msg)
        lengthscale = self.lengthscale
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

    def params(self, n_inputs: int) -> list[Param]:
        """One `Param` per search entry: a lengthscale with ARD once per input."""
        if not isinstance(self.lengthscale, Param):
            return []
        return [self.lengthscale] * (n_inputs if self.ard else 1)

    def with_values(self, values: list[Any], n_inputs: int) -> "Stationary":
        """The kernel with its `Param` replaced: `values` has a number or a CasADi scalar
        per entry of `params`, and a lengthscale with ARD becomes a tuple of them."""
        if not isinstance(self.lengthscale, Param):
            return self
        value = tuple(values[:n_inputs]) if self.ard else values[0]
        return dataclasses.replace(self, lengthscale=value)


class Matern12(Stationary):
    """The Matern-1/2 (absolute exponential) kernel."""

    def covariance(self, X1: Any, X2: Any) -> Any:
        """exp(-r)."""
        r2 = self.squared_distance(X1, X2)
        return cas.exp(-cas.sqrt(r2))


class Matern32(Stationary):
    """The Matern-3/2 kernel."""

    def covariance(self, X1: Any, X2: Any) -> Any:
        """(1 + sqrt(3) r) exp(-sqrt(3) r)."""
        r2 = self.squared_distance(X1, X2)
        return (1.0 + np.sqrt(3.0) * cas.sqrt(r2)) * cas.exp(-cas.sqrt(3.0 * r2))


class Matern52(Stationary):
    """The Matern-5/2 kernel."""

    def covariance(self, X1: Any, X2: Any) -> Any:
        """(1 + sqrt(5) r + 5/3 r^2) exp(-sqrt(5) r)."""
        r2 = self.squared_distance(X1, X2)
        return (1.0 + np.sqrt(5.0) * cas.sqrt(r2) + 5.0 / 3.0 * r2) * cas.exp(
            -cas.sqrt(5.0 * r2)
        )


class SquaredExponential(Stationary):
    """The squared exponential (RBF, Gaussian) kernel."""

    def covariance(self, X1: Any, X2: Any) -> Any:
        """exp(-r^2 / 2)."""
        r2 = self.squared_distance(X1, X2)
        return cas.exp(-0.5 * r2)


@dataclass(frozen=True)
class Scale:
    """A kernel times an output scale, unsquared."""

    kernel: "Kernel"
    outputscale: float | Param = Param((0.1, 10.0))

    def covariance(self, X1: Any, X2: Any) -> Any:
        """The output scale times the kernel's covariance."""
        return self.outputscale * self.kernel.covariance(X1, X2)

    def params(self, n_inputs: int) -> list[Param]:
        """One `Param` per search entry: the output scale before the kernel it scales."""
        own = [self.outputscale] if isinstance(self.outputscale, Param) else []
        return own + self.kernel.params(n_inputs)

    def with_values(self, values: list[Any], n_inputs: int) -> "Scale":
        """The kernel with each `Param` replaced: `values` has a number or a CasADi scalar
        per entry of `params`, in that order."""
        if isinstance(self.outputscale, Param):
            kernel = self.kernel.with_values(values[1:], n_inputs)
            return Scale(kernel, values[0])
        return Scale(self.kernel.with_values(values, n_inputs), self.outputscale)


@dataclass(frozen=True)
class Zero:
    """The zero prior mean."""

    def at(self, X: Any) -> Any:
        """The prior mean at the rows of the CasADi matrix X (n, d), a (n, 1) column."""
        return cas.DM.zeros(X.shape[0], 1)


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

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rng: np.random.Generator,
        n_starts: int = 10,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
    ) -> "FittedGP":
        """This GP fitted to the inputs X (n, d) and outputs Y (n,). The GP itself doesn't
        change.

        Scales the inputs onto [0, 1] (`minmax`) and the outputs to zero mean and unit
        variance (`standard`), then maximizes the log marginal likelihood over every
        `Param` (log ones in log space) with `solve_with_multistart`, from `n_starts`
        Latin hypercube starts over their bounds; a lengthscale with ARD has one entry per
        column of X. A GP without a `Param` is only conditioned on the data, with no
        draws. It warns if the training covariance at the kept values has a condition
        number above `CONDITION_THRESHOLD`.

        Raises `ValueError` if X isn't (n, d), Y isn't (n,), a value isn't finite, or a
        fixed lengthscale per input doesn't have one entry per column of X. Raises
        `NoSolutionError` if every start fails.
        """
        # The checks
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if X.ndim != 2 or Y.shape != (len(X),):
            msg = (
                f"A fit needs X of shape (n, d) and Y of shape (n,), got {X.shape} and "
                f"{Y.shape}"
            )
            raise ValueError(msg)
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            msg = "A fit needs finite data, but X or Y holds NaN or infinity"
            raise ValueError(msg)

        # Scale the data, and list what there is to fit
        x_scaler, y_scaler = minmax(X), standard(Y[:, np.newaxis])
        X_scaled = scale(x_scaler, X)
        Y_scaled = scale(y_scaler, Y[:, np.newaxis]).ravel()
        n_inputs = X.shape[1]
        params = self.kernel.params(n_inputs)
        fitted_gp = self
        if params:
            # The search space: one entry per entry of `params`, log ones as logarithms
            is_log = np.array([param.log for param in params])
            lower = np.array([param.bounds[0] for param in params])
            upper = np.array([param.bounds[1] for param in params])

            def to_search_space(values: np.ndarray) -> np.ndarray:
                """Hyperparameter entries as search-space entries: logarithms for log
                ones."""
                return np.where(is_log, np.log(np.where(is_log, values, 1.0)), values)

            # The negative log marginal likelihood over the search space: a fitted GP
            # whose hyperparameters are symbols
            w = Symbolic.sym("w", len(params))
            symbols = [cas.exp(w[i]) if is_log[i] else w[i] for i in range(len(params))]
            kernel = self.kernel.with_values(symbols, n_inputs)
            symbolic_gp = dataclasses.replace(self, kernel=kernel)
            symbolic_fit = FittedGP(symbolic_gp, X_scaled, Y_scaled, x_scaler, y_scaler)
            nlp = NLP(
                cas.Function(
                    "negative_log_marginal_likelihood",
                    [w],
                    [-symbolic_fit.log_marginal_likelihood()],
                ),
                to_search_space(lower),
                to_search_space(upper),
            )

            # Multistart from Latin hypercube starts over the bounds
            starts = lhs(nlp.lower, nlp.upper, n_starts, rng)
            w_best, _ = solve_with_multistart(nlp, starts, solver)

            # Back from the search space, to the numbers the fitted GP holds
            values = np.where(is_log, np.exp(w_best), w_best).tolist()
            kernel = self.kernel.with_values(values, n_inputs)
            fitted_gp = dataclasses.replace(self, kernel=kernel)

        # Warn if the training covariance K is nearly singular at those values (a
        # non-finite one counts as singular; the SVD behind `cond` would fail on it)
        X_train = Symbolic(X_scaled)
        noise_variance = (self.noise_std * y_scaler.slope[0]) ** 2
        K = fitted_gp.kernel.covariance(X_train, X_train) + (
            noise_variance + self.jitter
        ) * Symbolic.eye(len(X))
        K = cas.evalf(K).full()
        condition_number = np.linalg.cond(K) if np.isfinite(K).all() else np.inf
        if condition_number > CONDITION_THRESHOLD:
            _logger.warning(
                "The fitted GP's training covariance is nearly singular: condition "
                "number %.3g with jitter %g",
                condition_number,
                self.jitter,
            )
        _logger.info("Fitted %r to %d points", fitted_gp.kernel, len(X))
        return FittedGP(fitted_gp, X_scaled, Y_scaled, x_scaler, y_scaler)


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
        if self.gp.kernel.params(1):
            msg = f"A fitted GP has no Param left, got {self.gp.kernel!r}"
            raise ValueError(msg)

    def log_marginal_likelihood(self) -> Any:
        """The log marginal likelihood of the scaled training data, the quantity `GP.fit`
        maximizes. A number, or a CasADi expression while the hyperparameters are the
        symbols `GP.fit` searches over."""
        gp = self.gp
        X, Y = Symbolic(self.X), Symbolic(self.Y)

        # K = k(X, X) + (noise variance + jitter) I, with the noise in scaled units
        noise_variance = (gp.noise_std * self.y_scaler.slope[0]) ** 2
        K = gp.kernel.covariance(X, X) + (noise_variance + gp.jitter) * Symbolic.eye(
            X.shape[0]
        )

        # log p(Y) = -1/2 (r' K^-1 r + log det K + n log 2 pi), with r = Y - m(X) and
        # K = L L'. CasADi's `chol` gives the upper factor
        residual = Y - gp.mean.at(X)
        L = cas.chol(K).T
        alpha = cas.solve(L.T, cas.solve(L, residual))
        value = -0.5 * (
            residual.T @ alpha
            + 2 * cas.sum1(cas.log(cas.diag(L)))
            + len(self.Y) * np.log(2 * np.pi)
        )
        return value if cas.symvar(value) else float(cas.evalf(value))

    def predict(self, X: Any) -> tuple[Any, Any]:
        """The conditional mean and standard deviation at the rows of X (n, d), in
        measured units. They are noise-free: the standard deviation is of the latent
        output, with no observation noise added.

        For a numpy X, returns two arrays (n,). For a symbolic X (SX or MX; the conditional
        mean and variance themselves are built in SX), returns two (n, 1) CasADi columns,
        for building acquisition problems.

        Raises `ValueError` unless X is a finite 2-D numpy array or symbolic, with the
        columns of the training inputs.
        """
        # The checks
        is_numeric = not isinstance(X, cas.SX | cas.MX)
        if is_numeric and not (isinstance(X, np.ndarray) and X.ndim == 2):
            msg = f"predict needs a numpy X of shape (n_points, d), got {X!r}"
            raise ValueError(msg)
        if X.shape[1] != self.X.shape[1]:
            msg = f"X has {X.shape[1]} inputs and the GP has {self.X.shape[1]}"
            raise ValueError(msg)
        if is_numeric and not np.isfinite(X).all():
            msg = "predict needs finite points, but X holds NaN or infinity"
            raise ValueError(msg)

        # X in scaled units, as symbols, with the hyperparameters and data as constants.
        # A numpy X is predicted one row at a time, mapped over its rows: predicting all
        # rows at once builds their whole covariance matrix
        gp = self.gp
        X_train, Y_train = Symbolic(self.X), Symbolic(self.Y)
        X_symbols = Symbolic.sym("X", 1 if is_numeric else X.shape[0], X.shape[1])
        X_scaled = scale(self.x_scaler, X_symbols)

        # K = k(X, X) + (noise variance + jitter) I = L L', with the noise in scaled
        # units. CasADi's `chol` gives the upper factor
        noise_variance = (gp.noise_std * self.y_scaler.slope[0]) ** 2
        K = gp.kernel.covariance(X_train, X_train) + (
            noise_variance + gp.jitter
        ) * Symbolic.eye(X_train.shape[0])
        L = cas.chol(K).T

        # mean = m(x) + k(x, X) K^-1 (Y - m(X)) and
        # variance = k(x, x) - k(x, X) K^-1 k(X, x), through L
        alpha = cas.solve(L, Y_train - gp.mean.at(X_train))
        beta = cas.solve(L, gp.kernel.covariance(X_scaled, X_train).T)
        mean = gp.mean.at(X_scaled) + beta.T @ alpha
        variance = (
            cas.diag(gp.kernel.covariance(X_scaled, X_scaled)) - cas.sum1(beta**2).T
        )

        # Back to measured units, evaluated at X
        function = cas.Function(
            "predict",
            [X_symbols],
            [
                unscale(self.y_scaler, mean),
                cas.sqrt(variance) / self.y_scaler.slope[0],
            ],
        )
        if is_numeric:
            mean, std = function.map(len(X))(X.reshape(1, -1))
            return mean.full().ravel(), std.full().ravel()
        return function(X)


def fit_surrogate(
    system: System,
    dataset: Dataset,
    *,
    gp: GP,
    rng: np.random.Generator,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> tuple[FittedGP, ...]:
    """The surrogate model of the system: the prior GP `gp` fitted to each unknown state
    of `dataset` over the arguments of h, one fitted GP per unknown state, in the order
    of `system.y`.

    Each GP is fitted by `GP.fit` from `n_starts` starts.
    """
    X = np.column_stack(system.h_arguments(dataset.u, dataset.x))
    fitted = []
    for var, y in zip(system.y, dataset.y.T, strict=True):
        _logger.info("Fitting the GP of %s", var.name)
        fitted.append(gp.fit(X, y, rng, n_starts, solver))
    return tuple(fitted)
