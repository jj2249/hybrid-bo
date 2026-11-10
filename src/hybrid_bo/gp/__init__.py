"""Gaussian processes: a prior `GP`, fitted to data as a `FittedGP` that predicts.

Kernel hyperparameters are in scaled units (inputs scaled onto [0, 1], outputs
standardized); `noise_std` is in measured units.
"""

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.gp.kernels import Matern52
from hybrid_bo.gp.means import Zero
from hybrid_bo.solvers import NLP, Ipopt
from hybrid_bo.utils import Scaler, Symbolic
from hybrid_bo.variables import Var

_logger = logging.getLogger(__name__)

# Above this condition number of the fitted training covariance, `GP.fit` warns
CONDITION_THRESHOLD = 1e10


@dataclass(frozen=True, kw_only=True)
class GP:
    """A prior GP: its kernel, its mean, the known observation noise std `noise_std`
    (0 is noise-free) and `jitter` added to the scaled training covariance.

    Raises `ValueError` if `noise_std` is negative.
    """

    kernel: Matern52
    mean: Zero
    noise_std: float
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        """Raises `ValueError` if `noise_std` is negative."""
        if not self.noise_std >= 0:
            msg = f"noise_std must be at least 0, got {self.noise_std}"
            raise ValueError(msg)

    def fit(
        self,
        data: Dataset,
        *,
        inputs: Sequence[Var],
        output: Var,
        rng: np.random.Generator,
        n_starts: int = 10,
        solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
    ) -> "FittedGP":
        """This GP of `output` over `inputs`, fitted to `data` by maximizing the log
        marginal likelihood from `n_starts` starts. Warns if the training covariance is
        nearly singular.

        Raises `ValueError` if the data aren't finite, and `NoSolutionError` if every
        start fails.
        """
        inputs = tuple(inputs)
        X = np.column_stack([data[var] for var in inputs])
        Y = data[output]
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            msg = "A fit needs finite data, but the data hold NaN or infinity"
            raise ValueError(msg)

        # Scale the inputs onto [0, 1] and the outputs to zero mean and unit variance
        x_scaler, y_scaler = Scaler.minmax(X), Scaler.standard(Y[:, np.newaxis])
        X_scaled = x_scaler.scale(X)
        Y_scaled = y_scaler.scale(Y[:, np.newaxis]).ravel()
        n_inputs = X.shape[1]
        bounds = self.kernel.learned_bounds(n_inputs)
        fitted_gp = self
        if bounds:
            # Maximize the log marginal likelihood over the log hyperparameters
            w = Symbolic.sym("w", len(bounds))
            symbolic_gp = dataclasses.replace(
                self,
                kernel=self.kernel.with_learned(
                    [cas.exp(w[i]) for i in range(len(bounds))], n_inputs
                ),
            )
            symbolic_fit = FittedGP(
                symbolic_gp, inputs, output, X_scaled, Y_scaled, x_scaler, y_scaler
            )
            log_bounds = np.log(np.array(bounds))
            nlp = NLP(
                cas.Function(
                    "negative_log_marginal_likelihood",
                    [w],
                    [-symbolic_fit.log_marginal_likelihood()],
                ),
                log_bounds[:, 0],
                log_bounds[:, 1],
            )
            w_best, _ = nlp.solve(rng, n_starts=n_starts, solver=solver)
            kernel = self.kernel.with_learned(np.exp(w_best).tolist(), n_inputs)
            fitted_gp = dataclasses.replace(self, kernel=kernel)

        # Warn if the training covariance is nearly singular (a non-finite one counts)
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
        _logger.info(
            "Fitted %r to %d points of %s", fitted_gp.kernel, len(X), output.name
        )
        return FittedGP(
            fitted_gp, inputs, output, X_scaled, Y_scaled, x_scaler, y_scaler
        )


@dataclass(frozen=True)
class FittedGP:
    """A GP with learned hyperparameters, the `inputs` and `output` it models, its
    scaled training data and the scalers."""

    gp: GP
    inputs: tuple[Var, ...]
    output: Var
    X: np.ndarray  # (n, d), scaled
    Y: np.ndarray  # (n,), scaled
    x_scaler: Scaler
    y_scaler: Scaler

    def log_marginal_likelihood(self) -> Any:
        """The log marginal likelihood of the scaled training data: a number, or a
        CasADi expression of symbolic hyperparameters."""
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
        """The noise-free mean and std at the rows of X (n, d), in measured units: two
        arrays (n,) for numpy X, two CasADi columns (n, 1) for symbolic X.

        Raises `ValueError` unless X is a finite 2-D numpy array or symbolic, with one
        column per input.
        """
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

        # A numpy X is predicted one row at a time, mapped over its rows
        gp = self.gp
        X_train, Y_train = Symbolic(self.X), Symbolic(self.Y)
        X_symbols = Symbolic.sym("X", 1 if is_numeric else X.shape[0], X.shape[1])
        X_scaled = self.x_scaler.scale(X_symbols)

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
                self.y_scaler.unscale(mean),
                cas.sqrt(variance) / self.y_scaler.slope[0],
            ],
        )
        if is_numeric:
            mean, std = function.map(len(X))(X.reshape(1, -1))
            return mean.full().ravel(), std.full().ravel()
        return function(X)
