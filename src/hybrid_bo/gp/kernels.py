"""Kernels: frozen dataclasses that hold their hyperparameters and their covariance."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np


@dataclass(frozen=True)
class Matern52:
    """The Matern-5/2 kernel times an output scale, in scaled units.

    A hyperparameter that is None is learned in log space within its bounds; a number
    fixes it. The output scale is fixed at 1 by default, the variance of the
    standardized outputs. With `ard`, a learned lengthscale has one entry per input.
    With `lengthscale_prior`, a learned lengthscale is MAP rather than maximum
    likelihood.
    """

    lengthscale: float | tuple[float, ...] | None = None
    lengthscale_bounds: tuple[float, float] = (1e-3, 1e2)
    outputscale: float | None = 1.0
    outputscale_bounds: tuple[float, float] = (0.1, 10.0)
    ard: bool = True
    lengthscale_prior: bool = False

    def learned_bounds(self, n_inputs: int) -> list[tuple[float, float]]:
        """The bounds of each learned entry: the output scale, then the lengthscales."""
        bounds = [self.outputscale_bounds] if self.outputscale is None else []
        if self.lengthscale is None:
            bounds += [self.lengthscale_bounds] * (n_inputs if self.ard else 1)
        return bounds

    def with_learned(self, values: list[Any], n_inputs: int) -> "Matern52":
        """This kernel with a number or CasADi scalar per entry of `learned_bounds`."""
        values = list(values)
        outputscale = values.pop(0) if self.outputscale is None else self.outputscale
        lengthscale = self.lengthscale
        if lengthscale is None:
            lengthscale = tuple(values) if self.ard else values[0]
        return dataclasses.replace(
            self, outputscale=outputscale, lengthscale=lengthscale
        )

    def log_prior(self, w: Any, n_inputs: int) -> Any:
        """The log prior, up to a constant, of the log hyperparameters w, a CasADi
        column ordered as `learned_bounds`: with `lengthscale_prior`, each learned log
        lengthscale ~ N(sqrt(2) + log(n_inputs) / 2, 3) (Hvarfner et al., 2024), and
        flat otherwise."""
        if not self.lengthscale_prior or self.lengthscale is not None:
            return 0.0
        log_lengthscale = w[1:] if self.outputscale is None else w
        mu = np.sqrt(2.0) + 0.5 * np.log(n_inputs)
        return -cas.sumsqr(log_lengthscale - mu) / (2 * 3.0)

    def covariance(self, X1: Any, X2: Any) -> Any:
        """outputscale (1 + sqrt(5) r + 5/3 r^2) exp(-sqrt(5) r) between the rows of the
        CasADi matrices X1 (n1, d) and X2 (n2, d), (n1, n2).

        Raises `ValueError` if a hyperparameter is unlearned, or the inputs don't match.
        """
        if self.lengthscale is None or self.outputscale is None:
            msg = f"The kernel has a hyperparameter left to learn: {self!r}"
            raise ValueError(msg)
        n_inputs = X1.shape[1]
        lengthscales = (
            self.lengthscale
            if isinstance(self.lengthscale, tuple)
            else (self.lengthscale,) * n_inputs
        )
        if X2.shape[1] != n_inputs or len(lengthscales) != n_inputs:
            msg = (
                f"X1 has {n_inputs} inputs, X2 {X2.shape[1]} and the lengthscale "
                f"{len(lengthscales)}"
            )
            raise ValueError(msg)

        # Squared distances, each input divided by its lengthscale, clamped at 1e-30 so
        # sqrt has a finite derivative at zero (as gpytorch does)
        row = cas.horzcat(*lengthscales)
        Z1 = X1 / cas.repmat(row, X1.shape[0], 1)
        Z2 = X2 / cas.repmat(row, X2.shape[0], 1)
        squared_norms = cas.repmat(cas.sum2(Z1**2), 1, Z2.shape[0]) + cas.repmat(
            cas.sum2(Z2**2).T, Z1.shape[0], 1
        )
        r2 = cas.fmax(-2.0 * (Z1 @ Z2.T) + squared_norms, 1e-30)
        matern = (1.0 + np.sqrt(5.0) * cas.sqrt(r2) + 5.0 / 3.0 * r2) * cas.exp(
            -cas.sqrt(5.0 * r2)
        )
        return self.outputscale * matern
