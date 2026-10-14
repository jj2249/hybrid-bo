"""Kernels: plain functions of two sets of points, combined with arithmetic.

Each kernel takes CasADi matrices X1 (n1, d) and X2 (n2, d) and returns (n1, n2). A
lengthscale or scale is either a scalar, shared by every input, or a (d, 1) column with
one entry per input (ARD), as a `Hyper` with `size=d` gives it. Kernels use only CasADi
operations, so they work for SX, MX and DM alike:

    kernel=lambda h, X1, X2: h.outputscale * kernels.matern52(X1, X2, h.lengthscale)
"""

from typing import Any

import casadi as cas
import numpy as np


def _per_column(values: Any, n_rows: int) -> Any:
    """`values` as a (n_rows, d) matrix with one row per point, for elementwise use
    with X (n_rows, d). A scalar is returned unchanged."""
    if isinstance(values, int | float) or values.numel() == 1:
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


def matern12(X1: Any, X2: Any, lengthscale: Any) -> Any:
    """The Matern-1/2 (absolute exponential) kernel."""
    return cas.exp(-cas.sqrt(_squared_distance(X1, X2, lengthscale)))


def matern32(X1: Any, X2: Any, lengthscale: Any) -> Any:
    """The Matern-3/2 kernel."""
    r2 = _squared_distance(X1, X2, lengthscale)
    return (1.0 + np.sqrt(3.0) * cas.sqrt(r2)) * cas.exp(-cas.sqrt(3.0 * r2))


def matern52(X1: Any, X2: Any, lengthscale: Any) -> Any:
    """The Matern-5/2 kernel."""
    r2 = _squared_distance(X1, X2, lengthscale)
    return (1.0 + np.sqrt(5.0) * cas.sqrt(r2) + 5.0 / 3.0 * r2) * cas.exp(
        -cas.sqrt(5.0 * r2)
    )


def squared_exponential(X1: Any, X2: Any, lengthscale: Any) -> Any:
    """The squared exponential (RBF, Gaussian) kernel."""
    return cas.exp(-0.5 * _squared_distance(X1, X2, lengthscale))


def linear(X1: Any, X2: Any, scale: Any = 1.0) -> Any:
    """The linear kernel X1 diag(scale) X2^T."""
    return (X1 * _per_column(scale, X1.shape[0])) @ X2.T


def constant(X1: Any, X2: Any) -> Any:
    """Ones, shape (n1, n2); multiply by a hyperparameter for a constant kernel."""
    return cas.DM.ones(X1.shape[0], X2.shape[0])
