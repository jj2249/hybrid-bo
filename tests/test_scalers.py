"""Scalers: affine maps of the GP's inputs and outputs."""

import casadi as cas
import numpy as np
import pytest

from hybrid_bo.utils import Scaler

X = np.random.default_rng(0).normal(3.0, 2.0, size=(10, 2))


def test_standard():
    """The standard scaler gives each column mean 0 and std 1, and unscale undoes it."""
    scaler = Scaler.standard(X)
    scaled = scaler.scale(X)
    np.testing.assert_allclose(scaled.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(0), 1)
    np.testing.assert_allclose(scaler.unscale(scaled), X)


def test_minmax():
    """The minmax scaler maps each column onto [0, 1], and unscale undoes it."""
    scaler = Scaler.minmax(X)
    scaled = scaler.scale(X)
    np.testing.assert_allclose(scaled.min(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.max(0), 1.0, atol=1e-12)
    np.testing.assert_allclose(scaler.unscale(scaled), X)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize("fit", [Scaler.standard, Scaler.minmax])
def test_constant_column_is_shifted_not_divided(fit):
    """A column with zero spread scales to a finite constant instead of dividing by
    zero."""
    values = np.column_stack([X[:, 0], np.full(10, 4.0)])
    scaled = fit(values).scale(values)
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled[:, 1], scaled[0, 1])


def test_symbolic_matches_numeric():
    """Scaling a CasADi symbol gives the same values as scaling the numbers."""
    scaler = Scaler.minmax(X)
    symbol = cas.SX.sym("X", *X.shape)
    function = cas.Function("scale", [symbol], [scaler.scale(symbol)])
    np.testing.assert_allclose(function(X).full(), scaler.scale(X))
