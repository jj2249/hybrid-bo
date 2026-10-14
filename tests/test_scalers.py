"""Scalers: affine maps of the GP's inputs and outputs."""

import casadi as cas
import numpy as np
import pytest

from hybrid_bo.gp import minmax, scale, standard, unscale

X = np.random.default_rng(0).normal(3.0, 2.0, size=(10, 2))


def test_standard():
    scaler = standard(X)
    scaled = scale(scaler, X)
    np.testing.assert_allclose(scaled.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(0), 1)
    np.testing.assert_allclose(unscale(scaler, scaled), X)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-1.0, 1.0)])
def test_minmax(bounds):
    scaler = minmax(X, *bounds)
    scaled = scale(scaler, X)
    np.testing.assert_allclose(scaled.min(0), bounds[0], atol=1e-12)
    np.testing.assert_allclose(scaled.max(0), bounds[1], atol=1e-12)
    np.testing.assert_allclose(unscale(scaler, scaled), X)


@pytest.mark.parametrize("fit", [standard, minmax])
def test_constant_column_is_shifted_not_divided(fit):
    values = np.column_stack([X[:, 0], np.full(10, 4.0)])
    scaled = scale(fit(values), values)
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled[:, 1], scaled[0, 1])


def test_symbolic_matches_numeric():
    scaler = minmax(X, -1.0, 1.0)
    symbol = cas.SX.sym("X", *X.shape)
    function = cas.Function("scale", [symbol], [scale(scaler, symbol)])
    np.testing.assert_allclose(function(X).full(), scale(scaler, X))
