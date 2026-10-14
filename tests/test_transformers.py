"""Affine input/output transformers."""

import numpy as np
import pytest

from hybrid_bo.legacy.affine_transformers import MinMaxTransformer, StandardTransformer

X = np.random.default_rng(0).normal(3.0, 2.0, size=(10, 2))


def test_standard():
    transformer = StandardTransformer()
    Xt = transformer.fit_transform(X)
    np.testing.assert_allclose(Xt.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Xt.std(0), 1)
    np.testing.assert_allclose(transformer.inverse_transform(Xt), X)


@pytest.mark.parametrize(
    "bounds",
    [
        (0.0, 1.0),
        pytest.param(
            (-1.0, 1.0),
            marks=pytest.mark.xfail(
                strict=True, reason="fit() intercept is only right for unit ranges"
            ),
        ),
    ],
)
def test_minmax(bounds):
    transformer = MinMaxTransformer(*bounds)
    Xt = transformer.fit_transform(X)
    np.testing.assert_allclose(Xt.min(0), bounds[0], atol=1e-12)
    np.testing.assert_allclose(Xt.max(0), bounds[1], atol=1e-12)
    np.testing.assert_allclose(transformer.inverse_transform(Xt), X)
