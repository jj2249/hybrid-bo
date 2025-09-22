from typing import overload, override

import casadi as cas
import numpy as np

from .type_aliases import CasadiType


class AffineTransformer:
    # y_transformed = slope * y + intercept

    def __init__(
        self,
        slope: np.ndarray | None = None,
        intercept: np.ndarray | None = None,
    ) -> None:
        self.n_inputs: int
        self.slope: np.ndarray
        self.intercept: np.ndarray

        if (slope is not None) and (intercept is not None):
            self.set_slope_and_intercept(slope, intercept)
        else:
            self.n_inputs: int = 0
            self.slope: np.ndarray = np.empty([0])
            self.intercept: np.ndarray = np.empty([0])

    def set_slope_and_intercept(self, slope: np.ndarray, intercept: np.ndarray) -> None:
        if len(slope.shape) != 2:
            msg: str = "len(slope.shape) != 2"
            raise Exception(msg)
        if len(intercept.shape) != 2:
            msg: str = "len(intercept.shape) != 2"
            raise Exception(msg)

        if slope.shape[0] != 1:
            msg: str = "slope.shape[0] != 1"
            raise Exception(msg)
        if intercept.shape[0] != 1:
            msg: str = "intercept.shape[0] != 1"
            raise Exception(msg)
        if slope.shape[1] != intercept.shape[1]:
            msg: str = "slope.shape[1] != intercept.shape[1]"
            raise Exception(msg)

        self.slope = slope
        self.intercept = intercept
        self.n_inputs = slope.shape[1]

    def fit(self, X: np.ndarray) -> None: ...

    @overload
    def transform(self, X: cas.SX) -> cas.SX: ...
    @overload
    def transform(self, X: cas.MX) -> cas.MX: ...
    @overload
    def transform(self, X: cas.DM) -> cas.DM: ...
    @overload
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def transform(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
        if len(X.shape) != 2:
            msg: str = "len(X.shape) != 2"
            raise Exception(msg)

        if X.shape[1] != self.n_inputs:
            msg: str = "X.shape[1] != self.n_inputs"
            raise Exception(msg)

        slope_tiled: np.ndarray = np.tile(self.slope, (X.shape[0], 1))
        intercept_tiled: np.ndarray = np.tile(self.intercept, (X.shape[0], 1))

        return slope_tiled * X + intercept_tiled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @overload
    def inverse_transform(self, X_transformed: cas.SX) -> cas.SX: ...
    @overload
    def inverse_transform(self, X_transformed: cas.MX) -> cas.MX: ...
    @overload
    def inverse_transform(self, X_transformed: cas.DM) -> cas.DM: ...
    @overload
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray: ...

    def inverse_transform(
        self,
        X_transformed: CasadiType | np.ndarray,
    ) -> CasadiType | np.ndarray:
        if len(X_transformed.shape) != 2:
            msg: str = "len(X_transformed.shape) != 2"
            raise Exception(msg)

        if X_transformed.shape[1] != self.n_inputs:
            msg: str = "X_transformed.shape[1] != self.n_inputs"
            raise Exception(msg)

        slope_tiled: np.ndarray = np.tile(self.slope, (X_transformed.shape[0], 1))
        intercept_tiled = np.tile(self.intercept, (X_transformed.shape[0], 1))

        return (X_transformed - intercept_tiled) / slope_tiled


class MinMaxTransformer(AffineTransformer):
    # y_transformed = (max - min) / (max_transformed - min_transformed) * (y - min) + min_transformed

    def __init__(
        self,
        min_transformed: float = 0.0,
        max_transformed: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_transformed: float = min_transformed
        self.max_transformed: float = max_transformed
        self.min: np.ndarray = np.empty([0])
        self.max: np.ndarray = np.empty([0])

    @override
    def fit(self, X: np.ndarray) -> None:
        if len(X.shape) != 2:
            msg: str = "len(X.shape) != 2"
            raise Exception(msg)

        self.n_inputs = X.shape[1]

        self.min = np.min(X, 0)[np.newaxis, :]
        self.max = np.max(X, 0)[np.newaxis, :]

        self.slope = (self.max_transformed - self.min_transformed) / (
            self.max - self.min
        )
        self.intercept = -self.min / (self.max - self.min) + self.min_transformed

    @override
    def set_slope_and_intercept(self, slope: np.ndarray, intercept: np.ndarray) -> None:
        msg: str = "This method is not allowed for this class!"
        raise Exception(msg)


class StandardTransformer(AffineTransformer):
    # y_transformed = (y - mean) / std

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        super().__init__()
        self.mean: np.ndarray = np.empty([0])
        self.std: np.ndarray = np.empty([0])
        self.with_mean: bool = with_mean
        self.with_std: bool = with_std

    @override
    def fit(self, X: np.ndarray, ddof: int = 0) -> None:
        # Parameter ddof: see parameter in numpy.std()

        if len(X.shape) != 2:
            msg: str = "len(X.shape) != 2"
            raise Exception(msg)

        self.n_inputs = X.shape[1]

        if self.with_mean:
            self.mean = np.mean(X, 0)[np.newaxis, :]
        else:
            self.mean = np.zeros((1, self.n_inputs))

        if self.with_std:
            self.std = np.std(X, 0, ddof=ddof)[np.newaxis, :]
        else:
            self.std = np.ones((1, self.n_inputs))

        self.slope = 1 / self.std
        self.intercept = -self.mean / self.std

    @override
    def set_slope_and_intercept(self, slope: np.ndarray, intercept: np.ndarray) -> None:
        msg: str = "This method is not allowed for this class!"
        raise Exception(msg)
