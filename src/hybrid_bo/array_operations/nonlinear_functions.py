from typing import overload

import casadi as cas
import numpy as np

from ..type_aliases import CasadiType, NumericType


@overload
def log(X: cas.SX) -> cas.SX: ...
@overload
def log(X: cas.MX) -> cas.MX: ...
@overload
def log(X: cas.DM) -> cas.DM: ...
@overload
def log(X: np.ndarray) -> np.ndarray: ...
@overload
def log(X: float | int) -> float: ...


def log(X: CasadiType | NumericType) -> CasadiType | NumericType:
    if isinstance(X, CasadiType):
        return cas.log(X)
    return np.log(X)


@overload
def exp(X: cas.SX) -> cas.SX: ...
@overload
def exp(X: cas.MX) -> cas.MX: ...
@overload
def exp(X: cas.DM) -> cas.DM: ...
@overload
def exp(X: np.ndarray) -> np.ndarray: ...
@overload
def exp(X: float | int) -> float: ...


def exp(X: CasadiType | NumericType) -> CasadiType | NumericType:
    if isinstance(X, CasadiType):
        return cas.exp(X)
    return np.exp(X)


@overload
def sqrt(X: cas.SX) -> cas.SX: ...
@overload
def sqrt(X: cas.MX) -> cas.MX: ...
@overload
def sqrt(X: cas.DM) -> cas.DM: ...
@overload
def sqrt(X: np.ndarray) -> np.ndarray: ...
@overload
def sqrt(X: float | int) -> float: ...


def sqrt(X: CasadiType | NumericType) -> CasadiType | NumericType:
    if isinstance(X, CasadiType):
        return cas.sqrt(X)
    return np.sqrt(X)
