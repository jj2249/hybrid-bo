from typing import overload

import casadi as cas
import numpy as np

from hybrid_bo.type_aliases import CasadiType, NumericType


# Maybe always use CasADi function?
@overload
def fmin(X1: cas.SX, X2: cas.SX) -> cas.SX: ...
@overload
def fmin(X1: cas.SX, X2: NumericType) -> cas.SX: ...
@overload
def fmin(X1: NumericType, X2: cas.SX) -> cas.SX: ...
@overload
def fmin(X1: cas.MX, X2: cas.MX) -> cas.MX: ...
@overload
def fmin(X1: cas.MX, X2: NumericType) -> cas.MX: ...
@overload
def fmin(X1: NumericType, X2: cas.MX) -> cas.MX: ...
@overload
def fmin(X1: cas.DM, X2: cas.DM) -> cas.DM: ...
@overload
def fmin(X1: cas.DM, X2: NumericType) -> cas.DM: ...
@overload
def fmin(X1: NumericType, X2: cas.DM) -> cas.DM: ...
@overload
def fmin(X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...


def fmin(
    X1: CasadiType | NumericType, X2: CasadiType | NumericType
) -> CasadiType | np.ndarray:
    if isinstance(X1, CasadiType) or isinstance(X2, CasadiType):
        return cas.fmin(X1, X2)
    return np.fmin(X1, X2)


# Maybe always use CasADi function?
@overload
def fmax(X1: cas.SX, X2: cas.SX) -> cas.SX: ...
@overload
def fmax(X1: cas.SX, X2: NumericType) -> cas.SX: ...
@overload
def fmax(X1: NumericType, X2: cas.SX) -> cas.SX: ...
@overload
def fmax(X1: cas.MX, X2: cas.MX) -> cas.MX: ...
@overload
def fmax(X1: cas.MX, X2: NumericType) -> cas.MX: ...
@overload
def fmax(X1: NumericType, X2: cas.MX) -> cas.MX: ...
@overload
def fmax(X1: cas.DM, X2: cas.DM) -> cas.DM: ...
@overload
def fmax(X1: cas.DM, X2: NumericType) -> cas.DM: ...
@overload
def fmax(X1: NumericType, X2: cas.DM) -> cas.DM: ...
@overload
def fmax(X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...


def fmax(
    X1: CasadiType | NumericType, X2: CasadiType | NumericType
) -> CasadiType | NumericType:
    if isinstance(X1, CasadiType) or isinstance(X2, CasadiType):
        return cas.fmax(X1, X2)
    return np.fmax(X1, X2)
