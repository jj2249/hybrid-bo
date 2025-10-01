from typing import overload

import casadi as cas
import numpy as np

from ..type_aliases import CasadiType, SymbolicType


@overload
def sum(X: cas.SX, axis: int | None = None) -> cas.SX: ...
@overload
def sum(X: cas.MX, axis: int | None = None) -> cas.MX: ...
@overload
def sum(X: cas.DM, axis: int | None = None) -> cas.DM: ...
@overload
def sum(X: np.ndarray, axis: int | None = None) -> np.ndarray: ...


def sum(X: CasadiType | np.ndarray, axis: int | None = None) -> CasadiType | np.ndarray:
    if len(X.shape) != 2:
        msg: str = "len(X.shape) != 2."
        raise Exception(msg)

    if (axis is not None) and (axis not in {0, 1}):
        msg: str = "(axis is not None) and (axis not in {0, 1})"
        raise Exception(msg)

    if isinstance(X, CasadiType):
        if axis is None:
            return cas.sum2(cas.sum1(X))
        if axis == 0:
            return cas.sum1(X)
        return cas.sum2(X)

    if axis is None:
        return np.sum(X, axis)
    if axis == 0:
        return np.sum(X, axis)[np.newaxis, :]
    return np.sum(X, axis)[:, np.newaxis]
