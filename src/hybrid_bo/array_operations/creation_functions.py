from typing import overload

import casadi as cas
import numpy as np

from ..type_aliases import CasadiType


@overload
def empty_like(X: cas.SX) -> cas.SX: ...
@overload
def empty_like(X: cas.MX) -> cas.MX: ...
@overload
def empty_like(X: cas.DM) -> cas.DM: ...
@overload
def empty_like(X: np.ndarray) -> np.ndarray: ...


def empty_like(X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
    if isinstance(X, CasadiType):
        return type(X)(X.shape[0], X.shape[1])
    return np.empty_like(X)
