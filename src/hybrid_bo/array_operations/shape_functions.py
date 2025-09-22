from typing import overload

import casadi as cas
import numpy as np

from ..type_aliases import CasadiType


@overload
def tile(X: cas.SX, repetitions: list[int]) -> cas.SX: ...
@overload
def tile(X: cas.MX, repetitions: list[int]) -> cas.MX: ...
@overload
def tile(X: cas.DM, repetitions: list[int]) -> cas.DM: ...
@overload
def tile(X: np.ndarray, repetitions: list[int]) -> np.ndarray: ...


def tile(
    X: CasadiType | np.ndarray,
    repetitions: list[int],
) -> CasadiType | np.ndarray:
    if len(X.shape) != 2:
        msg: str = "len(X.shape) != 2."
        raise Exception(msg)
    if len(repetitions) != 2:
        msg: str = "len(repetitions) != 2"
        raise Exception(msg)

    if isinstance(X, CasadiType):
        return cas.repmat(X, repetitions[0], repetitions[1])
    return np.tile(X, repetitions)


@overload
def diag(X: cas.SX) -> cas.SX: ...
@overload
def diag(X: cas.MX) -> cas.MX: ...
@overload
def diag(X: cas.DM) -> cas.DM: ...
@overload
def diag(X: np.ndarray) -> np.ndarray: ...


def diag(X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
    if len(X.shape) != 2:
        msg: str = "len(X.shape) != 2"
        raise Exception(msg)

    if isinstance(X, CasadiType):
        return cas.diag(X)

    X_diag: np.ndarray = np.diag(X)
    if len(X_diag.shape) == 1:
        X_diag = X_diag[:, np.newaxis]
    return X_diag
