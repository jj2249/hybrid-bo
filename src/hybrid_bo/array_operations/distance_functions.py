from typing import overload

import casadi as cas
import numpy as np

from ..type_aliases import CasadiType
from .arithmetic_functions import sum
from .comparison_functions import fmax
from .nonlinear_functions import sqrt
from .shape_functions import tile


@overload
def squared_distance(X1: cas.SX, X2: cas.SX, min: float) -> cas.SX: ...
@overload
def squared_distance(X1: cas.SX, X2: np.ndarray, min: float) -> cas.SX: ...
@overload
def squared_distance(X1: np.ndarray, X2: cas.SX, min: float) -> cas.SX: ...
@overload
def squared_distance(X1: cas.MX, X2: cas.MX, min: float) -> cas.MX: ...
@overload
def squared_distance(X1: cas.MX, X2: np.ndarray, min: float) -> cas.MX: ...
@overload
def squared_distance(X1: np.ndarray, X2: cas.MX, min: float) -> cas.MX: ...
@overload
def squared_distance(X1: cas.DM, X2: cas.DM, min: float) -> cas.DM: ...
@overload
def squared_distance(X1: cas.DM, X2: np.ndarray, min: float) -> cas.DM: ...
@overload
def squared_distance(X1: np.ndarray, X2: cas.DM, min: float) -> cas.DM: ...
@overload
def squared_distance(X1: np.ndarray, X2: np.ndarray, min: float) -> np.ndarray: ...


def squared_distance(
    X1: CasadiType | np.ndarray,
    X2: CasadiType | np.ndarray,
    min: float = 0.0,
) -> CasadiType | np.ndarray:
    """Computes the squared Euclidean distance between every row of X1 and X2."""

    # See https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/stationary.py,
    # method Stationary._unscaled_dist()

    if (len(X1.shape) != 2) or (len(X2.shape) != 2):  # noqa: PLR2004
        msg: str = "(len(X1.shape) != 2) or (len(X2.shape) != 2)"
        raise Exception(msg)

    if X1.shape[1] != X2.shape[1]:
        msg: str = "X1.shape[1] != X2.shape[1]"
        raise Exception(msg)

    n_rows: int = X1.shape[0]
    n_colums: int = X2.shape[0]

    X1_squared_summed_tiled: CasadiType | np.ndarray = tile(
        sum(X1**2, 1),
        [1, n_colums],
    )

    X2_squared_summed_tiled: CasadiType | np.ndarray = tile(
        sum(X2**2, 1).T,
        [n_rows, 1],
    )

    squared_distances: CasadiType | np.ndarray = -2.0 * (X1 @ X2.T) + (
        X1_squared_summed_tiled + X2_squared_summed_tiled
    )

    return fmax(squared_distances, min)  # pyright: ignore[reportCallIssue, reportArgumentType]


@overload
def distance(X1: cas.SX, X2: cas.SX, min_squared: float) -> cas.SX: ...
@overload
def distance(X1: cas.SX, X2: np.ndarray, min_squared: float) -> cas.SX: ...
@overload
def distance(X1: np.ndarray, X2: cas.SX, min_squared: float) -> cas.SX: ...
@overload
def distance(X1: cas.MX, X2: cas.MX, min_squared: float) -> cas.MX: ...
@overload
def distance(X1: cas.MX, X2: np.ndarray, min_squared: float) -> cas.MX: ...
@overload
def distance(X1: np.ndarray, X2: cas.MX, min_squared: float) -> cas.MX: ...
@overload
def distance(X1: cas.DM, X2: cas.DM, min_squared: float) -> cas.DM: ...
@overload
def distance(X1: cas.DM, X2: np.ndarray, min_squared: float) -> cas.DM: ...
@overload
def distance(X1: np.ndarray, X2: cas.DM, min_squared: float) -> cas.DM: ...
@overload
def distance(X1: np.ndarray, X2: np.ndarray, min_squared: float) -> np.ndarray: ...


def distance(
    X1: CasadiType | np.ndarray,
    X2: CasadiType | np.ndarray,
    min_squared: float = 0.0,
) -> CasadiType | np.ndarray:
    return sqrt(squared_distance(X1, X2, min_squared))  # pyright: ignore[reportArgumentType, reportCallIssue]
