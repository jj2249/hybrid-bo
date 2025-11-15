from typing import overload

import casadi as cas
import numpy as np

from hybrid_bo.type_aliases import CasadiType


@overload
def cholesky(X: cas.SX) -> cas.SX: ...
@overload
def cholesky(X: cas.MX) -> cas.MX: ...
@overload
def cholesky(X: cas.DM) -> cas.DM: ...
@overload
def cholesky(X: np.ndarray) -> np.ndarray: ...


def cholesky(
    X: CasadiType | np.ndarray,
    epsilon: float = 1.0e-10,
) -> CasadiType | np.ndarray:
    if isinstance(X, cas.MX):
        n_rows: int = X.size1()
        L: cas.MX = cas.MX.zeros(n_rows, n_rows)  # pyright: ignore[reportArgumentType]
        for i in range(n_rows):
            for j in range(i + 1):
                s: cas.MX | float = cas.dot(L[i, :j], L[j, :j]) if j > 0 else 0.0
                if j < i:
                    # Off-diagonal
                    L[i, j] = (X[i, j] - s) / L[j, j]
                else:
                    # Diagonal
                    L[i, i] = cas.sqrt(X[i, i] - s + epsilon)
        return L

    if isinstance(X, cas.SX | cas.DM):
        # Casadi returns upper triangular matrix!
        return cas.chol(X).T

    return np.linalg.cholesky(X)


@overload
def solve(A: cas.SX, B: cas.SX) -> cas.SX: ...
@overload
def solve(A: cas.SX, B: np.ndarray) -> cas.SX: ...
@overload
def solve(A: np.ndarray, B: cas.SX) -> cas.SX: ...
@overload
def solve(A: cas.MX, B: cas.MX) -> cas.MX: ...
@overload
def solve(A: cas.MX, B: np.ndarray) -> cas.MX: ...
@overload
def solve(A: np.ndarray, B: cas.MX) -> cas.MX: ...
@overload
def solve(A: cas.DM, B: cas.DM) -> cas.DM: ...
@overload
def solve(A: cas.DM, B: np.ndarray) -> cas.DM: ...
@overload
def solve(A: np.ndarray, B: cas.DM) -> cas.DM: ...
@overload
def solve(A: np.ndarray, B: np.ndarray) -> np.ndarray: ...


def solve(
    A: CasadiType | np.ndarray,
    B: CasadiType | np.ndarray,
) -> CasadiType | np.ndarray:
    if isinstance(A, CasadiType) or isinstance(B, CasadiType):
        return cas.solve(A, B)
    return np.linalg.solve(A, B)
