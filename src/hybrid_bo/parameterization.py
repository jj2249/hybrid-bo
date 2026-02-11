from typing import overload

import casadi as cas
import numpy as np

from hybrid_bo.array_operations.creation_functions import empty_like
from hybrid_bo.array_operations.nonlinear_functions import exp, log
from hybrid_bo.type_aliases import CasadiType, NumericType, SymbolicType


class Parameter:
    def __init__(
        self,
        name: str = "parameter",
        n_rows: int = 1,
        n_colums: int = 1,
        value: NumericType | None = None,
        min: NumericType | None = None,
        max: NumericType | None = None,
        trainable: bool = True,
        transform_mode: str = "identity",
    ) -> None:
        self.name: str = name
        self.n_rows: int = n_rows
        self.n_columns: int = n_colums
        self.shape: tuple[int, int] = (self.n_rows, self.n_columns)
        self.n_elements: int = self.n_rows * self.n_columns
        self.value: np.ndarray
        self.min: np.ndarray
        self.max: np.ndarray
        self.trainable: bool = trainable
        self.transform_mode: str = transform_mode  # Options: "identity" and "log"
        self.symbol: SymbolicType = SymbolicType.sym(
            self.name,  # pyright: ignore[reportArgumentType]
            self.n_rows,  # pyright: ignore[reportArgumentType]
            self.n_columns,  # pyright: ignore[reportArgumentType]
        )

        if value is None:
            self.value = np.ones((self.n_rows, self.n_columns))
        elif isinstance(value, np.ndarray):
            if value.shape != (self.n_rows, self.n_columns):
                msg: str = "value.shape != (self.n_rows, self.n_columns)"
                raise Exception(msg)
            self.value = value.copy()
        else:
            self.value = value * np.ones((self.n_rows, self.n_columns))

        if min is None:
            self.min = -np.inf * np.ones((self.n_rows, self.n_columns))
        elif isinstance(min, np.ndarray):
            if min.shape != (self.n_rows, self.n_columns):
                msg: str = "min.shape != (self.n_rows, self.n_columns)"
                raise Exception(msg)
            self.min = min.copy()
        else:
            self.min = min * np.ones((self.n_rows, self.n_columns))

        if max is None:
            self.max = np.inf * np.ones((self.n_rows, self.n_columns))
        elif isinstance(max, np.ndarray):
            if max.shape != (self.n_rows, self.n_columns):
                msg: str = "max.shape != (self.n_rows, self.n_columns)"
                raise Exception(msg)
            self.max = max.copy()
        else:
            self.max = max * np.ones((self.n_rows, self.n_columns))

    def fix(self, value: NumericType) -> None:
        self.trainable = False
        if isinstance(value, np.ndarray):
            if value.shape != (self.n_rows, self.n_columns):
                msg: str = "value.shape != (self.n_rows, self.n_columns)"
                raise Exception(msg)
            self.value = value.copy()
        else:
            self.value = value * np.ones((self.n_rows, self.n_columns))

    def set_name(self, name: str) -> None:
        self.name = name
        self.symbol = SymbolicType.sym(self.name, self.n_rows, self.n_columns)  # pyright: ignore[reportArgumentType]

    def variable(self) -> SymbolicType | np.ndarray:
        if self.trainable:
            return self.symbol
        return self.value

    def __getstate__(self) -> dict:
        # Required for pickling to be possible
        state: dict = self.__dict__.copy()
        state["symbol"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        # Required for pickling to be possible
        self.__dict__.update(state)
        self.symbol = SymbolicType.sym(self.name, self.n_rows, self.n_columns)  # pyright: ignore[reportArgumentType]


class Parameterized:
    def __init__(self, parameters: list[Parameter] | None = None) -> None:
        self.parameters: list[Parameter]
        if parameters is None:
            self.parameters = []
        else:
            self.parameters = parameters

    def trainable_parameters(self) -> list[Parameter]:
        return [parameter for parameter in self.parameters if parameter.trainable]

    def parameter_names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]


class JoinedParameters:
    def __init__(self, parameters: list[Parameter]) -> None:
        self.n_elements: int = 0
        self.value: np.ndarray
        self.min: np.ndarray
        self.max: np.ndarray
        self.transform_mode: list[str] = []
        self.symbol: SymbolicType

        value_list: list[np.ndarray] = []
        min_list: list[np.ndarray] = []
        max_list: list[np.ndarray] = []
        symbol_list: list[SymbolicType] = []

        for parameter in parameters:
            self.n_elements += parameter.n_elements
            value_list.append(parameter.value.flatten("F"))
            min_list.append(parameter.min.flatten("F"))
            max_list.append(parameter.max.flatten("F"))
            self.transform_mode += parameter.n_elements * [parameter.transform_mode]
            symbol_list.append(cas.vec(parameter.symbol))

        self.value = np.concatenate(value_list)[:, np.newaxis]
        self.min = np.concatenate(min_list)[:, np.newaxis]
        self.max = np.concatenate(max_list)[:, np.newaxis]
        self.symbol = cas.vertcat(*symbol_list)  # pyright: ignore[reportAttributeAccessIssue]


def set_values(values_flattened: np.ndarray, parameters: list[Parameter]) -> None:
    if values_flattened.shape[1] != 1:
        msg: str = "values_flattened.shape[1] != 1"
        raise Exception(msg)

    i_values: int = 0

    for parameter in parameters:
        value_current = values_flattened[i_values : i_values + parameter.n_elements, 0]
        parameter.value = value_current.reshape(
            (parameter.n_rows, parameter.n_columns),
            order="F",
        )
        i_values += parameter.n_elements


@overload
def transform(X: cas.SX, transform_mode: str | list[str]) -> cas.SX: ...
@overload
def transform(X: cas.MX, transform_mode: str | list[str]) -> cas.MX: ...
@overload
def transform(X: cas.DM, transform_mode: str | list[str]) -> cas.DM: ...
@overload
def transform(X: np.ndarray, transform_mode: str | list[str]) -> np.ndarray: ...


def transform(
    X: CasadiType | np.ndarray,
    transform_mode: str | list[str],
) -> CasadiType | np.ndarray:
    if isinstance(transform_mode, str):
        return transform_single(X, transform_mode)
    return transform_multiple(X, transform_mode)


@overload
def transform_single(X: cas.SX, transform_mode: str) -> cas.SX: ...
@overload
def transform_single(X: cas.MX, transform_mode: str) -> cas.MX: ...
@overload
def transform_single(X: cas.DM, transform_mode: str) -> cas.DM: ...
@overload
def transform_single(X: np.ndarray, transform_mode: str) -> np.ndarray: ...


def transform_single(
    X: CasadiType | np.ndarray,
    transform_mode: str,
) -> CasadiType | np.ndarray:
    if transform_mode == "identity":
        return X
    if transform_mode == "log":
        return log(X)
    msg: str = f"Transform_mode '{transform_mode}' not known."
    raise Exception(msg)


@overload
def transform_multiple(X: cas.SX, transform_mode: list[str]) -> cas.SX: ...
@overload
def transform_multiple(X: cas.MX, transform_mode: list[str]) -> cas.MX: ...
@overload
def transform_multiple(X: cas.DM, transform_mode: list[str]) -> cas.DM: ...
@overload
def transform_multiple(X: np.ndarray, transform_mode: list[str]) -> np.ndarray: ...


def transform_multiple(
    X: CasadiType | np.ndarray,
    transform_mode: list[str],
) -> CasadiType | np.ndarray:
    if X.shape[1] != 1:
        msg: str = "X.shape[1] != 1"
        raise Exception(msg)

    if X.shape[0] != len(transform_mode):
        msg: str = "X.shape[0] != len(transform_mode)"
        raise Exception(msg)

    X_transformed: CasadiType | np.ndarray = empty_like(X)
    for i_mode, mode in enumerate(transform_mode):
        if mode == "identity":
            X_transformed[i_mode, 0] = X[i_mode, 0]

        elif mode == "log":
            X_transformed[i_mode, 0] = log(X[i_mode, 0])

        else:
            msg: str = f"Transform_mode '{mode}' not known."
            raise Exception(msg)

    return X_transformed


@overload
def inv_transform(
    X_transformed: cas.SX,
    transform_mode: str | list[str],
) -> cas.SX: ...
@overload
def inv_transform(X_transformed: cas.MX, transform_mode: str | list[str]) -> cas.MX: ...
@overload
def inv_transform(X_transformed: cas.DM, transform_mode: str | list[str]) -> cas.DM: ...
@overload
def inv_transform(
    X_transformed: np.ndarray,
    transform_mode: str | list[str],
) -> np.ndarray: ...


def inv_transform(
    X_transformed: CasadiType | np.ndarray,
    transform_mode: str | list[str],
) -> CasadiType | np.ndarray:
    if isinstance(transform_mode, str):
        return inv_transform_single(X_transformed, transform_mode)
    return inv_transform_multiple(X_transformed, transform_mode)


@overload
def inv_transform_single(X_transformed: cas.SX, transform_mode: str) -> cas.SX: ...
@overload
def inv_transform_single(X_transformed: cas.MX, transform_mode: str) -> cas.MX: ...
@overload
def inv_transform_single(X_transformed: cas.DM, transform_mode: str) -> cas.DM: ...
@overload
def inv_transform_single(
    X_transformed: np.ndarray,
    transform_mode: str,
) -> np.ndarray: ...


def inv_transform_single(
    X_transformed: CasadiType | np.ndarray,
    transform_mode: str,
) -> CasadiType | np.ndarray:
    if transform_mode == "identity":
        return X_transformed
    if transform_mode == "log":
        return exp(X_transformed)
    msg: str = f"transform_mode '{transform_mode}' not known."
    raise Exception(msg)


@overload
def inv_transform_multiple(
    X_transformed: cas.SX,
    transform_mode: list[str],
) -> cas.SX: ...
@overload
def inv_transform_multiple(
    X_transformed: cas.MX,
    transform_mode: list[str],
) -> cas.MX: ...
@overload
def inv_transform_multiple(
    X_transformed: cas.DM,
    transform_mode: list[str],
) -> cas.DM: ...
@overload
def inv_transform_multiple(
    X_transformed: np.ndarray,
    transform_mode: list[str],
) -> np.ndarray: ...


def inv_transform_multiple(
    X_transformed: CasadiType | np.ndarray,
    transform_mode: list[str],
) -> CasadiType | np.ndarray:
    if X_transformed.shape[1] != 1:
        msg: str = "X.shape[1] != 1"
        raise Exception(msg)

    if X_transformed.shape[0] != len(transform_mode):
        msg: str = "X.shape[0] != len(transform_mode)"
        raise Exception(msg)

    X: CasadiType | np.ndarray = empty_like(X_transformed)
    for i_mode, mode in enumerate(transform_mode):
        if mode == "identity":
            X[i_mode, 0] = X_transformed[i_mode, 0]
        elif mode == "log":
            X[i_mode, 0] = exp(X_transformed[i_mode, 0])
        else:
            msg: str = f"transform_mode '{mode}' not known."
            raise Exception(msg)

    return X
