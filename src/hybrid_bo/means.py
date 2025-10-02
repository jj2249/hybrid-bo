from abc import ABC, abstractmethod
from typing import overload, override

import casadi as cas
import numpy as np

from .parameterization import Parameter, Parameterized
from .type_aliases import CasadiType


class Mean(Parameterized, ABC):
    def __init__(self, n_inputs: int = 1) -> None:
        super().__init__()
        self.n_inputs: int = n_inputs

    @overload
    def m(self, X: cas.SX) -> cas.SX: ...
    @overload
    def m(self, X: cas.MX) -> cas.MX: ...
    @overload
    def m(self, X: cas.DM) -> cas.DM: ...
    @overload
    def m(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def m(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
        # Uses attribute value for all trainable parameters
        ...

    @abstractmethod
    def m_variable(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
        # Uses attribute variable for all trainable parameters
        ...


class ZeroMean(Mean):
    def __init__(self, n_inputs: int = 1) -> None:
        super().__init__(n_inputs)

    @override
    def m(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(X, CasadiType):
            return type(X).zeros(X.shape[0], 1)  # pyright: ignore[reportArgumentType]
        return np.zeros((X.shape[0], 1))

    @override
    def m_variable(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
        return self.m(X)


class ConstantMean(Mean):
    def __init__(self, n_inputs: int = 1, constant: Parameter | None = None) -> None:
        super().__init__(n_inputs)
        self.constant: Parameter

        if constant is None:
            self.constant = Parameter(
                "constant",
                value=0.0,
                min=-1.0,
                max=1.0,
                transform_mode="identity",
            )
        else:
            if constant.n_elements != 1:
                msg: str = "constant.n_elements != 1"
                raise Exception(msg)

            self.constant = constant
            self.constant.set_name("constant")

        self.parameters.append(self.constant)

    @override
    def m(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(X, CasadiType):
            return self.constant.value * type(X).ones(X.shape[0], 1)  # pyright: ignore[reportArgumentType]
        return self.constant.value * np.ones((X.shape[0], 1))

    @override
    def m_variable(self, X: CasadiType | np.ndarray) -> CasadiType | np.ndarray:
        return self.constant.variable() * np.ones((X.shape[0], 1))
