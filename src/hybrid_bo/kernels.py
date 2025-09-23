from abc import ABC, abstractmethod
from typing import overload, override

import casadi as cas
import numpy as np

from .parameterization import Parameter, Parameterized
from .type_aliases import CasadiType


class Kernel(Parameterized, ABC):
    def __init__(self, n_inputs: int = 1) -> None:
        super().__init__()
        self.n_inputs: int = n_inputs

    @overload
    def k(self, X1: cas.SX, X2: cas.SX) -> cas.SX: ...
    @overload
    def k(self, X1: cas.SX, X2: np.ndarray) -> cas.SX: ...
    @overload
    def k(self, X1: np.ndarray, X2: cas.SX) -> cas.SX: ...
    @overload
    def k(self, X1: cas.MX, X2: cas.MX) -> cas.MX: ...
    @overload
    def k(self, X1: cas.MX, X2: np.ndarray) -> cas.MX: ...
    @overload
    def k(self, X1: np.ndarray, X2: cas.MX) -> cas.MX: ...
    @overload
    def k(self, X1: cas.DM, X2: cas.DM) -> cas.DM: ...
    @overload
    def k(self, X1: cas.DM, X2: np.ndarray) -> cas.DM: ...
    @overload
    def k(self, X1: np.ndarray, X2: cas.DM) -> cas.DM: ...
    @overload
    def k(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def k(
        self, X1: CasadiType | np.ndarray, X2: CasadiType | np.ndarray
    ) -> CasadiType | np.ndarray:
        # Uses attribute value for all trainable parameters
        ...

    @abstractmethod
    def k_variable(
        self, X1: CasadiType | np.ndarray, X2: CasadiType | np.ndarray
    ) -> CasadiType | np.ndarray:
        # Uses attribute variable for all trainable parameters
        ...


class OutputScaleKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        outputscale: Parameter | None = None,
    ) -> None:
        super().__init__(base_kernel.n_inputs)

        self.base_kernel: Kernel = base_kernel
        self.outputscale: Parameter

        if outputscale is None:
            self.outputscale = Parameter(
                "outputscale",
                min=1e-5,
                max=1e5,
                transform_mode="log",
            )
        else:
            if outputscale.n_elements != 1:
                msg: str = "outputscale.n_elements != 1"
                raise Exception(msg)

            self.outputscale = outputscale
            self.outputscale.set_name("outputscale")

        # Add prefix to parameter names of base_kernel for uniqueness
        for parameter in self.base_kernel.parameters:
            parameter.set_name("base_kernel." + parameter.name)

        self.parameters.append(self.outputscale)
        self.parameters += self.base_kernel.parameters

    @override
    def k(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X1: CasadiType | np.ndarray,
        X2: CasadiType | np.ndarray,
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != self.n_inputs:
            msg: str = "X1.shape[1] != self.n_inputs"
            raise Exception(msg)
        if X2.shape[1] != self.n_inputs:
            msg: str = "X2.shape[1] != self.n_inputs"
            raise Exception(msg)

        return self.outputscale.value * self.base_kernel.k(X1, X2)  # pyright: ignore[reportCallIssue, reportArgumentType]

    @override
    def k_variable(
        self,
        X1: CasadiType | np.ndarray,
        X2: CasadiType | np.ndarray,
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != self.n_inputs:
            msg: str = "X1.shape[1] != self.n_inputs"
            raise Exception(msg)
        if X2.shape[1] != self.n_inputs:
            msg: str = "X2.shape[1] != self.n_inputs"
            raise Exception(msg)

        return self.outputscale.variable() * self.base_kernel.k_variable(X1, X2)


class ZeroKernel(Kernel):
    def __init__(self, n_inputs: int = 1) -> None:
        super().__init__(n_inputs)

    @override
    def k(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X1: CasadiType | np.ndarray,
        X2: CasadiType | np.ndarray,
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != X2.shape[1]:
            msg: str = "X1.shape[1] != X2.shape[1]"
            raise Exception(msg)

        if isinstance(X1, CasadiType):
            return type(X1).zeros(X1.shape[0], X2.shape[0])
        if isinstance(X2, CasadiType):
            return type(X2).zeros(X1.shape[0], X2.shape[0])
        return np.zeros((X1.shape[0], X2.shape[0]))

    @override
    def k_variable(
        self,
        X1: CasadiType | np.ndarray,
        X2: CasadiType | np.ndarray,
    ) -> CasadiType | np.ndarray:
        return self.k(X1, X2)


class ConstantKernel(Kernel):
    def __init__(self, n_inputs: int = 1, constant: Parameter | None = None) -> None:
        super().__init__(n_inputs)
        self.constant: Parameter

        if constant is None:
            self.constant = Parameter(
                "constant",
                value=0,
                min=-1,
                max=1,
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
    def k(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X1: CasadiType | np.ndarray,
        X2: CasadiType | np.ndarray,
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != X2.shape[1]:
            msg: str = "X1.shape[1] != X2.shape[1]"
            raise Exception(msg)

        if isinstance(X1, CasadiType):
            return self.constant.value * type(X1).ones(X1.shape[0], X2.shape[0])
        if isinstance(X2, CasadiType):
            return self.constant.value * type(X2).ones(X1.shape[0], X2.shape[0])
        return self.constant.value * np.ones((X1.shape[0], X2.shape[0]))

    def k_variable(
        self, X1: CasadiType | np.ndarray, X2: CasadiType | np.ndarray
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != X2.shape[1]:
            msg: str = "X1.shape[1] != X2.shape[1]"
            raise Exception(msg)

        return self.constant.variable() * np.ones((X1.shape[0], X2.shape[0]))


class LinearKernel(Kernel):
    def __init__(
        self, n_inputs: int = 1, scale: Parameter | None = None, ARD: bool = True
    ) -> None:
        # Parameter ARD is only important if scale is None
        # Otherwise the given scale determines if ARD is True or False
        super().__init__(n_inputs)
        self.scale: Parameter
        self.ARD: bool

        if scale is None:
            if ARD:
                self.scale = Parameter(
                    "scale",
                    self.n_inputs,
                    min=1e-5,
                    max=1e5,
                    transform_mode="log",
                )
            else:
                self.scale = Parameter(
                    "scale",
                    1,
                    min=1e-5,
                    max=1e5,
                    transform_mode="log",
                )
            self.ARD = ARD

        else:
            if (scale.n_elements != 1) and (scale.shape != (self.n_inputs, 1)):
                msg: str = (
                    "(scale.n_elements != 1) and (scale.shape != (self.n_inputs, 1))"
                )
                raise Exception(msg)

            if scale.n_elements == 1:
                self.ARD = False
            else:
                self.ARD = True

            self.scale = scale
            self.scale.set_name("scale")

        self.parameters.append(self.scale)

    def k(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, X1: CasadiType | np.ndarray, X2: CasadiType | np.ndarray
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != self.n_inputs:
            msg: str = "X1.shape[1] != self.n_inputs"
            raise Exception(msg)
        if X2.shape[1] != self.n_inputs:
            msg: str = "X2.shape[1] != self.n_inputs"
            raise Exception(msg)

        if self.ARD:
            scale_sqrt_tiled_1: np.ndarray = np.tile(
                np.sqrt(self.scale.value.T),
                (X1.shape[0], 1),
            )

            scale_sqrt_tiled_2: np.ndarray = np.tile(
                np.sqrt(self.scale.value.T),
                (X2.shape[0], 1),
            )

            return (X1 * scale_sqrt_tiled_1) @ (X2 * scale_sqrt_tiled_2).T

        return (X1 @ X2.T) * self.scale.value

    def k_variable(
        self, X1: CasadiType | np.ndarray, X2: CasadiType | np.ndarray
    ) -> CasadiType | np.ndarray:
        if X1.shape[1] != self.n_inputs:
            msg: str = "X1.shape[1] != self.n_inputs"
            raise Exception(msg)
        if X2.shape[1] != self.n_inputs:
            msg: str = "X2.shape[1] != self.n_inputs"
            raise Exception(msg)

        if self.ARD:
            scale_sqrt_tiled_1: np.ndarray = np.tile(
                np.sqrt(self.scale.value.T),
                (X1.shape[0], 1),
            )

            scale_sqrt_tiled_2: np.ndarray = np.tile(
                np.sqrt(self.scale.value.T),
                (X2.shape[0], 1),
            )

            return (X1 * scale_sqrt_tiled_1) @ (X2 * scale_sqrt_tiled_2).T

        return (X1 @ X2.T) * self.scale.variable()
