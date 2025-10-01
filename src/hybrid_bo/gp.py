from typing import Any, overload

import casadi as cas
import numpy as np

from .array_operations import cholesky, diag, log, solve, sum
from .kernels import Kernel
from .means import Mean
from .optimizers import CasadiOptimizer, LocalOptimizer, MultiStartOptimizer, Optimizer
from .parameterization import (
    JoinedParameters,
    Parameter,
    Parameterized,
    inv_transform,
    set_values,
    transform,
)
from .type_aliases import CasadiType, SymbolicType


class GP(Parameterized):
    def __init__(
        self,
        n_inputs: int,
        kernel: Kernel,
        mean: Mean,
        noise_std: Parameter | None = None,
        X_train: np.ndarray | None = None,
        Y_train: np.ndarray | None = None,
        optimizer: Optimizer | None = None,
        epsilon: float = 1e-10,
    ) -> None:
        super().__init__()
        self.n_inputs: int = n_inputs
        self.kernel: Kernel = kernel
        self.mean: Mean = mean
        self.noise_std: Parameter
        self.X_train: np.ndarray
        self.Y_train: np.ndarray
        self.n_training_points: int
        self.optimizer: Optimizer
        self.epsilon: float = epsilon

        if kernel.n_inputs != n_inputs:
            msg: str = "kernel.n_inputs != n_inputs"
            raise Exception(msg)
        if mean.n_inputs != n_inputs:
            msg: str = "mean.n_inputs != n_inputs"
            raise Exception(msg)

        if noise_std is None:
            self.noise_std = Parameter(
                "noise_std",
                value=1e-2,
                min=1e-6,
                max=10,
                transform_mode="log",
            )

        else:
            if noise_std.shape != (1, 1):
                msg: str = "noise_std.shape != (1, 1)"
                raise Exception(msg)

            self.noise_std = noise_std
            self.noise_std.set_name("noise_std")

        if (X_train is None) and (Y_train is not None):
            msg: str = "(X_train is None) and (Y_train is not None)"
            raise Exception(msg)

        if (X_train is not None) and (Y_train is None):
            msg: str = "(X_train is not None) and (Y_train is None)"
            raise Exception(msg)

        if (X_train is not None) and (Y_train is not None):
            if X_train.shape[1] != n_inputs:
                msg: str = "X_train.shape[1] != n_inputs"
                raise Exception(msg)
            if X_train.shape[0] != Y_train.shape[0]:
                msg: str = "X_train.shape[0] != Y_train.shape[0]"
                raise Exception(msg)

            self.X_train = X_train
            self.Y_train = Y_train
            self.n_training_points = X_train.shape[0]

        else:  # (X_train is None) and (Y_train is None):
            self.X_train = np.empty([0])
            self.Y_train = np.empty([0])
            self.n_training_points = 0

        # Add prefix to parameter names of kernel and mean
        for kernel_parameter in self.kernel.parameters:
            kernel_parameter.set_name("kernel." + kernel_parameter.name)
        for mean_parameter in self.mean.parameters:
            mean_parameter.set_name("mean." + mean_parameter.name)

        self.parameters.append(self.noise_std)
        self.parameters += self.kernel.parameters
        self.parameters += self.mean.parameters

        if optimizer is None:
            plugin_options: dict[str, Any] = {"print_time": 0}
            solver_options: dict[str, Any] = {"print_level": 0}
            local_optimizer: LocalOptimizer = CasadiOptimizer(
                "ipopt",
                "nonlinear",
                plugin_options,
                solver_options,
            )
            self.optimizer = MultiStartOptimizer(local_optimizer)
        else:
            self.optimizer = optimizer

    def set_XY_train(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        if X_train.shape[1] != self.n_inputs:
            msg: str = "X_train.shape[1] != self.n_inputs"
            raise Exception(msg)
        if Y_train.shape[1] != 1:
            msg: str = "Y_train.shape[1] != 1"
            raise Exception(msg)
        if X_train.shape[0] != Y_train.shape[0]:
            msg: str = "X_train.shape[0] != Y_train.shape[0]"
            raise Exception(msg)

        self.X_train = X_train
        self.Y_train = Y_train
        self.n_training_points = X_train.shape[0]

    def create_training_problem(self) -> JoinedParameters:
        trainables: JoinedParameters = JoinedParameters(self.trainable_parameters())

        if trainables.n_elements == 0:
            self.optimizer.set_problem(0, cas.Function())

        else:
            # Transformed domain, used for optimization problem
            x: SymbolicType = SymbolicType.sym("x", trainables.n_elements, 1)  # pyright: ignore[reportArgumentType]

            # Original domain
            x_original: SymbolicType = inv_transform(x, trainables.transform_mode)

            # Transformed domain
            lower_bounds: np.ndarray = transform(
                trainables.min,
                trainables.transform_mode,
            )

            # Transformed domain
            upper_bounds: np.ndarray = transform(
                trainables.max,
                trainables.transform_mode,
            )

            # Original domain
            negative_log_mll_function: cas.Function = cas.Function(
                "log_mll_function",
                [trainables.symbol],
                [-self.log_mll_variable()],
            )

            # Transformed domain
            f: cas.Function = cas.Function(
                "f",
                [x],
                [negative_log_mll_function(x_original)],
            )

            self.optimizer.set_problem(
                trainables.n_elements,
                f,
                None,
                None,
                lower_bounds,
                upper_bounds,
            )

        return trainables

    def solve_training_problem(self) -> None:
        trainables: JoinedParameters = JoinedParameters(self.trainable_parameters())
        solution: dict[str, float | np.ndarray] | None = self.optimizer.solve()

        if solution is not None:
            x_opt_original: np.ndarray = inv_transform(  # pyright: ignore[reportCallIssue]
                solution["x"],  # pyright: ignore[reportArgumentType]
                trainables.transform_mode,
            )
            set_values(x_opt_original, self.trainable_parameters())

        else:
            print("Could not solve the training problem. Parameters were not changed.")

    @overload
    def predict(self, X: cas.SX) -> tuple[cas.SX, cas.SX]: ...
    @overload
    def predict(self, X: cas.MX) -> tuple[cas.MX, cas.MX]: ...
    @overload
    def predict(self, X: cas.DM) -> tuple[cas.DM, cas.DM]: ...
    @overload
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(
        self, X: CasadiType | np.ndarray
    ) -> tuple[CasadiType | np.ndarray, CasadiType | np.ndarray]:
        m_t: CasadiType | np.ndarray = self.mean.m(self.X_train)
        K_tt: CasadiType | np.ndarray = self.kernel.k(self.X_train, self.X_train) + (
            self.noise_std.value**2 + self.epsilon
        ) * np.eye(self.n_training_points)
        L_tt: CasadiType | np.ndarray = np.linalg.cholesky(K_tt)

        m_x: CasadiType | np.ndarray = self.mean.m(X)
        K_xx: CasadiType | np.ndarray = self.kernel.k(X, X)  # pyright: ignore[reportArgumentType, reportCallIssue]
        K_xt: CasadiType | np.ndarray = self.kernel.k(X, self.X_train)

        alpha: CasadiType | np.ndarray = solve(L_tt, (self.Y_train - m_t))
        beta: CasadiType | np.ndarray = solve(L_tt, K_xt.T)

        mean: CasadiType | np.ndarray = m_x + beta.T @ alpha
        covariance: CasadiType | np.ndarray = K_xx - beta.T @ beta

        return mean, covariance

    def log_mll(self) -> float:
        m_t: np.ndarray = self.mean.m(self.X_train)  # pyright: ignore[reportAssignmentType]
        K_tt: np.ndarray = self.kernel.k(self.X_train, self.X_train) + (
            self.noise_std.value**2 + self.epsilon
        ) * np.eye(self.n_training_points)

        L_tt: np.ndarray = np.linalg.cholesky(K_tt)
        alpha: np.ndarray = np.linalg.solve(
            L_tt.T,
            np.linalg.solve(L_tt, (self.Y_train - m_t)),
        )

        return -0.5 * (
            (self.Y_train - m_t).T @ alpha
            + 2 * np.sum(np.log(np.diag(L_tt)))
            + self.n_training_points * np.log(2 * np.pi)
        )

    def log_mll_variable(self) -> CasadiType | float:
        trainables: JoinedParameters = JoinedParameters(self.trainable_parameters())
        if trainables.n_elements == 0:
            return self.log_mll()

        m_t: CasadiType = self.mean.m_variable(self.X_train)  # pyright: ignore[reportAssignmentType]

        K_tt: CasadiType = self.kernel.k_variable(self.X_train, self.X_train) + (
            self.noise_std.variable() ** 2 + self.epsilon
        ) * np.eye(self.n_training_points)  # pyright: ignore[reportAssignmentType]

        L_tt: CasadiType = cholesky(K_tt)
        alpha: CasadiType = solve(L_tt.T, solve(L_tt, (self.Y_train - m_t)))

        return -0.5 * (
            (self.Y_train - m_t).T @ alpha
            + 2 * sum(log(diag(L_tt)))
            + self.n_training_points * np.log(2 * np.pi)
        )
