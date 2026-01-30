from typing import Any

import numpy as np
from scipy.stats import qmc

from hybrid_bo import GP, Config, Problem
from hybrid_bo.kernels import (
    Kernel,
    LinearKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    OutputScaleKernel,
    RBFKernel,
)
from hybrid_bo.means import ConstantMean, Mean, ZeroMean
from hybrid_bo.optimizers import (
    CasadiOptimizer,
    LocalOptimizer,
    MultiStartOptimizer,
    Optimizer,
    SciPyLocalOptimizer,
)
from hybrid_bo.parameterization import Parameter


def get_u_train_initial(problem: Problem, config: Config) -> list[np.ndarray]:
    # u_train: np.ndarray = np.array([[-3], [-1], [0]])
    # return [u_train]

    n_initial_points: int = 3
    sampler: qmc.LatinHypercube = qmc.LatinHypercube(
        problem.n_u,
        rng=config.rng,
    )

    u_train_initial_array_unscaled: np.ndarray = sampler.random(
        config.n_runs_bo * n_initial_points,
    )

    u_train_initial_array: np.ndarray = (
        problem.u_lower_bounds_starting_points.T
        + (
            problem.u_upper_bounds_starting_points.T
            - problem.u_lower_bounds_starting_points.T
        )
        * u_train_initial_array_unscaled
    )

    u_train_initial: list[np.ndarray] = []
    i_array: int = 0
    for _ in range(config.n_runs_bo):
        i_array_new = i_array + n_initial_points
        u_train_initial.append(u_train_initial_array[i_array:i_array_new])
        i_array = i_array_new

    return u_train_initial


def get_gp_hybrid_bo(n_inputs: int) -> GP:
    # %% Define kernel

    # Options:
    # OutputScaleKernel(Matern12Kernel(n_inputs_gp))
    # OutputScaleKernel(Matern32Kernel(n_inputs_gp))
    # OutputScaleKernel(Matern52Kernel(n_inputs_gp))
    # OutputScaleKernel(RBFKernel(n_inputs_gp))
    # OutputScaleKernel(LinearKernel(n_inputs_gp))
    kernel: Kernel = OutputScaleKernel(Matern52Kernel(n_inputs))

    # %% Define mean

    # Options:
    # ZeroMean(n_inputs)
    # ConstantMean(n_inputs, Parameter(min=-1.0, max=1.0))
    mean: Mean = ConstantMean(n_inputs, Parameter(min=-1.0, max=1.0))

    # %% Define optimizer

    # Alternative: local_optimizer: LocalOptimizer = SciPyLocalOptimizer("L-BFGS-B")
    plugin_options: dict[str, Any] = {"print_time": 0}
    solver_options: dict[str, Any] = {"print_level": 0}
    local_optimizer: LocalOptimizer = CasadiOptimizer(
        "ipopt", "nonlinear", plugin_options, solver_options
    )
    optimizer: Optimizer = MultiStartOptimizer(local_optimizer)

    gp: GP = GP(n_inputs, kernel, mean, optimizer=optimizer)

    gp.noise_std.fix(0.0)

    return gp


def get_gp_standard_bo(n_inputs: int) -> GP:
    return get_gp_hybrid_bo(n_inputs)


def get_optimizer_hybrid_bo() -> Optimizer:
    plugin_options: dict[str, Any] = {"print_time": 0}
    solver_options: dict[str, Any] = {"print_level": 0}
    local_optimizer: LocalOptimizer = CasadiOptimizer(
        "ipopt",
        "nonlinear",
        plugin_options,
        solver_options,
    )
    optimizer: Optimizer = MultiStartOptimizer(local_optimizer)
    return optimizer


def get_optimizer_standard_bo() -> Optimizer:
    return get_optimizer_hybrid_bo()
