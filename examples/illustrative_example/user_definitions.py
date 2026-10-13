import numpy as np
from scipy.stats import qmc

from hybrid_bo import GP, Config, Problem
from hybrid_bo.kernels import (
    Matern52Kernel,
    OutputScaleKernel,
)
from hybrid_bo.means import ZeroMean
from hybrid_bo.optimizers import (
    CasadiOptimizer,
    MultiStartOptimizer,
    Optimizer,
)


def get_u_initial(problem: Problem, config: Config) -> list[np.ndarray]:
    n_initial_points = 2
    sampler = qmc.LatinHypercube(
        problem.n_u,
        rng=config.rng,
    )

    u_initial_array_unscaled = sampler.random(
        config.n_runs_bo * n_initial_points,
    )

    u_initial_array = (
        problem.u_lower_bounds_starting_points.T
        + (
            problem.u_upper_bounds_starting_points.T
            - problem.u_lower_bounds_starting_points.T
        )
        * u_initial_array_unscaled
    )

    u_initial = []
    i_array = 0
    for _ in range(config.n_runs_bo):
        i_array_new = i_array + n_initial_points
        u_initial.append(u_initial_array[i_array:i_array_new])
        i_array = i_array_new

    return u_initial


def get_gp_hybrid_bo(n_inputs: int) -> GP:
    # %% Define kernel

    # Options:
    # OutputScaleKernel(Matern12Kernel(n_inputs_gp))
    # OutputScaleKernel(Matern32Kernel(n_inputs_gp))
    # OutputScaleKernel(Matern52Kernel(n_inputs_gp))
    # OutputScaleKernel(RBFKernel(n_inputs_gp))
    # OutputScaleKernel(LinearKernel(n_inputs_gp))
    kernel = OutputScaleKernel(Matern52Kernel(n_inputs))

    # %% Define mean

    # Options:
    # ZeroMean(n_inputs)
    # ConstantMean(n_inputs, Parameter(min=-1.0, max=1.0))
    mean = ZeroMean(n_inputs)

    # %% Define optimizer

    # Alternative: local_optimizer: LocalOptimizer = SciPyLocalOptimizer("L-BFGS-B")
    plugin_options = {"print_time": 0}
    solver_options = {"print_level": 0}
    local_optimizer = CasadiOptimizer(
        "ipopt",
        "nonlinear",
        plugin_options,
        solver_options,
    )
    optimizer = MultiStartOptimizer(local_optimizer)

    gp = GP(n_inputs, kernel, mean, optimizer=optimizer)

    gp.noise_std.fix(0.0)

    return gp


def get_gp_standard_bo(n_inputs: int) -> GP:
    return get_gp_hybrid_bo(n_inputs)


def get_optimizer_hybrid_bo() -> Optimizer:
    plugin_options = {"print_time": 0}
    solver_options = {"print_level": 0}
    local_optimizer = CasadiOptimizer(
        "ipopt",
        "nonlinear",
        plugin_options,
        solver_options,
    )
    optimizer = MultiStartOptimizer(local_optimizer)
    return optimizer


def get_optimizer_standard_bo() -> Optimizer:
    return get_optimizer_hybrid_bo()
