import numpy as np

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


def u_train_initial(problem: Problem, config: Config) -> list[np.ndarray]: ...


def gp_hybrid_bo(n_inputs: int) -> GP: ...


def gp_standard_bo(n_inputs: int) -> GP: ...


def optimizer_hybrid_bo() -> Optimizer: ...


def optimizer_standard_bo() -> Optimizer: ...
