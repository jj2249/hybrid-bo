import matplotlib.pyplot as plt
import numpy as np

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer

from .plot_functions_latin_hypercube_sampling import (
    create_plots as latin_hypercube_sampling_create_plots,
)

# plt.rcParams["text.usetex"] = True


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    n_points_initial: int,
    i_iteration: int,
) -> None:
    latin_hypercube_sampling_create_plots(
        problem,
        config,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        f,
        u,
        x,
        n_points_initial,
        i_iteration,
    )
