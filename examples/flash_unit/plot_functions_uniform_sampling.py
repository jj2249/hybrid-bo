import numpy as np

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer

from .plot_functions_latin_hypercube_sampling import (
    create_plots as latin_hypercube_sampling_create_plots,
)


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f_train: np.ndarray,
    u_train: np.ndarray,
    x_train: np.ndarray,
    n_initial_training_points: int,
    i_iteration: int,
) -> None:
    latin_hypercube_sampling_create_plots(
        problem,
        config,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        f_train,
        u_train,
        x_train,
        n_initial_training_points,
        i_iteration,
    )
