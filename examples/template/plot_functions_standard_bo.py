import numpy as np

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    n_initial_points: int,
    i_iteration: int,
) -> None: ...
