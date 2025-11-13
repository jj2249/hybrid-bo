from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.routines.utils import get_evaluation_data

from .plot_functions_standard_bo import plot_output_gp as standard_bo_plot_output_gp

if TYPE_CHECKING:
    from matplotlib.figure import Figure

plt.rcParams["text.usetex"] = True


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
    # See https://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data
    confidence_factors = {
        25: 0.32,
        50: 0.67,
        68: 0.99,
        80: 1.28,
        90: 1.64,
        95: 1.96,
        99: 2.58,
    }
    if config.confidence_level not in confidence_factors:
        msg: str = (
            "For 'confidence_level',"
            "only the values 25, 50, 68, 80, 90, 95 and 99 are allowed."
        )
        raise Exception(msg)

    # %% Get evaluation points

    u_eval: np.ndarray
    f_eval: np.ndarray
    u_eval, _, f_eval, _, _, _ = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.check_bounds_evaluate_problem,
        config.n_eval_points,
        False,
        config.seed,
    )

    u_eval_transformed: np.ndarray = input_transformer_gp.transform(u_eval)  # pyright: ignore[reportAssignmentType]

    mean_gp_eval_transformed: np.ndarray
    var_gp_eval_transformed: np.ndarray
    mean_gp_eval_transformed, var_gp_eval_transformed = gp.predict(u_eval_transformed)  # pyright: ignore[reportAssignmentType]

    mean_gp_eval: np.ndarray = output_transformer_gp.inverse_transform(
        mean_gp_eval_transformed,
    )  # pyright: ignore[reportAssignmentType]
    std_gp_eval: np.ndarray = (
        np.sqrt(np.diag(var_gp_eval_transformed))[:, np.newaxis]
        / output_transformer_gp.slope
    )

    print("Preparing plotting done.")

    # %% Plotting

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6, 6))

    plot_output_gp(
        ax,
        u_eval,
        f_eval,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        u_train,
        f_train,
        n_initial_training_points,
        confidence_factors[config.confidence_level],
    )

    print("Plotting output of GP done.")

    ax.set_xlabel("u")
    fig.suptitle(f"Iteration {i_iteration}")
    fig.tight_layout()
    plt.show()


def plot_output_gp(
    ax: Axes,
    u_eval: np.ndarray,
    f_eval: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    u_train: np.ndarray,
    f_train: np.ndarray,
    n_initial_training_points: int,
    confidence_factor: float,
) -> None:
    standard_bo_plot_output_gp(
        ax,
        u_eval,
        f_eval,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        u_train,
        f_train,
        n_initial_training_points,
        confidence_factor,
    )
