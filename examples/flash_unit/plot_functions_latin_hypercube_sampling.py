import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.routines.utils import get_evaluation_data

from .custom_problem import plot_physical_boundary
from .plot_functions_standard_bo import plot_f as standard_bo_plot_f
from .plot_functions_standard_bo import plot_output_gp as standard_bo_plot_output_gp

plt.rcParams["text.usetex"] = True

n_eval_points: list[int] = [51, 51]


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
) -> None:
    # %% Get evaluation points

    u_eval: np.ndarray
    x_eval: np.ndarray
    f_eval: np.ndarray
    u_eval_grid: list[np.ndarray]
    x_eval_grid: list[np.ndarray]
    f_eval_grid: np.ndarray
    u_eval, x_eval, f_eval, u_eval_grid, x_eval_grid, f_eval_grid = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        n_eval_points,
        False,
        config.rng,
    )

    x_eval_original: np.ndarray
    f_eval_original: np.ndarray
    x_eval_grid_original: list[np.ndarray]
    f_eval_grid_original: np.ndarray
    (
        _,
        x_eval_original,
        f_eval_original,
        _,
        x_eval_grid_original,
        f_eval_grid_original,
    ) = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        n_eval_points,
        True,
        config.rng,
    )

    print("Preparing plotting done.")

    # %% Plotting

    ax: Axes = plot_f(
        u_eval_grid,
        f_eval_grid,
        u,
        n_initial_points,
        i_iteration,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    ax: Axes = plot_f(
        u_eval_grid,
        f_eval_grid_original,
        u,
        n_initial_points,
        i_iteration,
        True,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    print("Plotting f done.")

    ax1: Axes
    ax2: Axes
    ax1, ax2 = plot_output_gp(
        config,
        u_eval,
        u_eval_grid,
        u,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        n_initial_points,
        i_iteration,
    )

    plot_physical_boundary(problem, ax1)  # pyright: ignore[reportArgumentType]
    plot_physical_boundary(problem, ax2)  # pyright: ignore[reportArgumentType]

    print("Plotting output of GP done.")

    plt.show()


def plot_f(
    u_eval_grid: list[np.ndarray],
    f_eval_grid: np.ndarray,
    u: np.ndarray,
    n_initial_points: int,
    i_iteration: int,
    original: bool = False,
) -> Axes:
    return standard_bo_plot_f(
        u_eval_grid,
        f_eval_grid,
        u,
        n_initial_points,
        i_iteration,
        original,
    )


def plot_output_gp(
    config: Config,
    u_eval: np.ndarray,
    u_eval_grid: list[np.ndarray],
    u: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    n_initial_points: int,
    i_iteration: int,
) -> tuple[Axes, Axes]:
    return standard_bo_plot_output_gp(
        config,
        u_eval,
        u_eval_grid,
        u,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        n_initial_points,
        i_iteration,
    )
