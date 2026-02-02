from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import LogLocator, ScalarFormatter

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.routines.utils import ei_gp, get_evaluation_data

from .custom_problem import plot_physical_boundary

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure


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
    # %% Get evaluation points

    u_eval: np.ndarray
    u_eval_grid: list[np.ndarray]
    f_eval_grid: np.ndarray
    u_eval, _, _, u_eval_grid, _, f_eval_grid = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.n_eval_points,
        False,
        config.rng,
    )

    f_eval_grid_original: np.ndarray
    (
        _,
        _,
        _,
        _,
        _,
        f_eval_grid_original,
    ) = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.n_eval_points,
        True,
        config.rng,
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

    mean_gp_eval_grid: np.ndarray = mean_gp_eval.reshape(config.n_eval_points)
    std_gp_eval_grid: np.ndarray = std_gp_eval.reshape(config.n_eval_points)

    print("Preparing plotting done.")

    # %% Plotting

    ax: Axes = plot_f(
        u_eval_grid,
        f_eval_grid,
        u_train,
        n_initial_training_points,
        i_iteration,
        False,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    ax: Axes = plot_f(
        u_eval_grid,
        f_eval_grid_original,
        u_train,
        n_initial_training_points,
        i_iteration,
        True,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    print("Plotting f done.")

    ax: Axes = plot_acq(
        config,
        u_train,
        u_eval_grid,
        mean_gp_eval_grid,
        std_gp_eval_grid,
        np.min(f_train),
        n_initial_training_points,
        i_iteration,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    print("Plotting acquisition function done.")

    ax1: Axes
    ax2: Axes
    ax1, ax2 = plot_output_gp(
        config,
        u_eval,
        u_eval_grid,
        u_train,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        n_initial_training_points,
        i_iteration,
    )

    plot_physical_boundary(problem, ax1)  # pyright: ignore[reportArgumentType]
    plot_physical_boundary(problem, ax2)  # pyright: ignore[reportArgumentType]
    print("Plotting output of GP done.")

    plt.show()


def plot_f(
    u_eval_grid: list[np.ndarray],
    f_eval_grid: np.ndarray,
    u_train: np.ndarray,
    n_initial_training_points: int,
    i_iteration: int,
    original: bool = False,
) -> Axes:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()

    base: float = 2.0
    contour_set: QuadContourSet = ax.contourf(
        u_eval_grid[0] * 1000,
        u_eval_grid[1],
        f_eval_grid,
        locator=LogLocator(base=base),
        cmap="summer",
    )

    color_bar: Colorbar = fig.colorbar(contour_set)
    color_bar.formatter = ScalarFormatter()
    color_bar.ax.set_ylabel("Objective")

    ax.scatter(
        u_train[:n_initial_training_points, 0] * 1000,
        u_train[:n_initial_training_points, 1],
        c="k",
        label="Initial training points",
    )

    ax.scatter(
        u_train[n_initial_training_points:, 0] * 1000,
        u_train[n_initial_training_points:, 1],
        c="r",
        label="Training points found with BO",
    )

    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Pressure / bar")
    ax.legend()

    if original:
        ax.set_title(f"Iteration {i_iteration}, original Problem")
    else:
        ax.set_title(f"Iteration {i_iteration}")

    fig.tight_layout()

    return ax


def plot_acq(
    config: Config,
    u_train: np.ndarray,
    u_eval_grid: list[np.ndarray],
    mean_gp_eval_grid: np.ndarray,
    std_gp_eval_grid: np.ndarray,
    incumbent: float,
    n_initial_training_points: int,
    i_iteration: int,
) -> Axes:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()

    label: str
    acq_eval_grid: np.ndarray

    # Lower confidence bound
    if config.formulation_acq.startswith("lcb"):
        # We want to minimize the lower confidence bound.
        acq_eval_grid = mean_gp_eval_grid - config.factor_lcb_std * std_gp_eval_grid
        label = "Lower Confidence Bound"

    # Expected improvement
    else:
        # In the plot we show the negative expected improvement with values > 0.
        # This is important for the log scale contourf plot.
        # We want to maximize the shown expected improvement.
        acq_eval_grid = ei_gp(
            mean_gp_eval_grid,
            std_gp_eval_grid,
            incumbent,
            maximize=False,
        )  # pyright: ignore[reportCallIssue]
        acq_eval_grid = np.fmax(acq_eval_grid, 2.0e-10)
        label = "Expected Improvement"

    base: float = 2.0
    contour_set: QuadContourSet = ax.contourf(
        u_eval_grid[0] * 1000,
        u_eval_grid[1],
        acq_eval_grid,
        locator=LogLocator(base=base),
        cmap="summer",
    )

    color_bar: Colorbar = fig.colorbar(contour_set)
    color_bar.formatter = ScalarFormatter()
    color_bar.ax.set_ylabel(label)

    ax.scatter(
        u_train[:n_initial_training_points, 0] * 1000,
        u_train[:n_initial_training_points, 1],
        c="k",
        label="Initial training points",
    )

    ax.scatter(
        u_train[n_initial_training_points:, 0] * 1000,
        u_train[n_initial_training_points:, 1],
        c="r",
        label="Training points found with BO",
    )

    ax.set_xlabel("Temperature / K")
    ax.set_ylabel("Pressure / bar")
    ax.legend()
    ax.set_title(f"Iteration {i_iteration}")

    fig.tight_layout()

    return ax


def plot_output_gp(
    config: Config,
    u_eval: np.ndarray,
    u_eval_grid: list[np.ndarray],
    u_train: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    n_initial_training_points: int,
    i_iteration: int,
) -> tuple[Axes, Axes]:
    def plot_wrt_u(output: np.ndarray, label: str) -> Axes:
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        contour_set: QuadContourSet = ax.contourf(
            u_eval_grid[0] * 1000,
            u_eval_grid[1],
            output,
            levels=20,
            cmap="summer",
        )

        color_bar: Colorbar = fig.colorbar(contour_set)
        color_bar.formatter = ScalarFormatter()
        color_bar.ax.set_ylabel(label)

        ax.scatter(
            u_train[:n_initial_training_points, 0] * 1000,
            u_train[:n_initial_training_points, 1],
            c="k",
            label="Initial training points",
        )

        ax.scatter(
            u_train[n_initial_training_points:, 0] * 1000,
            u_train[n_initial_training_points:, 1],
            c="r",
            label="Training points found with BO",
        )

        ax.set_xlabel("Temperature / K")
        ax.set_ylabel("Pressure / bar")
        ax.legend()
        ax.set_title(f"Iteration {i_iteration}")

        fig.tight_layout()

        return ax

    # %% Get data

    u_eval_transformed: np.ndarray = input_transformer_gp.transform(
        u_eval,
    )  # pyright: ignore[reportAssignmentType]

    mean_gp_eval_transformed: np.ndarray
    var_gp_eval_transformed: np.ndarray
    mean_gp_eval_transformed, var_gp_eval_transformed = gp.predict(
        u_eval_transformed,
    )  # pyright: ignore[reportAssignmentType]

    std_gp_eval_transformed: np.ndarray = np.sqrt(
        np.diag(var_gp_eval_transformed),
    )[:, np.newaxis]

    mean_gp_eval: np.ndarray = output_transformer_gp.inverse_transform(
        mean_gp_eval_transformed,
    )  # pyright: ignore[reportAssignmentType]
    std_gp_eval: np.ndarray = (
        std_gp_eval_transformed / output_transformer_gp.slope.item()
    )

    mean_gp_eval_grid: np.ndarray = mean_gp_eval.reshape(
        config.n_eval_points,
    )
    std_gp_eval_grid: np.ndarray = std_gp_eval.reshape(
        config.n_eval_points,
    )

    ax1: Axes = plot_wrt_u(mean_gp_eval_grid, "Modeled GP output: mean")
    ax2: Axes = plot_wrt_u(std_gp_eval_grid, "Modeled GP output: std")

    return ax1, ax2
