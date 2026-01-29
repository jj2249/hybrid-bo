from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import LogLocator, ScalarFormatter

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.routines.utils import get_evaluation_data, get_f_eval_samples

from .custom_problem import plot_physical_boundary

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar
    from matplotlib.contour import QuadContourSet
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
    gaussian_standard_samples: np.ndarray,
    n_initial_training_points: int,
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
        config.n_eval_points,
        False,
        config.seed,
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
        config.n_eval_points,
        True,
        config.seed,
    )

    # %% Get samples of f at evaluation points

    f_eval_samples: np.ndarray = get_f_eval_samples(
        problem,
        config,
        u_eval,
        x_eval,
        gaussian_standard_samples,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        False,
    )

    f_eval_samples_original: np.ndarray = get_f_eval_samples(
        problem,
        config,
        u_eval,
        x_eval_original,
        gaussian_standard_samples,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        True,
    )

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
        f_eval_samples,
        np.min(f_train),
        n_initial_training_points,
        i_iteration,
        False,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    ax: Axes = plot_acq(
        config,
        u_train,
        u_eval_grid,
        f_eval_samples_original,
        np.min(f_train),
        n_initial_training_points,
        i_iteration,
        True,
    )

    plot_physical_boundary(problem, ax)  # pyright: ignore[reportArgumentType]

    print("Plotting acquisition function done.")

    ax1: Axes
    ax2: Axes
    ax3: Axes
    ax1, ax2, ax3 = plot_output_gp(
        config,
        u_eval,
        x_eval,
        u_eval_grid,
        u_train,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        n_initial_training_points,
        i_iteration,
        False,
    )

    plot_physical_boundary(problem, ax1)  # pyright: ignore[reportArgumentType]
    plot_physical_boundary(problem, ax2)  # pyright: ignore[reportArgumentType]
    plot_physical_boundary(problem, ax3)  # pyright: ignore[reportArgumentType]

    ax1: Axes
    ax2: Axes
    ax3: Axes
    ax1, ax2, ax3 = plot_output_gp(
        config,
        u_eval,
        x_eval_original,
        u_eval_grid,
        u_train,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        n_initial_training_points,
        i_iteration,
        True,
    )

    plot_physical_boundary(problem, ax1)  # pyright: ignore[reportArgumentType]
    plot_physical_boundary(problem, ax2)  # pyright: ignore[reportArgumentType]
    plot_physical_boundary(problem, ax3)  # pyright: ignore[reportArgumentType]

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
    f_eval_samples: np.ndarray,
    incumbent: float,
    n_initial_training_points: int,
    i_iteration: int,
    original: bool = False,
) -> Axes:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()

    f_eval_samples_grid: np.ndarray = f_eval_samples.T.reshape(
        [*config.n_eval_points, f_eval_samples.shape[0]],
    )

    label: str
    acq_eval_grid: np.ndarray

    # %% Lower confidence bound

    if config.formulation_acq.startswith("lcb"):
        f_eval_samples_grid_mean: np.ndarray = np.mean(f_eval_samples_grid, -1)
        f_eval_samples_grid_std: np.ndarray = np.std(f_eval_samples_grid, -1)

        # We want to minimize the lower confidence bound.
        acq_eval_grid = (
            f_eval_samples_grid_mean - config.factor_lcb_std * f_eval_samples_grid_std
        )
        label = "Lower Confidence Bound"

    # %% Expected improvement

    else:
        # In the plot we show the negative expected improvement with values > 0.
        # This is important for the log scale contourf plot.
        # We want to maximize the shown expected improvement.
        acq_eval_grid = -np.mean(np.fmin((f_eval_samples_grid - incumbent), -2e-10), -1)
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

    if original:
        ax.set_title(f"Iteration {i_iteration}, original Problem")
    else:
        ax.set_title(f"Iteration {i_iteration}")

    fig.tight_layout()

    return ax


def plot_output_gp(
    config: Config,
    u_eval: np.ndarray,
    x_eval: np.ndarray,
    u_eval_grid: list[np.ndarray],
    u_train: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    n_initial_training_points: int,
    i_iteration: int,
    original: bool = False,
) -> tuple[Axes, Axes, Axes]:
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

        if original:
            ax.set_title(f"Iteration {i_iteration}, original Problem")
        else:
            ax.set_title(f"Iteration {i_iteration}")

        fig.tight_layout()

        return ax

    # %% Get data

    input_gp_eval: np.ndarray = np.hstack(
        (
            u_eval[:, config.indices_u_input_gp],
            x_eval[:, config.indices_x_input_gp],
        ),
    )

    input_gp_eval_transformed: np.ndarray = input_transformer_gp.transform(
        input_gp_eval,
    )  # pyright: ignore[reportAssignmentType]

    output_gp_eval_true: np.ndarray = x_eval[:, config.indices_x_output_gp]
    output_gp_eval_true_grid: np.ndarray = output_gp_eval_true.reshape(
        config.n_eval_points,
    )

    output_gp_eval_mean_transformed: np.ndarray
    output_gp_eval_variance_transformed: np.ndarray
    output_gp_eval_mean_transformed, output_gp_eval_variance_transformed = gp.predict(
        input_gp_eval_transformed,
    )  # pyright: ignore[reportAssignmentType]

    output_gp_eval_std_transformed: np.ndarray = np.sqrt(
        np.diag(output_gp_eval_variance_transformed),
    )[:, np.newaxis]

    output_gp_eval_mean: np.ndarray = output_transformer_gp.inverse_transform(
        output_gp_eval_mean_transformed,
    )  # pyright: ignore[reportAssignmentType]
    output_gp_eval_std: np.ndarray = (
        output_gp_eval_std_transformed / output_transformer_gp.slope.item()
    )

    output_gp_eval_mean_grid: np.ndarray = output_gp_eval_mean.reshape(
        config.n_eval_points,
    )
    output_gp_eval_std_grid: np.ndarray = output_gp_eval_std.reshape(
        config.n_eval_points,
    )

    ax1: Axes = plot_wrt_u(output_gp_eval_true_grid, "True GP output")
    ax2: Axes = plot_wrt_u(output_gp_eval_mean_grid, "Modeled GP output: mean")
    ax3: Axes = plot_wrt_u(output_gp_eval_std_grid, "Modeled GP output: std")

    return ax1, ax2, ax3
