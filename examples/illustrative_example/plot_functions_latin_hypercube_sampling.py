from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# plt.rcParams["text.usetex"] = True


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f_bo: np.ndarray,
    u_bo: np.ndarray,
    x_bo: np.ndarray,
    n_points_initial: int,
    i_iteration: int,
) -> None:

    # %% User definitions

    original: bool = False
    confidence_level: int = 90
    n_points: list[int] = [101]

    # %% Get evaluation points

    if len(n_points) != problem.n_u:
        msg: str = "len(n_points) != problem.n_u"
        raise Exception(msg)

    n_points_total: np.int64 = np.prod(n_points)

    u_components: list[np.ndarray] = []
    for i_component in range(problem.n_u):
        u_component: np.ndarray = np.linspace(
            problem.u_lower_bounds_acquisition[i_component],
            problem.u_upper_bounds_acquisition[i_component],
            n_points[i_component],
        )
        u_components.append(u_component)

    u_grid: list[np.ndarray] = np.meshgrid(*u_components, indexing="ij")  # pyright: ignore[reportAssignmentType]

    # np.stack():
    #   shape (n_points[0], n_points[1],..., problem.n_u)
    #   Every vector spanning along the last axis corresponds to one evaluation point
    # reshape(): concatenates the evaluation points row-wise
    u: np.ndarray = np.stack(u_grid, axis=-1).reshape(
        (n_points_total, problem.n_u),
    )

    x: np.ndarray
    f: np.ndarray
    x, f = get_evaluation_data(
        problem,
        config,
        u,
        original,
    )

    f_grid: np.ndarray = f.reshape(n_points)
    x_grid: list[np.ndarray] = [
        x[:, i_component].reshape(n_points) for i_component in range(problem.n_x)
    ]

    # %% Get GP mean and std

    mean_gp: np.ndarray
    std_gp: np.ndarray
    mean_gp, std_gp = get_gp_mean_and_std(
        u,
        gp,
        input_transformer_gp,
        output_transformer_gp,
    )

    mean_gp_grid: np.ndarray = mean_gp.reshape(n_points)
    std_gp_grid: np.ndarray = std_gp.reshape(n_points)

    # %% Plotting

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()

    plot_gp(
        ax,
        u,
        f,
        mean_gp,
        std_gp,
        u_bo,
        f_bo,
        n_points_initial,
        confidence_level,
    )

    ax.legend(
        ncols=2,
        loc="lower right",
        mode="expand",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        borderaxespad=0,
    )

    ax.set_xlabel("u")

    fig.tight_layout()

    plt.show()


def plot_gp(
    ax: Axes,
    u: np.ndarray,
    f: np.ndarray,
    mean_gp: np.ndarray,
    std_gp: np.ndarray,
    u_bo: np.ndarray,
    f_bo: np.ndarray,
    n_points_initial: int,
    confidence_level: int,
) -> None:

    # %% Get confidence factor

    confidence_factors = {
        25: 0.32,
        50: 0.67,
        68: 0.99,
        80: 1.28,
        90: 1.64,
        95: 1.96,
        99: 2.58,
    }
    if confidence_level not in confidence_factors:
        msg: str = "For 'confidence_level', only the values 25, 50, 68, 80, 90, 95 and 99 are allowed."
        raise Exception(msg)

    confidence_factor: float = confidence_factors[confidence_level]

    output_gp_lower_confidence_bound: np.ndarray = (
        mean_gp - confidence_factor * std_gp
    ).flatten()
    output_gp_upper_confidence_bound: np.ndarray = (
        mean_gp + confidence_factor * std_gp
    ).flatten()

    # %% Plotting

    output_gp: np.ndarray = f.flatten()

    ax.scatter(
        u_bo[:n_points_initial, 0],
        f_bo[:n_points_initial, 0],
        c="k",
        label="Initial points",
    )

    ax.scatter(
        u_bo[n_points_initial:, 0],
        f_bo[n_points_initial:, 0],
        c="r",
        label="Points found with BO",
    )

    ax.plot(u.flatten(), output_gp, color="k", linestyle="--", label="True")
    ax.plot(u.flatten(), mean_gp, color="b", label="Predicted mean")
    ax.fill_between(
        u.flatten(),
        output_gp_lower_confidence_bound,
        output_gp_upper_confidence_bound,
        alpha=0.2,
        color="b",
        label="Confidence interval",
    )

    ax.set_ylabel("Objective")


def get_evaluation_data(
    problem: Problem,
    config: Config,
    u: np.ndarray,
    original: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    f: np.ndarray
    x: np.ndarray

    f, x = problem.evaluate_with_simulation(
        u,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.rng,
        original,
    )

    return x, f


def get_gp_mean_and_std(
    u: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
) -> tuple[np.ndarray, np.ndarray]:

    input_gp: np.ndarray = u

    input_gp_transformed: np.ndarray = input_transformer_gp.transform(
        input_gp,
    )

    mean_gp_transformed: np.ndarray
    cov_gp_transformed: np.ndarray
    mean_gp_transformed, cov_gp_transformed = gp.predict(
        input_gp_transformed,
    )

    std_gp_transformed: np.ndarray = np.sqrt(np.diag(cov_gp_transformed))[:, np.newaxis]

    mean_gp: np.ndarray = output_transformer_gp.inverse_transform(
        mean_gp_transformed,
    )
    std_gp: np.ndarray = std_gp_transformed / output_transformer_gp.slope.item()

    return mean_gp, std_gp
