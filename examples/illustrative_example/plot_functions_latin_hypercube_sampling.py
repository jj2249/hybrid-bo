import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import Config, Problem

# plt.rcParams["text.usetex"] = True


def create_plots(
    problem: Problem,
    config: Config,
    f_bo: np.ndarray,
    u_bo: np.ndarray,
    x_bo: np.ndarray,
    n_points_initial: int,
    i_iteration: int,
) -> None:

    # %% User definitions

    original = False
    n_points = [101]

    # %% Get evaluation points

    if len(n_points) != problem.n_u:
        msg = "len(n_points) != problem.n_u"
        raise Exception(msg)

    n_points_total = np.prod(n_points)

    u_components = []
    for i_component in range(problem.n_u):
        u_component = np.linspace(
            problem.u_bounds.lower[i_component],
            problem.u_bounds.upper[i_component],
            n_points[i_component],
        )
        u_components.append(u_component)

    u_grid = np.meshgrid(*u_components, indexing="ij")

    # np.stack():
    #   shape (n_points[0], n_points[1],..., problem.n_u)
    #   Every vector spanning along the last axis corresponds to one evaluation point
    # reshape(): concatenates the evaluation points row-wise
    u = np.stack(u_grid, axis=-1).reshape(
        (n_points_total, problem.n_u),
    )

    _, f = get_evaluation_data(
        problem,
        config,
        u,
        original,
    )

    # %% Plotting

    fig, ax = plt.subplots()

    plot_samples(ax, u, f, u_bo, f_bo, n_points_initial)

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


def plot_samples(
    ax: Axes,
    u: np.ndarray,
    f: np.ndarray,
    u_bo: np.ndarray,
    f_bo: np.ndarray,
    n_points_initial: int,
) -> None:
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
        label="Sampled points",
    )

    ax.plot(u.flatten(), f.flatten(), color="k", linestyle="--", label="True")

    ax.set_ylabel("Objective")


def get_evaluation_data(
    problem: Problem,
    config: Config,
    u: np.ndarray,
    original: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    f, x = problem.evaluate_with_simulation(
        u,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.rng,
        original,
    )

    return x, f
