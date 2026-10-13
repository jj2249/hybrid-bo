import numpy as np

from hybrid_bo import Config, Problem

from .plot_functions_latin_hypercube_sampling import (
    create_plots as latin_hypercube_sampling_create_plots,
)

# plt.rcParams["text.usetex"] = True


def create_plots(
    problem: Problem,
    config: Config,
    f: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    n_points_initial: int,
    i_iteration: int,
) -> None:
    latin_hypercube_sampling_create_plots(
        problem,
        config,
        f,
        u,
        x,
        n_points_initial,
        i_iteration,
    )
