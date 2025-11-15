import numpy as np

from hybrid_bo import Config, Problem


class ResultsBO:
    def __init__(self, problem: Problem, config: Config) -> None:
        self.f: np.ndarray = np.full(
            (config.n_runs_bo, config.n_iterations_bo, 1),
            np.nan,
        )

        self.f_no_noise: np.ndarray = np.full(
            (config.n_runs_bo, config.n_iterations_bo, 1),
            np.nan,
        )

        self.u: np.ndarray = np.full(
            (config.n_runs_bo, config.n_iterations_bo, problem.n_u),
            np.nan,
        )

        self.x: np.ndarray = np.full(
            (config.n_runs_bo, config.n_iterations_bo, problem.n_x),
            np.nan,
        )

        self.x_no_noise: np.ndarray = np.full(
            (config.n_runs_bo, config.n_iterations_bo, problem.n_x),
            np.nan,
        )
