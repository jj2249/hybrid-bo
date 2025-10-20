import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy

from ..affine_transformers import (
    AffineTransformer,
    MinMaxTransformer,
    StandardTransformer,
)
from ..config import Config
from ..gp import GP
from ..optimizers import MultiStartOptimizer
from ..problem import Problem
from ..results_bo import ResultsBO
from .utils import get_sampling_based_data, get_training_data


def latin_hypercube_sampling(
    problem: Problem,
    config: Config,
    gp: GP,
    u_train_initial_complete: list[np.ndarray],
    results_dir: Path,
    create_plots: Callable,
) -> None:
    # %% Data to create for all BO runs

    if len(u_train_initial_complete) != config.n_runs_bo:
        msg: str = "len(u_train) != config.n_runs_bo"
        raise Exception(msg)

    n_training_points_initial_complete: np.ndarray = np.array(
        [u_train_initial.shape[0] for u_train_initial in u_train_initial_complete],
        int,
    )
    n_training_points_complete: np.ndarray = (
        n_training_points_initial_complete + config.n_iterations_bo
    )

    measurement_noise_train_complete: list[np.ndarray]
    measurement_noise_train_complete, _ = get_sampling_based_data(
        config,
        n_training_points_complete,
    )

    # %% Results of BO

    results_bo: ResultsBO = ResultsBO(problem, config)

    # %% Iterate over multiple BO runs

    for i_run_bo in range(config.n_runs_bo):
        # %% Training data

        u_train: np.ndarray = u_train_initial_complete[i_run_bo].copy()

        # Contains the measurement noise for all training points (initial points and BO iterations)
        measurement_noise_train: np.ndarray = measurement_noise_train_complete[i_run_bo]

        f_train: np.ndarray
        x_train: np.ndarray
        f_train_no_noise: np.ndarray
        x_train_no_noise: np.ndarray

        f_train, x_train, f_train_no_noise, x_train_no_noise = get_training_data(
            u_train,
            measurement_noise_train,
            problem,
            config,
        )

        n_training_points_initial: int = n_training_points_initial_complete[i_run_bo]

        # %% Define GP

        input_transformer_gp: AffineTransformer = MinMaxTransformer()
        output_transformer_gp: AffineTransformer = StandardTransformer()

        set_gp_XY(gp, u_train, f_train, input_transformer_gp, output_transformer_gp)

        print("Preparation done.")

        # %% Train GP

        gp.create_training_problem()

        if isinstance(gp.optimizer, MultiStartOptimizer):
            starting_points: np.ndarray = gp.optimizer.create_lhs_samples(
                n_starts=config.n_starts_training_gp,
                seed=config.seed,
            )
            gp.optimizer.set_X0(starting_points)

        gp.solve_training_problem()

        print("Training GP done.")

        # %% Get initial incumbent

        incumbent: float = np.min(f_train)

        # %% Create initial plots

        if config.create_plots:
            create_plots(
                problem,
                config,
                gp,
                input_transformer_gp,
                output_transformer_gp,
                f_train,
                u_train,
                x_train,
                n_training_points_initial,
                0,
            )

            print("Plotting done.")

        print("Initialization done.\n---\n")

        # %% BO loop

        # %% Get additional points of u by latin hypercube sampling

        sampler: scipy.stats.qmc.LatinHypercube = scipy.stats.qmc.LatinHypercube(
            problem.n_u,
            rng=config.rng,
        )

        u_additional: np.ndarray = sampler.random(config.n_iterations_bo)
        u_additional: np.ndarray = (
            problem.u_lower_bounds.T
            + (problem.u_upper_bounds.T - problem.u_lower_bounds.T) * u_additional
        )

        # %% Actual loop

        for i_bo in range(config.n_iterations_bo):
            try:
                # %% Get u_next, x_next, f_next

                u_next: np.ndarray = u_additional[[i_bo], :].T

                f_next: float
                x_next: np.ndarray
                f_next_no_noise: float
                x_next_no_noise: np.ndarray

                f_next, x_next, f_next_no_noise, x_next_no_noise = (
                    problem.evaluate_with_noisy_simulation(
                        u_next,
                        measurement_noise_train[[n_training_points_initial + i_bo]].T,
                        config.indices_x_measured,
                        config.n_starts_max_evaluate_problem,
                        config.use_jacobian_evaluate_problem,
                        config.check_bounds_evaluate_problem,
                        config.seed,
                    )
                )

                print("Getting next training data done.")

                # %% Update GP

                f_train = np.vstack((f_train, f_next))
                f_train_no_noise = np.vstack((f_train_no_noise, f_next_no_noise))

                u_train = np.vstack((u_train, u_next.T))

                x_train = np.vstack((x_train, x_next.T))
                x_train_no_noise = np.vstack((x_train_no_noise, x_next_no_noise.T))

                set_gp_XY(
                    gp,
                    u_train,
                    f_train,
                    input_transformer_gp,
                    output_transformer_gp,
                )

                gp.create_training_problem()

                if isinstance(gp.optimizer, MultiStartOptimizer):
                    starting_points: np.ndarray = gp.optimizer.create_lhs_samples(
                        n_starts=config.n_starts_training_gp,
                        seed=config.seed,
                    )
                    gp.optimizer.set_X0(starting_points)

                gp.solve_training_problem()

                print("Training GP done.")

                # %% Get new incumbent

                if f_next < incumbent:
                    incumbent: float = f_next

                # %% Create plots

                if config.create_plots:
                    create_plots(
                        problem,
                        config,
                        gp,
                        input_transformer_gp,
                        output_transformer_gp,
                        f_train,
                        u_train,
                        x_train,
                        n_training_points_initial,
                        i_bo + 1,
                    )

                    print("Plotting done.")

                # %% Fill results of BO

                results_bo.f[i_run_bo, i_bo] = f_next
                results_bo.f_no_noise[i_run_bo, i_bo] = f_next_no_noise
                results_bo.u[i_run_bo, i_bo] = u_next.flatten()
                results_bo.x[i_run_bo, i_bo] = x_next.flatten()
                results_bo.x_no_noise[i_run_bo, i_bo] = x_next_no_noise.flatten()

                print(f"BO iteration {i_bo + 1} done.\n---\n")

            except Exception as e:
                print(e)

        print(f"BO run {i_run_bo + 1} done.\n\n---\n\n")

    # %% Save results of BO

    if config.save_results:
        path: Path = results_dir / "results_latin_hypercube_sampling.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(results_bo, file, protocol=pickle.HIGHEST_PROTOCOL)


def set_gp_XY(
    gp: GP,
    u_train: np.ndarray,
    f_train: np.ndarray,
    input_gp_transformer: AffineTransformer,
    output_gp_transformer: AffineTransformer,
) -> None:
    u_train_transformed: np.ndarray = input_gp_transformer.fit_transform(
        u_train,
    )

    f_train_transformed: np.ndarray = output_gp_transformer.fit_transform(
        f_train,
    )

    gp.set_XY_train(u_train_transformed, f_train_transformed)
