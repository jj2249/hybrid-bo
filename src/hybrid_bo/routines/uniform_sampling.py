import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np

from hybrid_bo.affine_transformers import (
    AffineTransformer,
    MinMaxTransformer,
    StandardTransformer,
)
from hybrid_bo.config import Config
from hybrid_bo.gp import GP
from hybrid_bo.optimizers import MultiStartOptimizer
from hybrid_bo.problem import Problem
from hybrid_bo.results_bo import ResultsBO
from hybrid_bo.routines.utils import get_sampling_based_data, get_training_data


def do_uniform_sampling(
    problem: Problem,
    config: Config,
    gp: GP,
    u_train_initial_complete: list[np.ndarray],
    results_dir: Path,
    create_plots: Callable | None = None,
) -> None:
    if len(u_train_initial_complete) != config.n_runs_bo:
        msg: str = "len(u_train) != config.n_runs_bo"
        raise Exception(msg)

    n_training_points_initial_complete: list[int] = [
        u_train_initial.shape[0] for u_train_initial in u_train_initial_complete
    ]

    n_training_points_complete: list[int] = [
        (n_training_points_initial + config.n_iterations_bo)
        for n_training_points_initial in n_training_points_initial_complete
    ]

    measurement_noise_train_complete: list[np.ndarray]
    gaussian_standard_samples_complete: list[np.ndarray]
    measurement_noise_train_complete, gaussian_standard_samples_complete = (
        get_sampling_based_data(config, n_training_points_complete)
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
            gp.optimizer.lower_bounds_starting_points = gp.optimizer.lower_bounds.copy()  # pyright: ignore[reportOptionalMemberAccess]
            gp.optimizer.upper_bounds_starting_points = gp.optimizer.upper_bounds.copy()  # pyright: ignore[reportOptionalMemberAccess]

            starting_points: np.ndarray = gp.optimizer.create_lhs_samples(
                config.n_starts_training_gp,
                config.rng,
            )
            gp.optimizer.X0 = starting_points

        gp.solve_training_problem()

        print("Training GP done.")

        # %% Get initial incumbent

        incumbent: float = np.min(f_train)

        # %% Create initial plots

        if config.create_plots and create_plots:
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

        # %% Get additional points of u by uniform sampling

        u_additional: np.ndarray = config.rng.uniform(
            size=(config.n_iterations_bo, problem.n_u),
        )

        u_additional: np.ndarray = (
            problem.u_lower_bounds_starting_points.T
            + (
                problem.u_upper_bounds_starting_points.T
                - problem.u_lower_bounds_starting_points.T
            )
            * u_additional
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
                        config.rng,
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
                    gp, u_train, f_train, input_transformer_gp, output_transformer_gp
                )

                gp.create_training_problem()

                if isinstance(gp.optimizer, MultiStartOptimizer):
                    gp.optimizer.lower_bounds_starting_points = (
                        gp.optimizer.lower_bounds.copy()  # pyright: ignore[reportOptionalMemberAccess]
                    )
                    gp.optimizer.upper_bounds_starting_points = (
                        gp.optimizer.upper_bounds.copy()  # pyright: ignore[reportOptionalMemberAccess]
                    )

                    starting_points: np.ndarray = gp.optimizer.create_lhs_samples(
                        config.n_starts_training_gp,
                        config.rng,
                    )
                    gp.optimizer.X0 = starting_points

                gp.solve_training_problem()

                print("Training GP done.")

                # %% Get new incumbent

                if f_next < incumbent:
                    incumbent: float = f_next

                # %% Create plots

                if config.create_plots and create_plots:
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
                results_bo.incumbent[i_run_bo, i_bo] = incumbent
                results_bo.u[i_run_bo, i_bo] = u_next.flatten()
                results_bo.x[i_run_bo, i_bo] = x_next.flatten()
                results_bo.x_no_noise[i_run_bo, i_bo] = x_next_no_noise.flatten()

                print(f"BO iteration {i_bo + 1} done.\n---\n")

            except Exception as e:
                print(e)

        print(f"BO run {i_run_bo + 1} done.\n\n---\n\n")

    # %% Save results of BO

    if config.save_results:
        path: Path = results_dir / "results_uniform_sampling.pkl"
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
