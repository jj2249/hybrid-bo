import pickle
import traceback
from collections.abc import Callable
from pathlib import Path

import casadi as cas
import numpy as np

from hybrid_bo.affine_transformers import (
    AffineTransformer,
    MinMaxTransformer,
    StandardTransformer,
)
from hybrid_bo.config import Config
from hybrid_bo.gp import GP
from hybrid_bo.optimizers import MultiStartOptimizer, Optimizer
from hybrid_bo.problem import Problem
from hybrid_bo.results_bo import ResultsBO
from hybrid_bo.routines.utils import ei_gp, get_sampling_based_data, get_training_data
from hybrid_bo.type_aliases import SymbolicType


def do_standard_bo(
    problem: Problem,
    config: Config,
    gp: GP,
    u_train_initial_complete: list[np.ndarray],
    optimizer: Optimizer,
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

        # %% Actual loop

        for i_bo in range(config.n_iterations_bo):
            try:
                # %% Get u_next, x_next, f_next with acquisition function problem

                n_variables: int
                f: cas.Function
                lower_bounds: np.ndarray
                upper_bounds: np.ndarray
                n_variables, f, lower_bounds, upper_bounds = setup_acq_problem(
                    config,
                    problem,
                    gp,
                    input_transformer_gp,
                    output_transformer_gp,
                    incumbent,
                )

                print("Setting up acquisition problem done.")

                optimizer.set_problem(
                    n_variables, f, None, None, lower_bounds, upper_bounds
                )

                if isinstance(optimizer, MultiStartOptimizer):
                    starting_points: np.ndarray = optimizer.create_lhs_samples(
                        n_starts=config.n_starts_acq_optimization,
                        seed=config.seed,
                    )
                    optimizer.set_X0(starting_points)

                solution: dict[str, float | np.ndarray] | None = optimizer.solve()

                if solution is None:
                    msg: str = "solution is None"
                    raise Exception(msg)

                print("Solving acquisition problem done.")

                u_next: np.ndarray = solution["x"][0 : problem.n_u]  # pyright: ignore[reportIndexIssue]

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
                    gp, u_train, f_train, input_transformer_gp, output_transformer_gp
                )

                gp.create_training_problem()

                if isinstance(gp.optimizer, MultiStartOptimizer):
                    starting_points: np.ndarray = gp.optimizer.create_lhs_samples(
                        n_starts=config.n_starts_training_gp, seed=config.seed
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
                traceback.print_exc()

        print(f"BO run {i_run_bo + 1} done.\n\n---\n\n")

    # %% Save results of BO

    if config.save_results:
        path: Path = results_dir / "results_standard_bo.pkl"
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


def setup_acq_problem(
    config: Config,
    problem: Problem,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    incumbent: float,
) -> tuple[int, cas.Function, np.ndarray, np.ndarray]:
    # n_variables, f, lower_bounds, upper_bounds

    u: SymbolicType = SymbolicType.sym("u", problem.n_u, 1)  # pyright: ignore[reportArgumentType]
    u_transformed: SymbolicType = input_transformer_gp.transform(u.T).T  # pyright: ignore[reportAssignmentType]
    n_variables: int = problem.n_u
    w: SymbolicType = u

    lower_bounds: np.ndarray = problem.u_lower_bounds
    upper_bounds: np.ndarray = problem.u_upper_bounds

    mean_transformed: SymbolicType
    var_transformed: SymbolicType
    mean_transformed, var_transformed = gp.predict(u_transformed.T)  # pyright: ignore[reportAssignmentType]

    mean: SymbolicType = output_transformer_gp.inverse_transform(mean_transformed)  # pyright: ignore[reportAssignmentType]
    std: SymbolicType = cas.sqrt(var_transformed) / output_transformer_gp.slope

    f_expression: SymbolicType

    # Lower confidence bound
    if config.formulation_acq.startswith("lcb"):
        f_expression = mean - config.factor_lcb_std * std

    # Expected improvement
    # Negative sign because max(a, b) = - min(-a, -b)
    else:
        f_expression = -ei_gp(mean, std, incumbent, False)

    f: cas.Function = cas.Function("f", [w], [f_expression])

    return n_variables, f, lower_bounds, upper_bounds
