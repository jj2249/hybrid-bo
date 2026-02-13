import copy
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
from hybrid_bo.optimizers import MultiStartOptimizer, OptimizationResult, Optimizer
from hybrid_bo.problem import Problem
from hybrid_bo.results import ResultsBO, ResultsGP
from hybrid_bo.routines.utils import ei_gp, get_sampling_based_data
from hybrid_bo.type_aliases import SymbolicType


def do_standard_bo(
    problem: Problem,
    config: Config,
    gp: GP,
    u_initial_complete: list[np.ndarray],
    optimizer: Optimizer,
    results_dir: Path,
    create_plots: Callable | None = None,
) -> None:
    # %% Data for all BO runs

    if len(u_initial_complete) != config.n_runs_bo:
        msg: str = "len(u_initial_complete) != config.n_runs_bo"
        raise Exception(msg)

    results_bo_complete: list[ResultsBO] = []
    results_gp_complete: list[ResultsGP] = []

    n_points_initial_complete: list[int] = [
        u_initial.shape[0] for u_initial in u_initial_complete
    ]

    n_points_complete: list[int] = [
        (n_points_initial + config.n_iterations_bo)
        for n_points_initial in n_points_initial_complete
    ]

    measurement_noise_complete: list[np.ndarray]
    measurement_noise_complete, _ = get_sampling_based_data(config, n_points_complete)

    # %% Iterate over multiple BO runs

    for i_run_bo in range(config.n_runs_bo):
        # %% Data for current run

        # Initial points AND BO iteration points
        n_points: int = n_points_complete[i_run_bo]
        measurement_noise: np.ndarray = measurement_noise_complete[i_run_bo]

        u: np.ndarray = np.full((n_points, config.n_u), np.nan)
        x: np.ndarray = np.full((n_points, config.n_x), np.nan)
        x_no_noise: np.ndarray = np.full((n_points, config.n_x), np.nan)
        f: np.ndarray = np.full((n_points, 1), np.nan)
        f_no_noise: np.ndarray = np.full((n_points, 1), np.nan)
        incumbents: np.ndarray = np.full((n_points, 1), np.nan)
        incumbents_no_noise: np.ndarray = np.full((n_points, 1), np.nan)

        # Initial points
        n_points_initial: int = n_points_initial_complete[i_run_bo]
        measurement_noise_initial: np.ndarray = measurement_noise[:n_points_initial, :]
        u_initial: np.ndarray = u_initial_complete[i_run_bo]

        x_initial: np.ndarray
        x_no_noise_initial: np.ndarray
        f_initial: np.ndarray
        f_no_noise_initial: np.ndarray
        incumbents_initial: np.ndarray
        incumbents_no_noise_initial: np.ndarray

        # Incumbents
        incumbent: float
        incumbent_no_noise: float

        # Results
        results_bo: ResultsBO | None = None
        results_gp: ResultsGP | None = None

        # %% Get initial points

        f_initial, x_initial, f_no_noise_initial, x_no_noise_initial = (
            problem.evaluate_with_noisy_simulation(
                u_initial,
                measurement_noise_initial,
                config.indices_x_measured,
                config.n_starts_max_evaluate_problem,
                config.use_jacobian_evaluate_problem,
                config.rng,
            )
        )

        incumbents_initial = np.minimum.accumulate(f_initial, 0)
        incumbents_no_noise_initial = np.minimum.accumulate(
            f_no_noise_initial,
            0,
        )

        incumbent = incumbents_initial[-1].item()
        incumbent_no_noise = incumbents_no_noise_initial[-1].item()

        # %% Fill points with initial values

        u[:n_points_initial, :] = u_initial
        x[:n_points_initial, :] = x_initial
        x_no_noise[:n_points_initial, :] = x_no_noise_initial
        f[:n_points_initial, :] = f_initial
        f_no_noise[:n_points_initial, :] = f_no_noise_initial
        incumbents[:n_points_initial, :] = incumbents_initial
        incumbents_no_noise[:n_points_initial, :] = incumbents_no_noise_initial

        print("Initialization done.")

        # %% Define GP

        input_transformer_gp: AffineTransformer = MinMaxTransformer()
        output_transformer_gp: AffineTransformer = StandardTransformer()

        set_gp_training_data(
            gp,
            u_initial,
            f_initial,
            input_transformer_gp,
            output_transformer_gp,
        )

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

        if config.save_extended_results:
            results_gp = ResultsGP(
                copy.deepcopy(input_transformer_gp),
                copy.deepcopy(output_transformer_gp),
                copy.deepcopy(gp),
            )

        print("Training GP done.")

        # %% Create initial plots

        if config.create_plots and create_plots:
            create_plots(
                problem,
                config,
                gp,
                input_transformer_gp,
                output_transformer_gp,
                f_initial,
                u_initial,
                x_initial,
                n_points_initial,
                0,
            )

            print("Plotting done.")

        print(f"\nBO iteration 0/{config.n_iterations_bo} done.\n\n---\n")

        # %% BO loop

        for i_bo in range(config.n_iterations_bo):
            try:
                # %% Get next point with acquisition function problem

                n_variables: int
                f_function: cas.Function
                lower_bounds: np.ndarray
                upper_bounds: np.ndarray
                n_variables, f_function, lower_bounds, upper_bounds = setup_acq_problem(
                    config,
                    problem,
                    gp,
                    input_transformer_gp,
                    output_transformer_gp,
                    incumbent,
                )

                print("Setting up acquisition problem done.")

                optimizer.set_problem(
                    n_variables, f_function, None, None, lower_bounds, upper_bounds
                )

                if isinstance(optimizer, MultiStartOptimizer):
                    optimizer.lower_bounds_starting_points = (
                        problem.u_lower_bounds_starting_points
                    )
                    optimizer.upper_bounds_starting_points = (
                        problem.u_upper_bounds_starting_points
                    )

                    starting_points: np.ndarray = optimizer.create_lhs_samples(
                        config.n_starts_acq_optimization,
                        config.rng,
                    )
                    optimizer.X0 = starting_points

                solution: OptimizationResult | None = optimizer.solve()

                if solution is None:
                    msg: str = "solution is None"
                    raise Exception(msg)

                print("Solving acquisition problem done.")

                u_next: np.ndarray = solution.x[: problem.n_u, :]  # pyright: ignore[reportIndexIssue]

                x_next: np.ndarray
                x_no_noise_next: np.ndarray
                f_next: float
                f_no_noise_next: float

                f_next, x_next, f_no_noise_next, x_no_noise_next = (
                    problem.single_evaluate_with_noisy_simulation(
                        u_next,
                        measurement_noise[[n_points_initial + i_bo], :].T,
                        config.indices_x_measured,
                        config.n_starts_max_evaluate_problem,
                        config.use_jacobian_evaluate_problem,
                        config.rng,
                    )
                )

                incumbent = min(incumbent, f_next)
                incumbent_no_noise = min(incumbent_no_noise, f_no_noise_next)

                # %% Fill points with next value

                u[[n_points_initial + i_bo], :] = u_next.T
                x[[n_points_initial + i_bo], :] = x_next.T
                x_no_noise[[n_points_initial + i_bo], :] = x_no_noise_next.T
                f[[n_points_initial + i_bo], :] = f_next
                f_no_noise[[n_points_initial + i_bo], :] = f_no_noise_next
                incumbents[[n_points_initial + i_bo], :] = incumbent
                incumbents_no_noise[[n_points_initial + i_bo], :] = incumbent_no_noise

                print("Getting next point done.")

                # %% Update GP

                set_gp_training_data(
                    gp,
                    u[: n_points_initial + i_bo + 1, :],
                    f[: n_points_initial + i_bo + 1, :],
                    input_transformer_gp,
                    output_transformer_gp,
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

                if results_gp is not None:
                    results_gp.input_transformers_bo.append(
                        copy.deepcopy(input_transformer_gp),
                    )
                    results_gp.output_transformers_bo.append(
                        copy.deepcopy(
                            output_transformer_gp,
                        ),
                    )
                    results_gp.gps_bo.append(copy.deepcopy(gp))

                print("Training GP done.")

                # %% Create plots

                if config.create_plots and create_plots:
                    create_plots(
                        problem,
                        config,
                        gp,
                        input_transformer_gp,
                        output_transformer_gp,
                        f[: n_points_initial + i_bo + 1, :],
                        u[: n_points_initial + i_bo + 1, :],
                        x[: n_points_initial + i_bo + 1, :],
                        n_points_initial,
                        i_bo + 1,
                    )

                    print("Plotting done.")

                print(
                    f"\nBO iteration {i_bo + 1}/{config.n_iterations_bo} done.\n\n---\n",
                )

            except Exception as e:
                traceback.print_exc()

            finally:
                if config.save_bo_results:
                    results_bo = ResultsBO(
                        u_initial,
                        x_initial,
                        f_initial,
                        incumbents_initial,
                        x_no_noise_initial,
                        f_no_noise_initial,
                        incumbents_no_noise_initial,
                        u[n_points_initial:, :],
                        x[n_points_initial:, :],
                        f[n_points_initial:, :],
                        incumbents[n_points_initial:, :],
                        x_no_noise[n_points_initial:, :],
                        f_no_noise[n_points_initial:, :],
                        incumbents_no_noise[n_points_initial:, :],
                    )

                    results_bo_complete.append(results_bo)

                if results_gp is not None:
                    results_gp_complete.append(results_gp)

        print(f"BO run {i_run_bo + 1}/{config.n_runs_bo} done.\n\n------\n")

    # %% Save results

    if config.save_bo_results:
        path: Path = results_dir / "results_bo.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(results_bo_complete, file, protocol=pickle.HIGHEST_PROTOCOL)

    if config.save_extended_results:
        path: Path = results_dir / "measurement_noise.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(
                    measurement_noise_complete,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

        path: Path = results_dir / "results_gp.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(
                    results_gp_complete,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )


def set_gp_training_data(
    gp: GP,
    u: np.ndarray,
    f: np.ndarray,
    input_gp_transformer: AffineTransformer,
    output_gp_transformer: AffineTransformer,
) -> None:
    u_transformed: np.ndarray = input_gp_transformer.fit_transform(
        u,
    )

    f_transformed: np.ndarray = output_gp_transformer.fit_transform(
        f,
    )

    gp.set_XY_train(u_transformed, f_transformed)


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

    lower_bounds: np.ndarray = problem.u_lower_bounds_acquisition
    upper_bounds: np.ndarray = problem.u_upper_bounds_acquisition

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
    else:
        f_expression = -ei_gp(mean, std, incumbent, False)

    f: cas.Function = cas.Function("f", [w], [f_expression])

    return n_variables, f, lower_bounds, upper_bounds
