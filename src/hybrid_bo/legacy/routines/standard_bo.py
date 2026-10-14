import copy
import pickle
import traceback
from collections.abc import Callable
from pathlib import Path

import casadi as cas
import numpy as np

from hybrid_bo.legacy.affine_transformers import (
    AffineTransformer,
    MinMaxTransformer,
    StandardTransformer,
)
from hybrid_bo.legacy.config import Config
from hybrid_bo.legacy.gp import GP
from hybrid_bo.legacy.optimizers import MultiStartOptimizer, Optimizer
from hybrid_bo.legacy.problem import Problem
from hybrid_bo.legacy.results import ResultsBO, ResultsGP
from hybrid_bo.legacy.routines.utils import ei_gp
from hybrid_bo.legacy.type_aliases import SymbolicType


def do_standard_bo(
    problem: Problem,
    config: Config,
    gp: GP,
    u_initial_complete: list[np.ndarray],
    rngs: list[np.random.Generator],
    optimizer: Optimizer,
    results_dir: Path,
    create_plots: Callable | None = None,
) -> None:
    # %% Data for all BO runs

    if len(u_initial_complete) != config.n_runs_bo:
        msg = "len(u_initial_complete) != config.n_runs_bo"
        raise Exception(msg)
    if len(rngs) != config.n_runs_bo:
        msg = "len(rngs) != config.n_runs_bo"
        raise Exception(msg)

    results_bo_complete = []
    results_gp_complete = []

    # %% Iterate over multiple BO runs

    for i_run_bo in range(config.n_runs_bo):
        # %% Data for current run

        # Each run draws only from its own generator: each evaluation (noise and fsolve
        # guesses), then per iteration GP-fit and acquisition starts
        rng = rngs[i_run_bo]

        # Initial points AND BO iteration points
        n_points = u_initial_complete[i_run_bo].shape[0] + config.n_iterations_bo

        u = np.full((n_points, config.n_u), np.nan)
        x = np.full((n_points, config.n_x), np.nan)
        x_no_noise = np.full((n_points, config.n_x), np.nan)
        f = np.full((n_points, 1), np.nan)
        f_no_noise = np.full((n_points, 1), np.nan)
        incumbents = np.full((n_points, 1), np.nan)
        incumbents_no_noise = np.full((n_points, 1), np.nan)

        # Initial points
        n_points_initial = u_initial_complete[i_run_bo].shape[0]
        u_initial = u_initial_complete[i_run_bo]

        x_initial = np.full((n_points, config.n_x), np.nan)
        x_no_noise_initial = np.full((n_points, config.n_x), np.nan)
        f_initial = np.full((n_points, 1), np.nan)
        f_no_noise_initial = np.full((n_points, 1), np.nan)
        incumbents_initial = np.full((n_points, 1), np.nan)
        incumbents_no_noise_initial = np.full((n_points, 1), np.nan)

        # Incumbents

        # Results
        results_bo = None
        results_gp = None

        try:
            # %% Get initial points

            f_initial, x_initial, f_no_noise_initial, x_no_noise_initial = (
                problem.evaluate_with_noisy_simulation(
                    u_initial,
                    config.std_measurement_noise,
                    config.indices_x_measured,
                    config.n_starts_max_evaluate_problem,
                    rng,
                    False,
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

            input_transformer_gp = MinMaxTransformer()
            output_transformer_gp = StandardTransformer()

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
                gp.optimizer.lower_bounds_starting_points = (
                    gp.optimizer.lower_bounds.copy()
                )
                gp.optimizer.upper_bounds_starting_points = (
                    gp.optimizer.upper_bounds.copy()
                )

                starting_points = gp.optimizer.create_lhs_samples(
                    config.n_starts_training_gp,
                    rng,
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

            for i_iteration_bo in range(config.n_iterations_bo):
                # %% Get next point with acquisition function problem

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

                    starting_points = optimizer.create_lhs_samples(
                        config.n_starts_acq_optimization,
                        rng,
                    )
                    optimizer.X0 = starting_points

                solution = optimizer.solve()

                if solution is None:
                    msg = "solution is None"
                    raise Exception(msg)

                print("Solving acquisition problem done.")

                u_next = solution.x[: problem.n_u, :]

                f_next, x_next, f_no_noise_next, x_no_noise_next = (
                    problem.single_evaluate_with_noisy_simulation(
                        u_next,
                        config.std_measurement_noise,
                        config.indices_x_measured,
                        config.n_starts_max_evaluate_problem,
                        rng,
                    )
                )

                incumbent = min(incumbent, f_next)
                incumbent_no_noise = min(incumbent_no_noise, f_no_noise_next)

                # %% Fill points with next value

                u[[n_points_initial + i_iteration_bo], :] = u_next.T
                x[[n_points_initial + i_iteration_bo], :] = x_next.T
                x_no_noise[[n_points_initial + i_iteration_bo], :] = x_no_noise_next.T
                f[[n_points_initial + i_iteration_bo], :] = f_next
                f_no_noise[[n_points_initial + i_iteration_bo], :] = f_no_noise_next
                incumbents[[n_points_initial + i_iteration_bo], :] = incumbent
                incumbents_no_noise[[n_points_initial + i_iteration_bo], :] = (
                    incumbent_no_noise
                )

                print("Getting next point done.")

                # %% Update GP

                set_gp_training_data(
                    gp,
                    u[: n_points_initial + i_iteration_bo + 1, :],
                    f[: n_points_initial + i_iteration_bo + 1, :],
                    input_transformer_gp,
                    output_transformer_gp,
                )

                gp.create_training_problem()

                if isinstance(gp.optimizer, MultiStartOptimizer):
                    gp.optimizer.lower_bounds_starting_points = (
                        gp.optimizer.lower_bounds.copy()
                    )
                    gp.optimizer.upper_bounds_starting_points = (
                        gp.optimizer.upper_bounds.copy()
                    )

                    starting_points = gp.optimizer.create_lhs_samples(
                        config.n_starts_training_gp,
                        rng,
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
                        f[: n_points_initial + i_iteration_bo + 1, :],
                        u[: n_points_initial + i_iteration_bo + 1, :],
                        x[: n_points_initial + i_iteration_bo + 1, :],
                        n_points_initial,
                        i_iteration_bo + 1,
                    )

                    print("Plotting done.")

                print(
                    f"\nBO iteration {i_iteration_bo + 1}/{config.n_iterations_bo} done.\n\n---\n",
                )

            print(f"BO run {i_run_bo + 1}/{config.n_runs_bo} done.\n\n------\n")

        except Exception:
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

    # %% Save results

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    if config.save_bo_results:
        path = results_dir / "results_bo.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(results_bo_complete, file, protocol=pickle.HIGHEST_PROTOCOL)

    if config.save_extended_results:
        path = results_dir / "results_gp.pkl"
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
    u_transformed = input_gp_transformer.fit_transform(
        u,
    )

    f_transformed = output_gp_transformer.fit_transform(
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

    u = SymbolicType.sym("u", problem.n_u, 1)
    u_transformed = input_transformer_gp.transform(u.T).T
    n_variables = problem.n_u
    w = u

    lower_bounds = problem.u_lower_bounds_acquisition
    upper_bounds = problem.u_upper_bounds_acquisition

    mean_transformed, var_transformed = gp.predict(u_transformed.T)

    mean = output_transformer_gp.inverse_transform(mean_transformed)
    std = cas.sqrt(var_transformed) / output_transformer_gp.slope

    # Lower confidence bound
    if config.formulation_acq.startswith("lcb"):
        f_expression = mean - config.factor_lcb_std * std

    # Expected improvement
    else:
        f_expression = -ei_gp(mean, std, incumbent, False)

    f = cas.Function("f", [w], [f_expression])

    return n_variables, f, lower_bounds, upper_bounds
