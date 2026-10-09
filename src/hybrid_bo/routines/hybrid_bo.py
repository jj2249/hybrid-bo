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
from hybrid_bo.array_operations.statistics_functions import sample_mean, sample_variance
from hybrid_bo.config import Config
from hybrid_bo.gp import GP
from hybrid_bo.optimizers import MultiStartOptimizer, OptimizationResult, Optimizer
from hybrid_bo.problem import Problem
from hybrid_bo.results import ResultsBO, ResultsGP
from hybrid_bo.routines.utils import get_sampling_based_data
from hybrid_bo.type_aliases import SymbolicType


def do_hybrid_bo(
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
    gaussian_standard_samples_complete: list[np.ndarray]
    measurement_noise_complete, gaussian_standard_samples_complete = (
        get_sampling_based_data(config, n_points_complete)
    )

    # %% Iterate over multiple BO runs

    for i_run_bo in range(config.n_runs_bo):
        # %% Data for current run

        gaussian_standard_samples: np.ndarray = gaussian_standard_samples_complete[
            i_run_bo
        ]

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

        x_initial: np.ndarray = np.full((n_points, config.n_x), np.nan)
        x_no_noise_initial: np.ndarray = np.full((n_points, config.n_x), np.nan)
        f_initial: np.ndarray = np.full((n_points, 1), np.nan)
        f_no_noise_initial: np.ndarray = np.full((n_points, 1), np.nan)
        incumbents_initial: np.ndarray = np.full((n_points, 1), np.nan)
        incumbents_no_noise_initial: np.ndarray = np.full((n_points, 1), np.nan)

        # Incumbents
        incumbent: float
        incumbent_no_noise: float

        # Results
        results_bo: ResultsBO | None = None
        results_gp: ResultsGP | None = None

        try:
            # %% Get initial points

            f_initial, x_initial, f_no_noise_initial, x_no_noise_initial = (
                problem.evaluate_with_noisy_simulation(
                    u_initial,
                    measurement_noise_initial,
                    config.indices_x_measured,
                    config.n_starts_max_evaluate_problem,
                    config.use_jacobian_evaluate_problem,
                    config.rng,
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

            input_transformer_gp: AffineTransformer = MinMaxTransformer()
            output_transformer_gp: AffineTransformer = StandardTransformer()

            set_gp_training_data(
                gp,
                config,
                u_initial,
                x_initial,
                input_transformer_gp,
                output_transformer_gp,
            )

            # %% Train GP

            gp.create_training_problem()

            if isinstance(gp.optimizer, MultiStartOptimizer):
                gp.optimizer.lower_bounds_starting_points = (
                    gp.optimizer.lower_bounds.copy()
                )  # pyright: ignore[reportOptionalMemberAccess]
                gp.optimizer.upper_bounds_starting_points = (
                    gp.optimizer.upper_bounds.copy()
                )  # pyright: ignore[reportOptionalMemberAccess]

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
                    gaussian_standard_samples,
                    n_points_initial,
                    0,
                )

                print("Plotting done.")

            print(f"\nBO iteration 0/{config.n_iterations_bo} done.\n\n---\n")

            # %% BO loop

            for i_iteration_bo in range(config.n_iterations_bo):
                # %% Get next point with acquisition function problem

                n_variables: int
                f_function: cas.Function
                h_function: cas.Function
                lower_bounds: np.ndarray
                upper_bounds: np.ndarray
                lower_bounds_starting_points: np.ndarray
                upper_bounds_starting_points: np.ndarray
                (
                    n_variables,
                    f_function,
                    h_function,
                    lower_bounds,
                    upper_bounds,
                    lower_bounds_starting_points,
                    upper_bounds_starting_points,
                ) = setup_acq_problem(
                    config,
                    problem,
                    gp,
                    input_transformer_gp,
                    output_transformer_gp,
                    gaussian_standard_samples,
                    incumbent,
                )

                print("Setting up acquisition problem done.")

                optimizer.set_problem(
                    n_variables,
                    f_function,
                    None,
                    h_function,
                    lower_bounds,
                    upper_bounds,
                )

                if isinstance(optimizer, MultiStartOptimizer):
                    optimizer.lower_bounds_starting_points = (
                        lower_bounds_starting_points
                    )
                    optimizer.upper_bounds_starting_points = (
                        upper_bounds_starting_points
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
                        measurement_noise[[n_points_initial + i_iteration_bo], :].T,
                        config.indices_x_measured,
                        config.n_starts_max_evaluate_problem,
                        config.use_jacobian_evaluate_problem,
                        config.rng,
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
                    config,
                    u[: n_points_initial + i_iteration_bo + 1, :],
                    x[: n_points_initial + i_iteration_bo + 1, :],
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
                        f[: n_points_initial + i_iteration_bo + 1, :],
                        u[: n_points_initial + i_iteration_bo + 1, :],
                        x[: n_points_initial + i_iteration_bo + 1, :],
                        gaussian_standard_samples,
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

        path: Path = results_dir / "gaussian_standard_samples.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(
                    gaussian_standard_samples_complete,
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
    config: Config,
    u: np.ndarray,
    x: np.ndarray,
    input_gp_transformer: AffineTransformer,
    output_gp_transformer: AffineTransformer,
) -> None:
    input_gp_train: np.ndarray = np.hstack(
        (
            u[:, config.indices_u_input_gp],
            x[:, config.indices_x_input_gp],
        ),
    )

    output_gp_train: np.ndarray = x[:, config.indices_x_output_gp]

    input_gp_train_transformed: np.ndarray = input_gp_transformer.fit_transform(
        input_gp_train,
    )

    output_gp_train_transformed: np.ndarray = output_gp_transformer.fit_transform(
        output_gp_train,
    )

    gp.set_XY_train(input_gp_train_transformed, output_gp_train_transformed)


def setup_acq_problem(
    config: Config,
    problem: Problem,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    gaussian_standard_samples: np.ndarray,
    incumbent: float,
) -> tuple[
    int,
    cas.Function,
    cas.Function,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if config.formulation_acq not in ["ei", "lcb"]:
        msg: str = (
            f"The given formulation_acq '{config.formulation_acq}'"
            "is not among the allowed options."
        )
        raise Exception(msg)

    if len(config.indices_u_input_gp) + len(config.indices_x_input_gp) != problem.n_u:
        msg: str = (
            "len(config.indices_u_input_gp) + len(config.indices_x_input_gp)"
            "!= problem.n_u"
        )
        raise Exception(msg)

    # %% Get some important variables and bounds

    u: SymbolicType = SymbolicType.sym("u", problem.n_u, 1)  # pyright: ignore[reportArgumentType]
    u_input_gp: SymbolicType = u[config.indices_u_input_gp]
    u_input_gp_samples: SymbolicType = cas.repmat(u_input_gp.T, config.n_samples_gp, 1)

    x_input_gp_samples: SymbolicType = SymbolicType.sym(
        "x_input_gp",  # pyright: ignore[reportArgumentType]
        config.n_samples_gp,  # pyright: ignore[reportArgumentType]
        len(config.indices_x_input_gp),  # pyright: ignore[reportArgumentType]
    )
    x_no_gp_samples: SymbolicType = SymbolicType.sym(
        "x_no_gp_samples",  # pyright: ignore[reportArgumentType]
        config.n_samples_gp,  # pyright: ignore[reportArgumentType]
        len(config.indices_x_no_gp),  # pyright: ignore[reportArgumentType]
    )
    x_output_gp_samples: SymbolicType = SymbolicType.sym(
        "x_output_gp_samples",  # pyright: ignore[reportArgumentType]
        config.n_samples_gp,  # pyright: ignore[reportArgumentType]
        len(config.indices_x_output_gp),  # pyright: ignore[reportArgumentType]
    )

    input_gp_samples: SymbolicType = cas.horzcat(u_input_gp_samples, x_input_gp_samples)  # pyright: ignore[reportAssignmentType]

    x_lower_bounds_samples: np.ndarray = np.tile(
        problem.x_lower_bounds_acquisition.T,
        (config.n_samples_gp, 1),
    )

    x_upper_bounds_samples: np.ndarray = np.tile(
        problem.x_upper_bounds_acquisition.T,
        (config.n_samples_gp, 1),
    )

    x_lower_bounds_starting_points_samples: np.ndarray = np.tile(
        problem.x_lower_bounds_starting_points.T,
        (config.n_samples_gp, 1),
    )

    x_upper_bounds_starting_points_samples: np.ndarray = np.tile(
        problem.x_upper_bounds_starting_points.T,
        (config.n_samples_gp, 1),
    )

    # %% Samples of GP

    output_gp_mean_transformed: SymbolicType
    output_gp_variance_transformed: SymbolicType
    output_gp_mean_transformed, output_gp_variance_transformed = gp.predict(
        input_transformer_gp.transform(input_gp_samples),
    )  # pyright: ignore[reportAssignmentType]

    output_gp_samples_transformed: SymbolicType = (
        output_gp_mean_transformed
        + cas.sqrt(cas.diag(output_gp_variance_transformed)) * gaussian_standard_samples
    )

    output_gp_samples: SymbolicType = output_transformer_gp.inverse_transform(
        output_gp_samples_transformed,
    )  # pyright: ignore[reportAssignmentType]

    # %% Get optimization variables w, f_expression, h_expression,
    # lower_bounds, upper_bounds, lower_bounds_starting_points, and upper_bounds_starting_points

    w: SymbolicType
    n_variables: int
    f_expression: SymbolicType
    h_expression: list[SymbolicType] = []
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    lower_bounds_starting_points: np.ndarray
    upper_bounds_starting_points: np.ndarray

    f_expression_parts: SymbolicType = SymbolicType(config.n_samples_gp, 1)

    for i_sample in range(config.n_samples_gp):
        x_sample: SymbolicType = SymbolicType(problem.n_x, 1)

        if config.indices_x_input_gp:
            x_sample[config.indices_x_input_gp] = x_input_gp_samples[i_sample, :].T

        if config.use_output_gp_as_opt_var:
            x_sample[config.indices_x_output_gp] = x_output_gp_samples[i_sample, :].T
        else:
            x_sample[config.indices_x_output_gp] = output_gp_samples[i_sample, :].T

        if config.indices_x_no_gp:
            x_sample[config.indices_x_no_gp] = x_no_gp_samples[i_sample, :].T

        if config.use_output_gp_as_opt_var:
            h_expression.extend(
                [
                    problem.h_known(u, x_sample),  # pyright: ignore[reportArgumentType]
                    x_sample[config.indices_x_output_gp]
                    - output_gp_samples[i_sample, :].T,
                ],
            )
        else:
            h_expression.append(problem.h_known(u, x_sample))  # pyright: ignore[reportArgumentType]

        if config.formulation_acq == "ei":
            f_expression_parts[i_sample] = cas.fmin(
                problem.f(u, x_sample) - incumbent,  # pyright: ignore[reportOperatorIssue]
                0,
            )  # pyright: ignore[reportOperatorIssue]

        else:  # lcb
            f_expression_parts[i_sample] = problem.f(u, x_sample)

    if config.formulation_acq == "ei":
        f_expression = sample_mean(f_expression_parts)  # pyright: ignore[reportAssignmentType]
    else:  # lcb
        f_expression = sample_mean(
            f_expression_parts,
        ) - config.factor_lcb_std * cas.sqrt(sample_variance(f_expression_parts))

    if config.use_output_gp_as_opt_var:
        w = cas.vertcat(
            u,
            cas.vec(x_input_gp_samples),
            cas.vec(x_output_gp_samples),
            cas.vec(x_no_gp_samples),
        )  # pyright: ignore[reportAssignmentType]

        lower_bounds = np.concatenate(
            (
                problem.u_lower_bounds_acquisition,
                x_lower_bounds_samples[:, config.indices_x_input_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_samples[:, config.indices_x_output_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_samples[:, config.indices_x_no_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

        upper_bounds = np.concatenate(
            (
                problem.u_upper_bounds_acquisition,
                x_upper_bounds_samples[:, config.indices_x_input_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_upper_bounds_samples[:, config.indices_x_output_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_upper_bounds_samples[:, config.indices_x_no_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

        lower_bounds_starting_points = np.concatenate(
            (
                problem.u_lower_bounds_starting_points,
                x_lower_bounds_starting_points_samples[
                    :,
                    config.indices_x_input_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_starting_points_samples[
                    :,
                    config.indices_x_output_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_starting_points_samples[
                    :,
                    config.indices_x_no_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

        upper_bounds_starting_points = np.concatenate(
            (
                problem.u_upper_bounds_starting_points,
                x_upper_bounds_starting_points_samples[
                    :,
                    config.indices_x_input_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_upper_bounds_starting_points_samples[
                    :,
                    config.indices_x_output_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_upper_bounds_starting_points_samples[
                    :,
                    config.indices_x_no_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

    else:
        w = cas.vertcat(
            u,
            cas.vec(x_input_gp_samples),
            cas.vec(x_no_gp_samples),
        )  # pyright: ignore[reportAssignmentType]

        lower_bounds = np.concatenate(
            (
                problem.u_lower_bounds_acquisition,
                x_lower_bounds_samples[:, config.indices_x_input_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_samples[:, config.indices_x_no_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

        upper_bounds = np.concatenate(
            (
                problem.u_upper_bounds_acquisition,
                x_upper_bounds_samples[:, config.indices_x_input_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_upper_bounds_samples[:, config.indices_x_no_gp].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

        lower_bounds_starting_points = np.concatenate(
            (
                problem.u_lower_bounds_starting_points,
                x_lower_bounds_starting_points_samples[
                    :,
                    config.indices_x_input_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_starting_points_samples[
                    :,
                    config.indices_x_no_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

        upper_bounds_starting_points = np.concatenate(
            (
                problem.u_upper_bounds_acquisition,
                x_upper_bounds_starting_points_samples[
                    :,
                    config.indices_x_input_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_upper_bounds_starting_points_samples[
                    :,
                    config.indices_x_no_gp,
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
            ),
        )

    # %% Get number of optimization variables, f, and h

    n_variables = w.shape[0]
    h: cas.Function = cas.Function("h", [w], [cas.vertcat(*h_expression)])
    f: cas.Function = cas.Function("f", [w], [f_expression])

    return (
        n_variables,
        f,
        h,
        lower_bounds,
        upper_bounds,
        lower_bounds_starting_points,
        upper_bounds_starting_points,
    )
