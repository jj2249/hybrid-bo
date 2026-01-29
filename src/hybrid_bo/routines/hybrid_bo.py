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
from hybrid_bo.optimizers import MultiStartOptimizer, Optimizer
from hybrid_bo.problem import Problem
from hybrid_bo.results_bo import ResultsBO
from hybrid_bo.routines.utils import get_sampling_based_data, get_training_data
from hybrid_bo.type_aliases import SymbolicType


def do_hybrid_bo(
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
    gaussian_standard_samples_complete: np.ndarray
    measurement_noise_train_complete, gaussian_standard_samples_complete = (
        get_sampling_based_data(config, n_training_points_complete)
    )

    # %% Results of BO

    results_bo: ResultsBO = ResultsBO(problem, config)

    # %% Iterate over multiple BO runs

    for i_run_bo in range(config.n_runs_bo):
        # %% Samples of the Gaussian standard distribution for the reparameterization trick

        gaussian_standard_samples: np.ndarray = gaussian_standard_samples_complete[
            [i_run_bo]
        ].T

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

        _set_gp_XY(
            gp,
            config,
            u_train,
            x_train,
            input_transformer_gp,
            output_transformer_gp,
        )

        print("Preparation done.")

        # %% Train GP

        gp.create_training_problem()

        if isinstance(gp.optimizer, MultiStartOptimizer):
            gp.optimizer.lower_bounds_starting_points = gp.optimizer.lower_bounds.copy()  # pyright: ignore[reportOptionalMemberAccess]
            gp.optimizer.upper_bounds_starting_points = gp.optimizer.upper_bounds.copy()  # pyright: ignore[reportOptionalMemberAccess]

            starting_points: np.ndarray = gp.optimizer.create_lhs_samples(
                n_starts=config.n_starts_training_gp,
                seed=config.seed,
            )
            gp.optimizer.X0 = starting_points

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
                gaussian_standard_samples,
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
                h: cas.Function
                lower_bounds: np.ndarray
                upper_bounds: np.ndarray
                lower_bounds_starting_points: np.ndarray
                upper_bounds_starting_points: np.ndarray
                (
                    n_variables,
                    f,
                    h,
                    lower_bounds,
                    upper_bounds,
                    lower_bounds_starting_points,
                    upper_bounds_starting_points,
                ) = _setup_acq_problem(
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
                    f,
                    None,
                    h,
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
                        n_starts=config.n_starts_acq_optimization,
                        seed=config.seed,
                    )
                    optimizer.X0 = starting_points

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

                _set_gp_XY(
                    gp,
                    config,
                    u_train,
                    x_train,
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
                        n_starts=config.n_starts_training_gp,
                        seed=config.seed,
                    )
                    gp.optimizer.X0 = starting_points

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
                        gaussian_standard_samples,
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
        path: Path = results_dir / "results_hybrid_bo.pkl"
        if path.exists():
            print(f"WARNING: File {path} exists. Will not overwrite.")
        else:
            with path.open("wb") as file:
                pickle.dump(results_bo, file, protocol=pickle.HIGHEST_PROTOCOL)


def _set_gp_XY(
    gp: GP,
    config: Config,
    u_train: np.ndarray,
    x_train: np.ndarray,
    input_gp_transformer: AffineTransformer,
    output_gp_transformer: AffineTransformer,
) -> None:
    input_gp_train: np.ndarray = np.hstack(
        (
            u_train[:, config.indices_u_input_gp],
            x_train[:, config.indices_x_input_gp],
        ),
    )

    output_gp_train: np.ndarray = x_train[:, config.indices_x_output_gp]

    input_gp_train_transformed: np.ndarray = input_gp_transformer.fit_transform(
        input_gp_train,
    )

    output_gp_train_transformed: np.ndarray = output_gp_transformer.fit_transform(
        output_gp_train,
    )

    gp.set_XY_train(input_gp_train_transformed, output_gp_train_transformed)


def _setup_acq_problem(
    config: Config,
    problem: Problem,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    gaussian_standard_samples: np.ndarray,
    incumbent: float,
) -> tuple[
    int, cas.Function, cas.Function, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Sets up the deterministic equivalent of the acquisition function problem according to a selected formulation.

    Parameters
    ----------
    gaussian_standard_samples : np.ndarray with shape (n, 1)
        Samples of the Gaussian standard distribution for the reparameterization trick

    Returns
    -------
    int
        Number of optimization variables
    cas.Function
        Objective function of the problem
    cas.Function
        Function representing the equality constraints (h(w) = 0)
    np.ndarray with shape (m, 1)
        Lower bounds of the optimization variables
    np.ndarray with shape (m, 1)
        Upper bounds of the optimization variables
    np.ndarray with shape (m, 1)
        Lower bounds of the optimization variables used for generating starting points for multi start optimizers
    np.ndarray with shape (m, 1)
        Upper bounds of the optimization variables used for generating starting points for multi start optimizers
    """

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
                    :, config.indices_x_input_gp
                ].flatten("F")[
                    :,
                    np.newaxis,
                ],
                x_lower_bounds_starting_points_samples[
                    :, config.indices_x_no_gp
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
