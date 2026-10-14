import pickle
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from hybrid_bo.legacy.config import Config
from hybrid_bo.legacy.problem import Problem
from hybrid_bo.legacy.results import ResultsBO
from hybrid_bo.sampling import lhs


def do_latin_hypercube_sampling(
    problem: Problem,
    config: Config,
    u_initial_complete: list[np.ndarray],
    rngs: list[np.random.Generator],
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

    # %% Iterate over multiple BO runs

    for i_run_bo in range(config.n_runs_bo):
        # %% Data for current run

        # Each run draws only from its own generator: the design, then each evaluation
        # (noise and fsolve guesses)
        rng = rngs[i_run_bo]

        # Initial points AND BO iteration points
        n_points = u_initial_complete[i_run_bo].shape[0] + config.n_iterations_bo

        # Additional points of u for the BO loop by latin hypercube sampling
        u_additional = lhs(
            problem.u_lower_bounds_starting_points.ravel(),
            problem.u_upper_bounds_starting_points.ravel(),
            config.n_iterations_bo,
            rng,
        )

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

        # Results
        results_bo = None

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

            # %% Create initial plots

            if config.create_plots and (create_plots is not None):
                create_plots(
                    problem,
                    config,
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
                # %% Get next point

                u_next = u_additional[[i_iteration_bo], :].T

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

                # %% Create plots

                if config.create_plots and create_plots:
                    create_plots(
                        problem,
                        config,
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
