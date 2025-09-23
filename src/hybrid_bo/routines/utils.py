from typing import overload

import casadi as cas
import numpy as np
from scipy.special import erf as scipy_erf
from scipy.stats import norm as scipy_norm

from ..affine_transformers import AffineTransformer
from ..config import Config
from ..gp import GP
from ..problem import Problem
from ..type_aliases import MathArray


def get_training_data(
    u_train: np.ndarray,
    measurement_noise_train: np.ndarray,
    problem: Problem,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_initial_training_points: int = u_train.shape[0]

    f_train: np.ndarray = np.empty((n_initial_training_points, 1))
    x_train: np.ndarray = np.empty((n_initial_training_points, problem.n_x))
    f_train_no_noise: np.ndarray = f_train.copy()
    x_train_no_noise: np.ndarray = x_train.copy()

    for i_train in range(n_initial_training_points):
        f_train_current: float
        x_train_current: np.ndarray
        f_train_no_noise_current: float
        x_train_no_noise_current: np.ndarray

        (
            f_train_current,
            x_train_current,
            f_train_no_noise_current,
            x_train_no_noise_current,
        ) = problem.evaluate_with_noisy_simulation(
            u_train[[i_train]].T,
            measurement_noise_train[[i_train]].T,
            config.indices_x_measured,
            config.n_starts_max_evaluate_problem,
            config.use_jacobian_evaluate_problem,
            config.check_bounds_evaluate_problem,
            config.seed,
        )

        f_train[i_train] = f_train_current
        x_train[[i_train]] = x_train_current.T
        f_train_no_noise[i_train] = f_train_no_noise_current
        x_train_no_noise[[i_train]] = x_train_no_noise_current.T

    return f_train, x_train, f_train_no_noise, x_train_no_noise


def get_sampling_based_data(
    config: Config,
    n_training_points_complete: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    n_training_points_total: int = np.sum(n_training_points_complete)

    measurement_noise_train_temp: np.ndarray = np.atleast_2d(
        scipy_norm.rvs(
            0,
            config.std_measurement_noise,
            n_training_points_total,
            random_state=config.rng,
        ),
    ).T

    measurement_noise_train_complete: list[np.ndarray] = []
    i_measurement_noise_train: int = 0
    for i_run_bo in range(config.n_runs_bo):
        i_measurement_noise_train_new: int = (
            i_measurement_noise_train + n_training_points_complete[i_run_bo]
        )
        measurement_noise_train_complete.append(
            measurement_noise_train_temp[
                i_measurement_noise_train:i_measurement_noise_train_new
            ],
        )
        i_measurement_noise_train = i_measurement_noise_train_new

    gaussian_standard_samples_temp: np.ndarray = scipy_norm.rvs(
        size=config.n_samples_gp * config.n_runs_bo,
        random_state=config.rng,
    )  # pyright: ignore[reportAssignmentType]

    gaussian_standard_samples_complete: np.ndarray = (
        gaussian_standard_samples_temp.reshape(
            (config.n_runs_bo, config.n_samples_gp),
        )
    )

    return measurement_noise_train_complete, gaussian_standard_samples_complete


def get_evaluation_data(
    problem: Problem,
    n_starts_max_evaluate_problem: int,
    use_jacobian_evaluate_problem: bool,
    check_bounds_evaluate_problem: bool,
    n_eval_points: list[int],
    original: bool = False,
    seed: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
]:
    """Returns u_eval, x_eval, f_eval."""

    if len(n_eval_points) != problem.n_u:
        msg: str = "len(n_eval_points) != problem.n_u"
        raise Exception(msg)

    n_eval_points_total: np.int64 = np.prod(n_eval_points)

    u_eval_components: list[np.ndarray] = []
    for i_component in range(problem.n_u):
        u_eval_component: np.ndarray = np.linspace(
            problem.u_lower_bounds[i_component],
            problem.u_upper_bounds[i_component],
            n_eval_points[i_component],
        )
        u_eval_components.append(u_eval_component)

    u_eval_grid: list[np.ndarray] = np.meshgrid(*u_eval_components, indexing="ij")  # pyright: ignore[reportAssignmentType]

    # stack():
    #   shape (n_eval_points[0], n_eval_points[1],..., problem.n_u)
    #   Every vector spanning along the last axis corresponds to one evaluation point
    # reshape(): concatenates the evaluation points row-wise
    u_eval: np.ndarray = np.stack(u_eval_grid, axis=-1).reshape(
        (n_eval_points_total, problem.n_u),
    )

    x_eval: np.ndarray = np.empty((n_eval_points_total, problem.n_x))
    f_eval: np.ndarray = np.empty((n_eval_points_total, 1))
    for i_eval in range(n_eval_points_total):
        if original:
            f_current, x_current = problem.evaluate_with_simulation_original(
                u_eval[[i_eval]].T,
                n_starts_max_evaluate_problem,
                use_jacobian_evaluate_problem,
                check_bounds_evaluate_problem,
                seed,
            )
        else:
            f_current, x_current = problem.evaluate_with_simulation(
                u_eval[[i_eval]].T,
                n_starts_max_evaluate_problem,
                use_jacobian_evaluate_problem,
                check_bounds_evaluate_problem,
                seed,
            )

        f_eval[i_eval] = f_current
        x_eval[[i_eval]] = x_current.T

    f_eval_grid: np.ndarray = f_eval.reshape(n_eval_points)
    x_eval_grid: list[np.ndarray] = []
    x_eval_grid: list[np.ndarray] = [
        x_eval[:, i_component].reshape(n_eval_points)
        for i_component in range(problem.n_x)
    ]

    return u_eval, x_eval, f_eval, u_eval_grid, x_eval_grid, f_eval_grid


def get_f_eval_samples(
    problem: Problem,
    config: Config,
    u_eval: np.ndarray,
    x_eval: np.ndarray,
    gaussian_standard_samples: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    original: bool = False,
) -> np.ndarray:
    n_eval_points_total: int = u_eval.shape[0]

    input_gp_eval: np.ndarray = np.hstack(
        (
            u_eval[:, config.indices_u_input_gp],
            x_eval[:, config.indices_x_input_gp],
        ),
    )

    input_gp_eval_transformed: np.ndarray = input_transformer_gp.transform(
        input_gp_eval,
    )

    n_samples_gp: int = gaussian_standard_samples.shape[0]

    output_gp_eval_mean_transformed: np.ndarray
    output_gp_eval_variance_transformed: np.ndarray
    output_gp_eval_mean_transformed, output_gp_eval_variance_transformed = gp.predict(
        input_gp_eval_transformed,
    )

    output_gp_eval_std_transformed: np.ndarray = np.sqrt(
        np.diag(output_gp_eval_variance_transformed),
    )[:, np.newaxis]

    # Shape: (n_samples_gp, n_eval_points)
    output_gp_eval_samples_transformed: np.ndarray = (
        output_gp_eval_mean_transformed
        + output_gp_eval_std_transformed @ gaussian_standard_samples.T
    ).T

    output_gp_eval_samples_transformed_flattened: np.ndarray = (
        output_gp_eval_samples_transformed.flatten()[:, np.newaxis]
    )

    output_gp_eval_samples_flattened: np.ndarray = (
        output_transformer_gp.inverse_transform(
            output_gp_eval_samples_transformed_flattened,
        )
    )

    output_gp_eval_samples: np.ndarray = output_gp_eval_samples_flattened.reshape(
        (n_samples_gp, n_eval_points_total),
    )

    f_eval_samples: np.ndarray = np.empty((n_samples_gp, n_eval_points_total))
    for i_samples in range(n_samples_gp):
        for i_eval in range(n_eval_points_total):
            u_eval_current: np.ndarray = u_eval[[i_eval]].T
            output_gp_eval_samples_current: np.ndarray = output_gp_eval_samples[
                [i_samples],
                i_eval,
            ][:, np.newaxis]

            if original:
                f_eval_samples[i_samples, i_eval], _ = (
                    problem.evaluate_with_fixed_x_original(
                        u_eval_current,
                        output_gp_eval_samples_current,
                        config.indices_x_output_gp,
                        config.n_starts_max_evaluate_problem,
                        config.use_jacobian_evaluate_problem,
                        config.check_bounds_evaluate_problem,
                        config.seed,
                    )
                )

            else:
                f_eval_samples[i_samples, i_eval], _ = problem.evaluate_with_fixed_x(
                    u_eval_current,
                    output_gp_eval_samples_current,
                    config.indices_x_output_gp,
                    config.n_starts_max_evaluate_problem,
                    config.use_jacobian_evaluate_problem,
                    config.check_bounds_evaluate_problem,
                    config.seed,
                )

    return f_eval_samples


@overload
def pdf_normal(x: cas.SX) -> cas.SX: ...
@overload
def pdf_normal(x: np.ndarray) -> np.ndarray: ...


def pdf_normal(x: cas.SX | np.ndarray) -> cas.SX | np.ndarray:
    pdf: cas.SX | np.ndarray
    if isinstance(x, cas.SX):
        pdf = cas.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    else:
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return pdf


@overload
def cdf_normal(x: cas.SX) -> cas.SX: ...
@overload
def cdf_normal(x: np.ndarray) -> np.ndarray: ...


def cdf_normal(x: MathArray) -> MathArray:
    cdf: MathArray
    if isinstance(x, cas.SX):
        cdf = 0.5 * (1 + cas.erf(x / np.sqrt(2)))
    else:
        cdf = 0.5 * (1 + scipy_erf(x / np.sqrt(2)))
    return cdf


@overload
def ei_gp(
    mean: cas.SX,
    std: cas.SX,
    incumbent: float,
    maximize: bool,
    epsilon: float,
) -> cas.SX: ...
@overload
def ei_gp(
    mean: np.ndarray,
    std: np.ndarray,
    incumbent: float,
    maximize: bool,
    epsilon: float,
) -> np.ndarray: ...


def ei_gp(
    mean: MathArray,
    std: MathArray,
    incumbent: float,
    maximize: bool = True,
    epsilon: float = 1.0e-10,
) -> MathArray:
    """Returns expected improvement for BO with Gaussian processes as surrogate model.

    See https://botorch.org/docs/acquisition#analytic-acquisition-functions for maximization problem.
    See https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html?utm_source=chatgpt.com#ego for minimization problem.

    Parameters
    ----------
    mean : MathArray
        Mean of the Gaussian process
    std : MathArray, m x n
        Standard deviation of the Gaussian process
    incumbent : float
        Best observation of the function to minimize so far
    maximize : bool, optional
        Determines if dealing with a maximization (True) or minimization (False) problem.
        By default True
    epsilon : float, optional
        Sometimes added to the standard deviation to prevent devision by zero.
        By default 1.0e-10

    Returns
    -------
    MathArray
        Expected improvement
    """

    if mean.shape != std.shape:
        msg: str = "mean.shape != std.shape"
        raise Exception(msg)

    sign: float = 1.0
    if not maximize:
        sign = -1.0

    gamma: MathArray = sign * (mean - incumbent) / (std + epsilon)
    ei: MathArray = sign * (mean - incumbent) * cdf_normal(gamma) + std * pdf_normal(
        gamma,
    )

    return ei
