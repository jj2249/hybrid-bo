from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.array_operations import sample_mean
from hybrid_bo.routines.utils import get_evaluation_data, get_f_eval_samples

if TYPE_CHECKING:
    from matplotlib.figure import Figure

plt.rcParams["text.usetex"] = True


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f_train: np.ndarray,
    u_train: np.ndarray,
    x_train: np.ndarray,
    gaussian_standard_samples: np.ndarray,
    n_initial_training_points: int,
    i_iteration: int,
) -> None:
    # See https://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data
    confidence_factors = {
        25: 0.32,
        50: 0.67,
        68: 0.99,
        80: 1.28,
        90: 1.64,
        95: 1.96,
        99: 2.58,
    }
    if config.confidence_level not in confidence_factors:
        msg: str = "For 'confidence_level', only the values 25, 50, 68, 80, 90, 95 and 99 are allowed."
        raise Exception(msg)

    # %% Get evaluation points

    u_eval: np.ndarray
    x_eval: np.ndarray
    f_eval: np.ndarray
    u_eval_grid: list[np.ndarray]
    x_eval_grid: list[np.ndarray]
    f_eval_grid: np.ndarray
    u_eval, x_eval, f_eval, u_eval_grid, x_eval_grid, f_eval_grid = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.check_bounds_evaluate_problem,
        config.n_eval_points,
        True,
        config.seed,
    )

    # %% Get samples of f at evaluation points

    f_eval_samples: np.ndarray = get_f_eval_samples(
        problem,
        config,
        u_eval,
        x_eval,
        gaussian_standard_samples,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        True,
    )

    print("Preparing plotting done.")

    # %% Plotting

    fig: Figure
    ax: np.ndarray  # of Axes
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

    plot_gp(
        ax[0],
        config,
        u_eval,
        x_eval,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        u_train,
        x_train,
        n_initial_training_points,
        confidence_factors[config.confidence_level],
    )

    print("Plotting output of GP done.")

    plot_f(
        ax[1],
        config,
        u_eval,
        f_eval,
        f_eval_samples,
        u_train,
        f_train,
        n_initial_training_points,
    )

    print("Plotting f done.")

    plot_acq(ax[2], config, u_eval, f_eval_samples, np.min(f_train))

    print("Plotting acquisition function done")

    ax[2].set_xlabel("u")
    fig.suptitle(f"Iteration {i_iteration}")
    fig.tight_layout()
    plt.show()


def plot_gp(
    ax: Axes,
    config: Config,
    u_eval: np.ndarray,
    x_eval: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    u_train: np.ndarray,
    x_train: np.ndarray,
    n_initial_training_points: int,
    confidence_factor: float,
) -> None:
    # %% Get plotting data

    u_eval_transformed: np.ndarray = input_transformer_gp.transform(u_eval)  # pyright: ignore[reportAssignmentType]

    y_eval: np.ndarray = x_eval[:, config.indices_x_output_gp]
    y_eval_transformed: np.ndarray = output_transformer_gp.transform(y_eval)  # pyright: ignore[reportAssignmentType]

    y_mean_transformed: np.ndarray
    y_variance_transformed: np.ndarray
    y_mean_transformed, y_variance_transformed = gp.predict(u_eval_transformed)  # pyright: ignore[reportAssignmentType]
    y_std_transformed: np.ndarray = np.sqrt(np.diag(y_variance_transformed))[
        :,
        np.newaxis,
    ]

    y_mean: np.ndarray = output_transformer_gp.inverse_transform(y_mean_transformed)  # pyright: ignore[reportAssignmentType]
    y_std: np.ndarray = y_std_transformed / output_transformer_gp.slope

    y_lower_confidence_bound: np.ndarray = (
        y_mean - confidence_factor * y_std
    ).flatten()
    y_upper_confidence_bound: np.ndarray = (
        y_mean + confidence_factor * y_std
    ).flatten()

    y_train: np.ndarray = x_train[:, config.indices_x_output_gp]

    # %% Plotting

    ax.scatter(
        u_train[:n_initial_training_points],
        y_train[:n_initial_training_points],
        c="k",
        label="Initial training points",
    )

    ax.scatter(
        u_train[n_initial_training_points:],
        y_train[n_initial_training_points:],
        c="r",
        label="Training points found with BO",
    )

    ax.plot(u_eval, y_eval, color="k", linestyle="--", label="True")
    ax.plot(u_eval, y_mean, color="b", label="Predicted mean")
    ax.fill_between(
        u_eval.flatten(),
        y_lower_confidence_bound,
        y_upper_confidence_bound,
        alpha=0.2,
        color="b",
        label="Confidence interval",
    )

    ax.legend(ncols=2, loc="lower right")
    ax.set_ylabel("Variable $y$")


def plot_f(
    ax: Axes,
    config: Config,
    u_eval: np.ndarray,
    f_eval: np.ndarray,
    f_eval_samples: np.ndarray,
    u_train: np.ndarray,
    f_train: np.ndarray,
    n_initial_training_points: int,
) -> None:
    # %% Get sample mean and confidence interval bounds

    f_mean: np.ndarray = sample_mean(f_eval_samples).T

    exclusive_confidence_quantiles: float = (100 - config.confidence_level) / 200.0
    f_confidence_bounds: np.ndarray = np.quantile(
        f_eval_samples,
        [exclusive_confidence_quantiles, 1 - exclusive_confidence_quantiles],
        axis=0,
    )

    f_lower_confidence_bound: np.ndarray = f_confidence_bounds[0, :]
    f_upper_confidence_bound: np.ndarray = f_confidence_bounds[1, :]

    # %%Plotting

    ax.scatter(
        u_train[:n_initial_training_points],
        f_train[:n_initial_training_points],
        c="k",
    )

    ax.scatter(
        u_train[n_initial_training_points:],
        f_train[n_initial_training_points:],
        c="r",
    )

    ax.plot(u_eval, f_eval, color="k", linestyle="--")
    ax.plot(u_eval, f_mean, color="b")
    ax.fill_between(
        u_eval.flatten(),
        f_lower_confidence_bound,
        f_upper_confidence_bound,
        alpha=0.2,
        color="b",
        label="Confidence interval",
    )
    ax.set_ylabel("f")


def plot_acq(
    ax: Axes,
    config: Config,
    u_eval: np.ndarray,
    f_eval_samples: np.ndarray,
    incumbent: float,
) -> None:
    label: str
    acq_eval: np.ndarray

    # %% Lower confidence bound

    if config.formulation_acq.startswith("lcb"):
        f_mean: np.ndarray = np.mean(f_eval_samples, 0).T
        f_std: np.ndarray = np.std(f_eval_samples, 0).T

        # We want to minimize the lower confidence bound
        acq_eval = f_mean - config.factor_lcb_std * f_std
        label = "Lower Confidence Bound"

    # %% Expected improvement

    else:
        # We want to minimize the expected improvement
        acq_eval = np.mean(np.fmin((f_eval_samples - incumbent), 0), 0)[:, np.newaxis]  # pyright: ignore[reportAssignmentType]
        label = "Expected Improvement"

    ax.plot(u_eval, acq_eval, color="k", label=label)
    ax.legend()
    ax.set_ylabel("Acq.-Fun. $\\alpha _{\\phi}$")
