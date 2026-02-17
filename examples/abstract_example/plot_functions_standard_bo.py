from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer
from hybrid_bo.routines.utils import ei_gp, get_evaluation_data

if TYPE_CHECKING:
    from matplotlib.figure import Figure

plt.rcParams["text.usetex"] = True

confidence_level: int = 95
n_eval_points: list[int] = [101]


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    n_initial_points: int,
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
    if confidence_level not in confidence_factors:
        msg: str = (
            "For 'confidence_level',"
            "only the values 25, 50, 68, 80, 90, 95 and 99 are allowed."
        )
        raise Exception(msg)

    # %% Get evaluation points

    u_eval: np.ndarray
    f_eval: np.ndarray
    u_eval, _, f_eval, _, _, _ = get_evaluation_data(
        problem,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        n_eval_points,
        False,
        config.rng,
    )

    u_eval_transformed: np.ndarray = input_transformer_gp.transform(u_eval)  # pyright: ignore[reportAssignmentType]

    mean_gp_eval_transformed: np.ndarray
    var_gp_eval_transformed: np.ndarray
    mean_gp_eval_transformed, var_gp_eval_transformed = gp.predict(u_eval_transformed)  # pyright: ignore[reportAssignmentType]

    mean_gp_eval: np.ndarray = output_transformer_gp.inverse_transform(
        mean_gp_eval_transformed,
    )  # pyright: ignore[reportAssignmentType]
    std_gp_eval: np.ndarray = (
        np.sqrt(np.diag(var_gp_eval_transformed))[:, np.newaxis]
        / output_transformer_gp.slope
    )

    print("Preparing plotting done.")

    # %% Plotting

    fig: Figure
    ax: np.ndarray  # of Axes
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    plot_output_gp(
        ax[0],
        u_eval,
        f_eval,
        gp,
        input_transformer_gp,
        output_transformer_gp,
        u,
        f,
        n_initial_points,
        confidence_factors[confidence_level],
    )

    print("Plotting output of GP done.")

    plot_acq(ax[1], config, u_eval, mean_gp_eval, std_gp_eval, np.min(f))

    print("Plotting acquisition function done")

    ax[1].set_xlabel("u")
    fig.suptitle(f"Iteration {i_iteration}")
    fig.tight_layout()
    plt.show()


def plot_output_gp(
    ax: Axes,
    u_eval: np.ndarray,
    f_eval: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    u: np.ndarray,
    f: np.ndarray,
    n_initial_points: int,
    confidence_factor: float,
) -> None:
    # %% Get plotting data

    u_eval_transformed: np.ndarray = input_transformer_gp.transform(u_eval)  # pyright: ignore[reportAssignmentType]
    f_eval_transformed: np.ndarray = output_transformer_gp.transform(f_eval)  # pyright: ignore[reportAssignmentType]

    f_mean_transformed: np.ndarray
    f_variance_transformed: np.ndarray
    f_mean_transformed, f_variance_transformed = gp.predict(u_eval_transformed)  # pyright: ignore[reportAssignmentType]
    f_std_transformed: np.ndarray = np.sqrt(np.diag(f_variance_transformed))[
        :,
        np.newaxis,
    ]

    f_mean: np.ndarray = output_transformer_gp.inverse_transform(f_mean_transformed)  # pyright: ignore[reportAssignmentType]
    f_std: np.ndarray = f_std_transformed / output_transformer_gp.slope

    f_lower_confidence_bound: np.ndarray = (
        f_mean - confidence_factor * f_std
    ).flatten()
    f_upper_confidence_bound: np.ndarray = (
        f_mean + confidence_factor * f_std
    ).flatten()

    # %% Plotting

    ax.scatter(
        u[:n_initial_points],
        f[:n_initial_points],
        c="k",
        label="Initial points",
    )

    ax.scatter(
        u[n_initial_points:],
        f[n_initial_points:],
        c="r",
        label="Points found with BO",
    )

    ax.plot(u_eval, f_eval, color="k", linestyle="--", label="True")
    ax.plot(u_eval, f_mean, color="b", label="GP mean")
    ax.fill_between(
        u_eval.flatten(),
        f_lower_confidence_bound,
        f_upper_confidence_bound,
        alpha=0.2,
        color="b",
        label="GP confidence interval",
    )

    ax.legend(ncols=2, loc="lower right")
    ax.set_ylabel("$f$")


def plot_acq(
    ax: Axes,
    config: Config,
    u_eval: np.ndarray,
    mean_gp_eval: np.ndarray,
    std_gp_eval: np.ndarray,
    incumbent: float,
) -> Axes:
    label: str
    acq_eval: np.ndarray

    # Lower confidence bound
    if config.formulation_acq.startswith("lcb"):
        # We want to minimize the lower confidence bound.
        acq_eval = mean_gp_eval - config.factor_lcb_std * std_gp_eval
        label = "Lower Confidence Bound"

    # Expected improvement
    else:
        acq_eval = -ei_gp(
            mean_gp_eval,
            std_gp_eval,
            incumbent,
            maximize=False,
        )  # pyright: ignore[reportCallIssue]
        label = "Expected Improvement"

    ax.plot(u_eval, acq_eval, color="k", label=label)
    ax.legend()
    ax.set_ylabel("Acq.-Fun. $\\alpha _{\\phi}$")

    return ax
