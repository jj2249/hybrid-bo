import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import GP, Config, Problem
from hybrid_bo.affine_transformers import AffineTransformer

# plt.rcParams["text.usetex"] = True


def create_plots(
    problem: Problem,
    config: Config,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
    f_bo: np.ndarray,
    u_bo: np.ndarray,
    x_bo: np.ndarray,
    gaussian_standard_samples: np.ndarray,
    n_points_initial: int,
    i_iteration: int,
) -> None:

    # %% User definitions

    original = False
    confidence_level = 90
    n_points = [101]

    # %% Get evaluation points

    if len(n_points) != problem.n_u:
        msg = "len(n_points) != problem.n_u"
        raise Exception(msg)

    n_points_total = np.prod(n_points)

    u_components = []
    for i_component in range(problem.n_u):
        u_component = np.linspace(
            problem.u_bounds.lower[i_component],
            problem.u_bounds.upper[i_component],
            n_points[i_component],
        )
        u_components.append(u_component)

    u_grid = np.meshgrid(*u_components, indexing="ij")

    # np.stack():
    #   shape (n_points[0], n_points[1],..., problem.n_u)
    #   Every vector spanning along the last axis corresponds to one evaluation point
    # reshape(): concatenates the evaluation points row-wise
    u = np.stack(u_grid, axis=-1).reshape(
        (n_points_total, problem.n_u),
    )

    x, f = get_evaluation_data(
        problem,
        config,
        u,
        original,
    )

    # %% Get GP mean and std

    mean_gp, std_gp = get_gp_mean_and_std(
        config,
        u,
        x,
        gp,
        input_transformer_gp,
        output_transformer_gp,
    )

    # %% Get samples of f depending on GP samples

    f_samples = get_f_samples(
        problem,
        config,
        u,
        gaussian_standard_samples,
        mean_gp,
        std_gp,
        original,
    )

    # %% Plotting

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

    plot_gp(
        ax[0],
        config,
        u,
        x,
        mean_gp,
        std_gp,
        u_bo,
        x_bo,
        n_points_initial,
        confidence_level,
    )

    plot_f(
        ax[1],
        u,
        f,
        f_samples,
        u_bo,
        f_bo,
        n_points_initial,
        confidence_level,
    )

    plot_acq(ax[2], config, u, f_samples, f_bo)

    ax[0].legend(
        ncols=2,
        loc="lower right",
        mode="expand",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        borderaxespad=0,
    )

    ax[2].set_xlabel("u")

    fig.tight_layout()

    plt.show()


def plot_gp(
    ax: Axes,
    config: Config,
    u: np.ndarray,
    x: np.ndarray,
    mean_gp: np.ndarray,
    std_gp: np.ndarray,
    u_bo: np.ndarray,
    x_bo: np.ndarray,
    n_points_initial: int,
    confidence_level: int,
) -> None:

    # %% Get confidence factor

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
        msg = "For 'confidence_level', only the values 25, 50, 68, 80, 90, 95 and 99 are allowed."
        raise Exception(msg)

    confidence_factor = confidence_factors[confidence_level]

    output_gp_lower_confidence_bound = (mean_gp - confidence_factor * std_gp).flatten()
    output_gp_upper_confidence_bound = (mean_gp + confidence_factor * std_gp).flatten()

    # %% Plotting

    output_gp = x[:, config.indices_x_output_gp].flatten()

    ax.scatter(
        u_bo[:n_points_initial, 0],
        x_bo[:n_points_initial, config.indices_x_output_gp],
        c="k",
        label="Initial points",
    )

    ax.scatter(
        u_bo[n_points_initial:, 0],
        x_bo[n_points_initial:, config.indices_x_output_gp],
        c="r",
        label="Points found with BO",
    )

    ax.plot(u.flatten(), output_gp, color="k", linestyle="--", label="True")
    ax.plot(u.flatten(), mean_gp, color="b", label="Predicted mean")
    ax.fill_between(
        u.flatten(),
        output_gp_lower_confidence_bound,
        output_gp_upper_confidence_bound,
        alpha=0.2,
        color="b",
        label="Confidence interval",
    )

    ax.set_ylabel("Variable $y$")


def plot_f(
    ax: Axes,
    u: np.ndarray,
    f: np.ndarray,
    f_samples: np.ndarray,
    u_bo: np.ndarray,
    f_bo: np.ndarray,
    n_points_initial: int,
    confidence_level: int,
) -> None:
    # %% Get sample mean and confidence interval bounds

    f_mean = np.mean(f_samples, 1)

    exclusive_confidence_quantiles = (100 - confidence_level) / 200.0
    f_confidence_bounds = np.quantile(
        f_samples,
        [exclusive_confidence_quantiles, 1 - exclusive_confidence_quantiles],
        axis=1,
    )

    f_lower_confidence_bound = f_confidence_bounds[0, :]
    f_upper_confidence_bound = f_confidence_bounds[1, :]

    # %% Plotting

    ax.scatter(
        u_bo[:n_points_initial, 0],
        f_bo[:n_points_initial, 0],
        c="k",
        label="Initial points",
    )

    ax.scatter(
        u_bo[n_points_initial:, 0],
        f_bo[n_points_initial:, 0],
        c="r",
        label="Points found with BO",
    )

    ax.plot(u, f, color="k", linestyle="--")
    ax.plot(u, f_mean, color="b")
    ax.fill_between(
        u.flatten(),
        f_lower_confidence_bound,
        f_upper_confidence_bound,
        alpha=0.2,
        color="b",
        label="Confidence interval",
    )

    ax.set_ylabel("Objective")


def plot_acq(
    ax: Axes,
    config: Config,
    u: np.ndarray,
    f_samples: np.ndarray,
    f_bo: np.ndarray,
) -> None:
    incumbent = np.min(f_bo)

    # %% Lower confidence bound

    if config.formulation_acq.startswith("lcb"):
        f_mean = np.mean(f_samples, 1)
        f_std = np.std(f_samples, 1)

        # We want to minimize the lower confidence bound
        acq_eval = f_mean - config.factor_lcb_std * f_std
        label = "Lower Confidence Bound"

    # %% Expected improvement

    else:
        # We want to minimize the expected improvement
        acq_eval = np.mean(np.fmin((f_samples - incumbent), 0), 1)
        label = "Expected Improvement"

    ax.plot(u, acq_eval, color="k", label=label)
    ax.legend(loc="best")
    ax.set_ylabel("Acq.-Fun.")


def get_evaluation_data(
    problem: Problem,
    config: Config,
    u: np.ndarray,
    original: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    f, x = problem.evaluate_with_simulation(
        u,
        config.n_starts_max_evaluate_problem,
        config.use_jacobian_evaluate_problem,
        config.rng,
        original,
    )

    return x, f


def get_gp_mean_and_std(
    config: Config,
    u: np.ndarray,
    x: np.ndarray,
    gp: GP,
    input_transformer_gp: AffineTransformer,
    output_transformer_gp: AffineTransformer,
) -> tuple[np.ndarray, np.ndarray]:

    input_gp = np.hstack(
        (
            u[:, config.indices_u_input_gp],
            x[:, config.indices_x_input_gp],
        ),
    )

    input_gp_transformed = input_transformer_gp.transform(
        input_gp,
    )

    mean_gp_transformed, cov_gp_transformed = gp.predict(
        input_gp_transformed,
    )

    std_gp_transformed = np.sqrt(np.diag(cov_gp_transformed))[:, np.newaxis]

    mean_gp = output_transformer_gp.inverse_transform(
        mean_gp_transformed,
    )
    std_gp = std_gp_transformed / output_transformer_gp.slope.item()

    return mean_gp, std_gp


def get_f_samples(
    problem: Problem,
    config: Config,
    u: np.ndarray,
    gaussian_standard_samples: np.ndarray,
    mean_gp: np.ndarray,
    std_gp: np.ndarray,
    original: bool,
) -> np.ndarray:
    n_points_total = u.shape[0]

    # Shape: (n_points_total, n_samples_gp)
    output_gp_samples = mean_gp + std_gp @ gaussian_standard_samples.T

    f_samples = np.empty((n_points_total, config.n_samples_gp))
    for i_sample in range(config.n_samples_gp):
        f_samples[:, [i_sample]], _ = problem.evaluate_with_fixed_x(
            u,
            output_gp_samples[:, [i_sample]],
            config.indices_x_output_gp,
            config.n_starts_max_evaluate_problem,
            config.use_jacobian_evaluate_problem,
            config.rng,
            original,
        )

    return f_samples
