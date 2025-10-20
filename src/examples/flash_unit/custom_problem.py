from typing import override

import casadi as cas
import numpy as np
from matplotlib.axes import Axes

from hybrid_bo import Config, Problem


def custom_problem(config: Config) -> Problem:
    return CustomProblem(config)


class CustomProblem(Problem):
    def __init__(self, config: Config) -> None:
        self.config: Config = config

        # %% Parameters

        self.F: float = 1.0  # Feed stream flow rate (mol/s)
        self.D: float = 0.3  # Top stream flow rate (mol/s)
        self.z_1: float = 0.5  # Mole Fraction CH3COOH of Feed Stream (-)

        # Antoine Coefficient A of H2O, to p in Pa and T in Kelvin (-)
        self.A_1: float = 10.19590302
        # Antoine Coefficient B of H2O, to p in Pa and T in Kelvin (-)
        self.B_1: float = 1730.6
        # Antoine Coefficient C of H2O, to p in Pa and T in Kelvin (-)
        self.C_1: float = -39.75

        self.a_12: float = 2.28422  # NRTL Aspen Parameter aij of H20, CH3COOH (-)
        self.a_21: float = -1.56707  # NRTL Aspen Parameter aji of CH3COOH, H2O (-)
        self.b_12: float = -478.146  # NRTL Aspen Parameter bij of H2O, CH3COOH (K)
        self.b_21: float = 542.84  # NRTL Aspen Parameter bji of CH3COOH, H2O (K)
        self.c_12: float = 0.373484  # NRTL Aspen Parameter cij of H2O, CH3COOH (-)
        self.c_21: float = 0.373484  # NRTL Aspen Parameter cji of CH3COOH, H2O (-)
        self.d_12: float = 0.0  # NRTL Aspen Parameter dij of H2O, CH3COOH (1/K)
        self.d_21: float = 0.0  # NRTL Aspen Parameter dji of CH3COOH, H2O (1/K)
        self.e_12: float = 0.0  # NRTL Aspen Parameter eij of H2O, CH3COOH (-)
        self.e_21: float = 0.0  # NRTL Aspen Parameter eji of CH3COOH, H2O (-)
        self.f_12: float = 0.0  # NRTL Aspen Parameter fij of H2O, CH3COOH (-)
        self.f_21: float = 0.0  # NRTL Aspen Parameter fji of CH3COOH, H2O (-)
        self.zeta: float = 100  # Violation of purity cost ($)
        self.beta: float = 0.000001  # Mixture heating cost ($/K^2)
        self.eta: float = 0.5  # Mixture energy/pressure cost ($/bar^2)
        self.y_1_set: float = 0.66  # Desired H2O fraction in top stream (-)
        self.p_amb: float = 1.01325  # Ambient pressure (bar)

        # %% u, x and their bounds

        T: cas.SX = cas.SX.sym("T", 1, 1)  # pyright: ignore[reportArgumentType] # Temperature in the flash unit (1000 * K)
        p: cas.SX = cas.SX.sym("p", 1, 1)  # pyright: ignore[reportArgumentType] # Pressure in the flash unit (bar)

        u: cas.SX = cas.vertcat(T, p)  # pyright: ignore[reportAssignmentType]

        T_lower_bound: float = 363.15 / 1000
        T_upper_bound: float = 403.15 / 1000
        p_lower_bound: float = 0.8
        p_upper_bound: float = 2.6

        u_lower_bounds: np.ndarray = np.array([T_lower_bound, p_lower_bound])[
            :,
            np.newaxis,
        ]
        u_upper_bounds: np.ndarray = np.array([T_upper_bound, p_upper_bound])[
            :,
            np.newaxis,
        ]

        x_1: cas.SX = cas.SX.sym("x_1", 1, 1)  # pyright: ignore[reportArgumentType] # Mole Fraction of H2O in bottom Stream (-)
        gamma_1_ln: cas.SX = cas.SX.sym("gamma_1_ln", 1, 1)  # pyright: ignore[reportArgumentType] # Activation coefficient (-)

        x: cas.SX = cas.vertcat(x_1, gamma_1_ln)  # pyright: ignore[reportAssignmentType]

        x_1_lower_bound: float = 0.0
        x_1_upper_bound: float = 1.0
        gamma_1_lower_bound: float = np.log(0.5)
        gamma_1_upper_bound: float = np.log(2.0)

        x_lower_bounds: np.ndarray = np.array([x_1_lower_bound, gamma_1_lower_bound])[
            :,
            np.newaxis,
        ]
        x_upper_bounds: np.ndarray = np.array([x_1_upper_bound, gamma_1_upper_bound])[
            :,
            np.newaxis,
        ]

        # %% Helper variables

        B: float = self.F - self.D
        y_1: cas.SX = (self.F * self.z_1 - B * x_1) / self.D
        x_2: cas.SX = 1 - x_1  # (-)
        p_1_sat: cas.SX = (
            cas.power(10, self.A_1 - self.B_1 / (self.C_1 + T * 1000)) / 1e5  # (bar)
        )
        tau_12: cas.SX = self.a_12 + self.b_12 / (T * 1000)  # (-)
        tau_21: cas.SX = self.a_21 + self.b_21 / (T * 1000)  # (-)
        alpha_12: cas.SX = self.c_12 + self.d_12 * (T * 1000 - 273.15)  # (-)
        alpha_21: cas.SX = self.c_21 + self.d_21 * (T * 1000 - 273.15)  # (-)
        G_12: cas.SX = cas.exp(-alpha_12 * tau_12)  # (-)
        G_21: cas.SX = cas.exp(-alpha_21 * tau_21)  # (-)

        # %% f, h and g

        f_expression: cas.SX = (
            self.zeta * ((y_1 - self.y_1_set) ** 2)
            + self.beta * ((1000 * T) ** 2)
            + self.eta * ((p - self.p_amb) ** 2)
        )
        f: cas.Function = cas.Function("f", [u, x], [f_expression])

        h_known_expression: list[cas.SX] = [
            p * y_1 - p_1_sat * x_1 * cas.exp(gamma_1_ln),
        ]

        h_known: cas.Function = cas.Function(
            "h_known",
            [u, x],
            [cas.vertcat(*h_known_expression)],
        )

        h_unknown_expression: cas.SX = gamma_1_ln - x_2**2 * (
            tau_21 * (G_21 / (x_1 + x_2 * G_21)) ** 2
            + ((tau_12 * G_12) / (x_2 + x_1 * G_12) ** 2)
        )

        h_unknown: cas.Function = cas.Function(
            "h_unknown",
            [u, x],
            [h_unknown_expression],
        )

        # %% Base class constructor

        super().__init__(
            f,
            h_known,
            h_unknown,
            u_lower_bounds,
            u_upper_bounds,
            x_lower_bounds,
            x_upper_bounds,
        )

    @override
    def evaluate_with_simulation(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        check_bounds: bool = True,
        seed: int | None = None,
    ) -> tuple[float, np.ndarray]:
        f_temp: float
        x_temp: np.ndarray
        f_temp, x_temp = self.evaluate_with_simulation_original(
            u,
            n_starts_max,
            use_jacobian,
            check_bounds,
            seed,
        )

        f: float
        x: np.ndarray
        if x_temp[0, 0] > self.z_1:
            x = np.array([self.config.rng.uniform(self.z_1, 1.0), 0.0])[:, np.newaxis]

            f = (
                self.zeta * (self.y_1_set**2)
                + self.beta * ((1000 * u[0, 0]) ** 2)
                + self.eta * ((u[1, 0] - self.p_amb) ** 2)
            )

        else:
            f = f_temp
            x = x_temp

        return f, x

    @override
    def evaluate_with_fixed_x(
        self,
        u: np.ndarray,
        x_fixed: np.ndarray,
        indices_x_fixed: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        check_bounds: bool = True,
        seed: int | None = None,
    ) -> tuple[float, np.ndarray]:
        f_temp: float
        x_temp: np.ndarray

        f_temp, x_temp = self.evaluate_with_fixed_x_original(
            u,
            x_fixed,
            indices_x_fixed,
            n_starts_max,
            use_jacobian,
            check_bounds,
            seed,
        )

        f: float
        x: np.ndarray

        if x_temp[0, 0] > self.z_1:
            x = np.array([self.config.rng.uniform(self.z_1, 1.0), 0.0])[:, np.newaxis]

            f = (
                self.zeta * (self.y_1_set**2)
                + self.beta * ((1000 * u[0, 0]) ** 2)
                + self.eta * ((u[1, 0] - self.p_amb) ** 2)
            )

        else:
            f = f_temp
            x = x_temp

        return f, x

    @override
    def evaluate_with_noisy_simulation(
        self,
        u: np.ndarray,
        measurement_noise: np.ndarray,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        check_bounds: bool = True,
        seed: int | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        f_temp: float
        x_temp: np.ndarray
        f_no_noise_temp: float
        x_no_noise_temp: np.ndarray

        f_temp, x_temp, f_no_noise_temp, x_no_noise_temp = (
            self.evaluate_with_noisy_simulation_original(
                u,
                measurement_noise,
                indices_x_measured,
                n_starts_max,
                use_jacobian,
                check_bounds,
            )
        )

        f: float
        x: np.ndarray
        f_no_noise: float
        x_no_noise: np.ndarray

        if x_temp[0, 0] > self.z_1:
            x = np.array([self.config.rng.uniform(self.z_1, 1.0), 0.0])[:, np.newaxis]

            f = (
                self.zeta * (self.y_1_set**2)
                + self.beta * ((1000 * u[0, 0]) ** 2)
                + self.eta * ((u[1, 0] - self.p_amb) ** 2)
            )

            f_no_noise = f
            x_no_noise = x

        else:
            f = f_temp
            x = x_temp
            f_no_noise = f_no_noise_temp
            x_no_noise = x_no_noise_temp

        return f, x, f_no_noise, x_no_noise


def plot_physical_boundary(
    problem: CustomProblem,
    ax: Axes,
) -> None:
    # %% Variables to create grid with

    n_eval_points: list[int] = [201, 201]

    T: np.ndarray = np.linspace(
        problem.u_lower_bounds[0],
        problem.u_upper_bounds[0],
        n_eval_points[0],
    )

    p: np.ndarray = np.linspace(
        problem.u_lower_bounds[1],
        problem.u_upper_bounds[1],
        n_eval_points[1],
    )

    # %% Create grid variables

    T_eval_grid: np.ndarray
    p_eval_grid: np.ndarray
    T_eval_grid, p_eval_grid = np.meshgrid(T, p)

    # %% Calculate "flat" variables for evaluation

    u_eval: np.ndarray = np.stack((T_eval_grid.flatten(), p_eval_grid.flatten()), 1)
    n_eval_points_total: int = u_eval.shape[0]

    f_eval: np.ndarray = np.empty((n_eval_points_total, 1))
    x_eval: np.ndarray = np.empty((n_eval_points_total, problem.n_x))
    for i_eval in range(n_eval_points_total):
        f_eval_current: float
        x_eval_current: np.ndarray
        f_eval_current, x_eval_current = problem.evaluate_with_simulation_original(
            u_eval[[i_eval]].T,
        )

        f_eval[i_eval] = f_eval_current
        x_eval[i_eval] = x_eval_current.flatten()

    # %% Get indices of physical boundary

    indices_boundary: np.ndarray = np.nonzero(
        (x_eval[:, 0] >= problem.z_1) & (x_eval[:, 0] <= (problem.z_1 + 0.001)),
    )  # pyright: ignore[reportAssignmentType]

    # %% Get u at physical boundary

    T_plot: np.ndarray = u_eval[indices_boundary, 0].flatten() * 1000
    p_plot: np.ndarray = u_eval[indices_boundary, 1].flatten()

    # %% Plotting

    ax.plot(T_plot, p_plot, c="k", linewidth=3.0)
