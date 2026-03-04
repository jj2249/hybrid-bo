import casadi as cas
import numpy as np

from hybrid_bo import Config, Problem
from hybrid_bo.type_aliases import SymbolicType


def get_custom_problem(config: Config) -> Problem:
    u: SymbolicType = SymbolicType.sym("u", config.n_u, 1)  # pyright: ignore[reportArgumentType]
    x: SymbolicType = SymbolicType.sym("x", config.n_x, 1)  # pyright: ignore[reportArgumentType]

    f_expression: SymbolicType = (6 * x[0] - 2) ** 2 * cas.sin(12 * x[0] - 4)
    f: cas.Function = cas.Function("f", [u, x], [f_expression])

    h_known_expression: SymbolicType = x[0] + cas.exp(x[0]) - x[1]
    h_known: cas.Function = cas.Function("h_known", [u, x], [h_known_expression])

    h_unkonwn_expression: SymbolicType = cas.sin(u) - x[1]
    h_unknown: cas.Function = cas.Function("h_unknown", [u, x], [h_unkonwn_expression])

    u_lower_bounds_acquisition: np.ndarray = np.array([[-2.0]]).T
    u_upper_bounds_acquisition: np.ndarray = np.array([[2.0]]).T

    x_lower_bounds_acquisition: np.ndarray = np.array([[-np.inf, -np.inf]]).T
    x_upper_bounds_acquisition: np.ndarray = np.array([[np.inf, np.inf]]).T

    u_lower_bounds_starting_points: np.ndarray = u_lower_bounds_acquisition.copy()
    u_upper_bounds_starting_points: np.ndarray = u_upper_bounds_acquisition.copy()

    x_lower_bounds_starting_points: np.ndarray = np.array([[-2.5, -2.0]]).T
    x_upper_bounds_starting_points: np.ndarray = np.array([[1.0, 2.0]]).T

    return Problem(
        f,
        h_known,
        h_unknown,
        u_lower_bounds_acquisition,
        u_upper_bounds_acquisition,
        x_lower_bounds_acquisition,
        x_upper_bounds_acquisition,
        u_lower_bounds_starting_points,
        u_upper_bounds_starting_points,
        x_lower_bounds_starting_points,
        x_upper_bounds_starting_points,
    )
