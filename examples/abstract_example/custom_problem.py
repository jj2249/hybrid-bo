import casadi as cas
import numpy as np

from hybrid_bo import Config, Problem
from hybrid_bo.type_aliases import SymbolicType


def get_custom_problem(config: Config) -> Problem:
    u: SymbolicType = SymbolicType.sym("u", config.n_u, 1)  # pyright: ignore[reportArgumentType]
    x: SymbolicType = SymbolicType.sym("x", config.n_x, 1)  # pyright: ignore[reportArgumentType]

    f_expression: SymbolicType = x[0]
    f: cas.Function = cas.Function("f", [u, x], [f_expression])

    h_known_expression: SymbolicType = 5 - x[0] ** 2 - x[1] ** 2
    h_known: cas.Function = cas.Function("h_known", [u, x], [h_known_expression])

    h_unkonwn_expression: SymbolicType = (
        x[1] - cas.exp(-((u - 2) ** 2)) - cas.exp(-(u**2) / 3)
    )
    h_unknown: cas.Function = cas.Function("h_unknown", [u, x], [h_unkonwn_expression])

    u_lower_bounds_acquisition: np.ndarray = np.array([[-3]]).T
    u_upper_bounds_acquisition: np.ndarray = np.array([[3]]).T

    x_lower_bounds: np.ndarray = np.array([[-np.inf, -np.inf]]).T
    x_upper_bounds: np.ndarray = np.array([[np.inf, np.inf]]).T

    u_lower_bounds_starting_points: np.ndarray = u_lower_bounds_acquisition.copy()
    u_upper_bounds_starting_points: np.ndarray = u_upper_bounds_acquisition.copy()

    x_lower_bounds_starting_points: np.ndarray = np.array([[0, 0]]).T
    x_upper_bounds_starting_points: np.ndarray = np.array([[np.sqrt(5), np.sqrt(5)]]).T

    x_lower_bounds_evaluation: np.ndarray = x_lower_bounds_starting_points.copy()
    x_upper_bounds_evaluation: np.ndarray = x_upper_bounds_starting_points.copy()

    return Problem(
        f,
        h_known,
        h_unknown,
        u_lower_bounds_acquisition,
        u_upper_bounds_acquisition,
        x_lower_bounds,
        x_upper_bounds,
        u_lower_bounds_starting_points,
        u_upper_bounds_starting_points,
        x_lower_bounds_starting_points,
        x_upper_bounds_starting_points,
        x_lower_bounds_evaluation,
        x_upper_bounds_evaluation,
    )
