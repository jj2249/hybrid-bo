import casadi as cas
import numpy as np

from hybrid_bo import Config, Problem
from hybrid_bo.type_aliases import SymbolicType


def custom_problem(config: Config) -> Problem:
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

    u_lower_bounds: np.ndarray = np.array([-3])[:, np.newaxis]
    u_upper_bounds: np.ndarray = np.array([3])[:, np.newaxis]

    x_lower_bounds: np.ndarray = np.array([0, 0])[:, np.newaxis]
    x_upper_bounds: np.ndarray = np.array([np.sqrt(5), np.sqrt(5)])[:, np.newaxis]

    return Problem(
        f,
        h_known,
        h_unknown,
        u_lower_bounds,
        u_upper_bounds,
        x_lower_bounds,
        x_upper_bounds,
    )
