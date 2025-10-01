import casadi as cas
import numpy as np

from hybrid_bo import Problem, SymbolicType


def custom_problem() -> Problem:
    n_u: int = 1
    n_x: int = 2

    u: SymbolicType = SymbolicType.sym("u", n_u, 1)  # pyright: ignore[reportArgumentType]
    x: SymbolicType = SymbolicType.sym("x", n_x, 1)  # pyright: ignore[reportArgumentType]

    f_expression: SymbolicType = x[0] * x[1]
    f: cas.Function = cas.Function("f", [u, x], [f_expression])

    h_known_expression: SymbolicType = x[0] - cas.sin(12 * u - 4)
    h_known: cas.Function = cas.Function("h_known", [u, x], [h_known_expression])

    h_unkonwn_expression: SymbolicType = x[1] - (6 * u - 2) ** 2
    h_unknown: cas.Function = cas.Function("h_unknown", [u, x], [h_unkonwn_expression])

    u_lower_bounds: np.ndarray = np.array([0])[:, np.newaxis]
    u_upper_bounds: np.ndarray = np.array([1])[:, np.newaxis]

    x_lower_bounds: np.ndarray = np.array([-1, 0])[:, np.newaxis]
    x_upper_bounds: np.ndarray = np.array([1, 16])[:, np.newaxis]

    return Problem(
        f,
        h_known,
        h_unknown,
        u_lower_bounds,
        u_upper_bounds,
        x_lower_bounds,
        x_upper_bounds,
    )
