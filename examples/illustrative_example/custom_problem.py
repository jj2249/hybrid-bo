import casadi as cas

from hybrid_bo import Bounds, Config, Problem
from hybrid_bo.type_aliases import SymbolicType


def get_custom_problem(config: Config) -> Problem:
    u = SymbolicType.sym("u", config.n_u, 1)
    x = SymbolicType.sym("x", config.n_x, 1)

    # Objective and constraint functions
    f = cas.Function("f", [u, x], [(6 * x[0] - 2) ** 2 * cas.sin(12 * x[0] - 4)])
    h_known = cas.Function("h_known", [u, x], [x[0] + cas.exp(x[0]) - x[1]])
    h_unknown = cas.Function("h_unknown", [u, x], [cas.sin(u) - x[1]])

    u_bounds = Bounds([-2.0], [2.0])

    return Problem(
        f,
        h_known,
        h_unknown,
        u_bounds=u_bounds,
        x_bounds=Bounds.unbounded(2),
        u_start_bounds=u_bounds,
        x_start_bounds=Bounds([-2.5, -2.0], [1.0, 2.0]),
    )
