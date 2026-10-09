import casadi as cas
import numpy as np

from hybrid_bo import Config, Problem
from hybrid_bo.type_aliases import SymbolicType


def get_custom_problem(config: Config) -> Problem:
    u = SymbolicType.sym("u", config.n_u, 1)
    x = SymbolicType.sym("x", config.n_x, 1)

    # Objective and constraint functions
    f = cas.Function("f", [u, x], [(6 * x[0] - 2) ** 2 * cas.sin(12 * x[0] - 4)])
    h_known = cas.Function("h_known", [u, x], [x[0] + cas.exp(x[0]) - x[1]])
    h_unknown = cas.Function("h_unknown", [u, x], [cas.sin(u) - x[1]])

    # Acquisition bounds (lower, upper)
    u_acq = np.array([[-2.0]]), np.array([[2.0]])
    x_acq = np.array([[-np.inf], [-np.inf]]), np.array([[np.inf], [np.inf]])

    # Starting point bounds (lower, upper)
    u_sp = u_acq
    x_sp = np.array([[-2.5], [-2.0]]), np.array([[1.0], [2.0]])

    return Problem(
        f,
        h_known,
        h_unknown,
        *u_acq,
        *x_acq,
        *u_sp,
        *x_sp,
    )
