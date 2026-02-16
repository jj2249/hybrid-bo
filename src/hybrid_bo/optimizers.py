import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, override
from dataclasses import dataclass

import casadi as cas
import numpy as np
import scipy

from hybrid_bo.type_aliases import SymbolicType


@dataclass
class OptimizationResult:
    x: np.ndarray
    f: float
    g: np.ndarray | None = None
    h: np.ndarray | None = None


class Optimizer(ABC):
    def __init__(self) -> None:
        self.n_variables: int = 0
        self.f: cas.Function = cas.Function()  # Objective function: min f(x)
        self.g: cas.Function | None = None  # Inequality constraints function: g(x) <= 0
        self.h: cas.Function | None = None  # Equality constraints function: h(x) = 0
        self.lower_bounds: np.ndarray | None = None
        self.upper_bounds: np.ndarray | None = None

    def set_problem(
        self,
        n_variables: int,
        f: cas.Function,
        g: cas.Function | None = None,
        h: cas.Function | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> None:
        self.n_variables = n_variables
        self.f = f  # Objective function: min f(x)
        self.g = g  # Inequality constraints function: g(x) <= 0
        self.h = h  # Equality constraints function: h(x) = 0

        # Lower bounds of optimization variables: x >= lower_bounds
        self.lower_bounds = lower_bounds
        # Upper bounds of optimization variables: x <= upper_bounds
        self.upper_bounds = upper_bounds

        if (lower_bounds is not None) and (lower_bounds.shape != (self.n_variables, 1)):
            msg: str = "lower_bounds.shape != (self.n_variables, 1)"
            raise Exception(msg)

        if (upper_bounds is not None) and (upper_bounds.shape != (self.n_variables, 1)):
            msg: str = "upper_bounds.shape != (self.n_variables, 1)"
            raise Exception(msg)

    @abstractmethod
    def solve(self) -> OptimizationResult | None: ...

    def __getstate__(self) -> dict:
        # Required for pickling to be possible
        state: dict = self.__dict__.copy()
        state["f"] = cas.Function()
        state["g"] = None
        state["h"] = None
        return state


class LocalOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()
        self.x0: np.ndarray = np.empty([0])

    def set_x0(self, x0: np.ndarray) -> None:
        if x0.shape[1] != 1:
            msg: str = "x0.shape[1] != 1"
            raise Exception(msg)
        self.x0 = x0


class MultiStartOptimizer(Optimizer):
    def __init__(
        self,
        local_optimizer: LocalOptimizer,
        X0: np.ndarray | None = None,
        lower_bounds_starting_points: np.ndarray | None = None,
        upper_bounds_starting_points: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        # %% Attributes

        self.local_optimizer: LocalOptimizer = local_optimizer
        self._X0: np.ndarray
        self.lower_bounds_starting_points: np.ndarray | None = (
            lower_bounds_starting_points
        )
        self.upper_bounds_starting_points: np.ndarray | None = (
            upper_bounds_starting_points
        )

        # %% Non-trivial assignments
        if X0 is None:
            self._X0 = np.empty([0])
            self.n_starts = 0
        else:
            self.X0 = X0

    @property
    def X0(self) -> np.ndarray:
        return self._X0

    @X0.setter
    def X0(self, X0: np.ndarray) -> None:
        if len(X0.shape) != 2:
            msg: str = "len(X0.shape) != 2"
            raise Exception(msg)

        if X0.shape[1] != self.n_variables:
            msg: str = "X0.shape[1] != self.n_variables"
            raise Exception(msg)

        if not np.isfinite(X0).all():
            msg: str = "not np.isfinite(X0).all()"
            raise Exception(msg)

        self._X0 = X0
        self.n_starts = self._X0.shape[0]

    @override
    def set_problem(
        self,
        n_variables: int,
        f: cas.Function,
        g: cas.Function | None = None,
        h: cas.Function | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> None:
        super().set_problem(n_variables, f, g, h, lower_bounds, upper_bounds)
        self.local_optimizer.set_problem(
            self.n_variables,
            self.f,
            self.g,
            self.h,
            self.lower_bounds,
            self.upper_bounds,
        )

    def create_lhs_samples(
        self,
        n_starts: int = 10,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        # lhs: latin hypercube sampling

        if self.lower_bounds_starting_points is None:
            msg: str = "self.lower_bounds_starting_points is None"
            raise Exception(msg)
        if self.upper_bounds_starting_points is None:
            msg: str = "self.upper_bounds_starting_points is None"

        if rng is None:
            rng = np.random.default_rng()

        sampler: scipy.stats.qmc.LatinHypercube = scipy.stats.qmc.LatinHypercube(
            self.n_variables,
            rng=rng,
        )

        samples: np.ndarray = sampler.random(n_starts)

        return (
            self.lower_bounds_starting_points.T
            + (self.upper_bounds_starting_points - self.lower_bounds_starting_points).T
            * samples
        )

    @override
    def solve(self) -> OptimizationResult | None:
        if self.n_starts < 1:
            msg: str = "self.n_starts() < 1"
            raise Exception(msg)

        solution: OptimizationResult | None = None
        f_min = np.inf

        for i_guess in range(self.n_starts):
            self.local_optimizer.set_x0(self._X0[i_guess, :][:, np.newaxis])

            current_solution: OptimizationResult | None = self.local_optimizer.solve()

            if (current_solution is not None) and (current_solution.f < f_min):
                solution = current_solution
                f_min = current_solution.f

        return solution


class CasadiOptimizer(LocalOptimizer):
    def __init__(
        self,
        solver: str = "ipopt",
        problem_type: str = "nonlinear",
        plugin_options: dict[str, Any] | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.solver: str = solver
        self.type: str = problem_type  # Options: "nonlinear" and "conic"
        self.opti: cas.Opti
        self.plugin_options: dict[str, Any]
        self.solver_options: dict[str, Any]

        if self.type == "conic":
            self.opti = cas.Opti("conic")
        else:
            self.opti = cas.Opti()

        if plugin_options is None:
            self.plugin_options = {}
        else:
            self.plugin_options = plugin_options

        if solver_options is None:
            self.solver_options = {}
        else:
            self.solver_options = solver_options

    @override
    def set_problem(
        self,
        n_variables: int,
        f: cas.Function,
        g: cas.Function | None = None,
        h: cas.Function | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> None:
        super().set_problem(n_variables, f, g, h, lower_bounds, upper_bounds)

        if self.type == "conic":
            self.opti = cas.Opti("conic")
        else:
            self.opti = cas.Opti()

        x: cas.MX = self.opti.variable(self.n_variables)

        self.opti.minimize(self.f(x))

        if self.g is not None:
            self.opti.subject_to(self.g(x) <= 0)  # pyright: ignore[reportOperatorIssue]

        if self.h is not None:
            self.opti.subject_to(self.h(x) == 0)

        if self.lower_bounds is not None:
            self.opti.subject_to(self.lower_bounds <= x)

        if self.upper_bounds is not None:
            self.opti.subject_to(x <= self.upper_bounds)

    @override
    def set_x0(self, x0: np.ndarray) -> None:
        super().set_x0(x0)
        self.opti.set_initial(self.opti.x, x0)

    @override
    def solve(self) -> OptimizationResult | None:
        try:
            self.opti.solver(self.solver, self.plugin_options, self.solver_options)

            solution: cas.OptiSol = self.opti.solve()

            x_opt: np.ndarray
            x_opt_temp: np.ndarray | float = solution.value(self.opti.x)

            if isinstance(x_opt_temp, float):
                x_opt = np.array([[x_opt_temp]])
            elif isinstance(x_opt_temp, np.ndarray):
                x_opt = x_opt_temp[:, np.newaxis]
            else:
                return None

            f_opt: float = solution.value(self.opti.f)
            g_opt: np.ndarray | None = None
            if self.g is not None:
                g_opt = self.g(x_opt).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            h_opt: np.ndarray | None = None
            if self.h is not None:
                h_opt = self.h(x_opt).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return OptimizationResult(x_opt, f_opt, g_opt, h_opt)

    def __getstate__(self) -> dict:
        state: dict = super().__getstate__()
        state["opti"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.type == "conic":
            self.opti = cas.Opti("conic")
        else:
            self.opti = cas.Opti()


class SciPyLocalOptimizer(LocalOptimizer):
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    def __init__(
        self,
        method: str = "BFGS",
        tolerance: float | None = None,
        callback: Callable[[scipy.optimize.OptimizeResult], Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.method: str = method
        self.tolerance: float | None = tolerance
        self.options: dict[str, Any] | None = options
        self.callback: Callable[[scipy.optimize.OptimizeResult], Any] | None = callback

        self.f_np: Callable[[np.ndarray], float]
        self.f_gradient_np: Callable[[np.ndarray], np.ndarray] | None
        self.f_hessian_np: Callable[[np.ndarray], np.ndarray] | None
        self.bounds: scipy.optimize.Bounds | None
        self.constraints: list[scipy.optimize.NonlinearConstraint | dict[str, Any]]

        self.gradient_methods: tuple[str, ...] = (
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "SLSQP",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
            "trust-constr",
        )

        self.hessian_methods: tuple[str, ...] = (
            "Newton-CG",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
            "trust-constr",
        )

        self.bounds_methods: tuple[str, ...] = (
            "Nelder-Mead",
            "L-BFGS-B",
            "TNC",
            "SLSQP",
            "Powell",
            "trust-constr",
            "COBYLA",
            "COBYQA",
        )

        self.constraints_methods: tuple[str, ...] = (
            "trust-constr",
            "COBYQA",
            "COBYLA",
            "SLSQP",
        )

    @override
    def set_problem(
        self,
        n_variables: int,
        f: cas.Function,
        g: cas.Function | None = None,
        h: cas.Function | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> None:
        super().set_problem(n_variables, f, g, h, lower_bounds, upper_bounds)

        x: SymbolicType = SymbolicType.sym("x", n_variables, 1)  # pyright: ignore[reportArgumentType]

        # %% Set self.f_np

        self.f_np = lambda x: self.f(x[:, np.newaxis]).full().item()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # %% Set self.f_gradient_np

        if self.method in self.gradient_methods:
            f_gradient_expression: SymbolicType = cas.gradient(self.f(x), x)
            f_gradient: cas.Function = cas.Function(
                "gradient",
                [x],
                [f_gradient_expression],
            )
            self.f_gradient_np = lambda x: f_gradient(x[:, np.newaxis]).full().flatten()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        else:
            self.f_gradient_np = None

        # %% Set self.f_hessian_np

        if self.method in self.hessian_methods:
            f_hessian_expression: SymbolicType
            f_hessian_expression, _ = cas.hessian(self.f(x), x)
            f_hessian: cas.Function = cas.Function(
                "hessian",
                [x],
                [f_hessian_expression],
            )
            self.f_hessian_np = lambda x: f_hessian(x[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        else:
            self.f_hessian_np = None

        # %% Set self.bounds

        if self.method in self.bounds_methods:
            lb: float | np.ndarray = (
                -np.inf if self.lower_bounds is None else self.lower_bounds[:, 0]
            )

            ub: float | np.ndarray = (
                np.inf if self.upper_bounds is None else self.upper_bounds[:, 0]
            )

            self.bounds = scipy.optimize.Bounds(lb, ub)

        else:
            self.bounds = None

        # Set %% self.constraints

        self.constraints = []
        if self.method in self.constraints_methods:
            # %% Equality constraints

            # COBYLA does not support equality constraints
            if (self.h is not None) and (self.method != "COBYLA"):

                def h_np(x: np.ndarray) -> np.ndarray:
                    return self.h(x[:, np.newaxis]).full().flatten()  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue, reportOptionalMemberAccess]

                h_jacobian_expression: SymbolicType = cas.jacobian(self.h(x), x)
                h_jacobian: cas.Function = cas.Function(
                    "h_jacobian",
                    [x],
                    [h_jacobian_expression],
                )

                def h_jacobian_np(x: np.ndarray) -> np.ndarray:
                    return h_jacobian(x[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

                h_constraints: scipy.optimize.NonlinearConstraint | dict[str, Any]
                if self.method in ("trust-constr", "COBYQA"):
                    v: SymbolicType = SymbolicType.sym("v", self.h.numel_out(0), 1)  # pyright: ignore[reportArgumentType]
                    h_v_hessian_expression: SymbolicType
                    h_v_hessian_expression, _ = cas.hessian(v.T @ self.h(x), x)
                    h_v_hessian: cas.Function = cas.Function(
                        "h_v_hessian",
                        [x, v],
                        [h_v_hessian_expression],
                    )

                    def h_v_hessian_np(x: np.ndarray, v: np.ndarray) -> np.ndarray:
                        return h_v_hessian(x[:, np.newaxis], v[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

                    # Type: scipy.optimize.NonlinearConstraint
                    h_constraints = scipy.optimize.NonlinearConstraint(
                        h_np,
                        0.0,
                        0.0,
                        h_jacobian_np,
                        h_v_hessian_np,
                    )

                else:  # self.method == "SLSQP"
                    # Type: dict[str, Any]
                    h_constraints = {"type": "eq", "fun": h_np, "jac": h_jacobian_np}

                self.constraints.append(h_constraints)

            # %% Inequality constraints

            if self.g is not None:

                def g_np(x: np.ndarray) -> np.ndarray:
                    return self.g(x[:, np.newaxis]).full().flatten()  # pyright: ignore[reportOptionalCall, reportAttributeAccessIssue, reportOptionalMemberAccess]

                g_jacobian_expression: SymbolicType = cas.jacobian(self.g(x), x)
                g_jacobian: cas.Function = cas.Function(
                    "g_jacobian", [x], [g_jacobian_expression]
                )

                def g_jacobian_np(x: np.ndarray) -> np.ndarray:
                    return g_jacobian(x[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

                g_constraints: scipy.optimize.NonlinearConstraint | dict[str, Any]
                if self.method in ("trust-constr", "COBYQA"):
                    v: SymbolicType = SymbolicType.sym("v", self.g.numel_out(0), 1)  # pyright: ignore[reportArgumentType]
                    g_v_hessian_expression: SymbolicType
                    g_v_hessian_expression, _ = cas.hessian(v.T @ self.g(x), x)

                    g_v_hessian: cas.Function = cas.Function(
                        "g_v_hessian",
                        [x, v],
                        [g_v_hessian_expression],
                    )

                    def g_v_hessian_np(x: np.ndarray, v: np.ndarray) -> np.ndarray:
                        return g_v_hessian(x[:, np.newaxis], v[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

                    g_constraints = scipy.optimize.NonlinearConstraint(
                        g_np,
                        -np.inf,
                        0,
                        g_jacobian_np,
                        g_v_hessian_np,
                    )

                else:  # self.method in ("SLSQP", "COBYLA")
                    g_constraints = {
                        "type": "ineq",
                        "fun": lambda x: -g_np(x),
                        "jac": lambda x: -g_jacobian_np(x),
                    }  # Negative sign because of non-negativity

                self.constraints.append(g_constraints)

    @override
    def solve(self) -> OptimizationResult | None:
        try:
            solution: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
                self.f_np,
                self.x0[:, 0],
                method=self.method,
                jac=self.f_gradient_np,
                hess=self.f_hessian_np,
                bounds=self.bounds,
                constraints=self.constraints,
                tol=self.tolerance,
                callback=self.callback,
                options=self.options,
            )

            x_opt: np.ndarray = solution.x[:, np.newaxis]
            f_opt: float = solution.fun

            g_opt: np.ndarray | None = None
            if self.g is not None:
                g_opt = self.g(x_opt).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            h_opt: np.ndarray = np.empty([0])
            if self.h is not None:
                h_opt = self.h(x_opt).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        except Exception as e:
            traceback.print_exc()
            return None

        else:
            return OptimizationResult(x_opt, f_opt, g_opt, h_opt)
