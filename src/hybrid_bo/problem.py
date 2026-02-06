from typing import final

import casadi as cas
import numpy as np
import scipy
import scipy.optimize

from hybrid_bo.optimizers import MultiStartOptimizer, Optimizer
from hybrid_bo.type_aliases import SymbolicType


class Problem:
    """Class that defines a problem.

    min     f(u, x)

    s.t.    0 = h_known(u, x)
            0 = h_unknown(u, x)

    where
    - f: objective function
    - h_known: known (mechanistic) state equations
    - h_unknown: unknown state equations to be replaced by a GP approximation
        (in simulation they are assumed to be known)

    - u: input (degrees of freedom)
    - x: states
    """

    def __init__(
        self,
        f: cas.Function,
        h_known: cas.Function,
        h_unknown: cas.Function,
        u_lower_bounds_acquisition: np.ndarray,
        u_upper_bounds_acquisition: np.ndarray,
        x_lower_bounds_acquisition: np.ndarray,
        x_upper_bounds_acquisition: np.ndarray,
        u_lower_bounds_starting_points: np.ndarray,
        u_upper_bounds_starting_points: np.ndarray,
        x_lower_bounds_starting_points: np.ndarray,
        x_upper_bounds_starting_points: np.ndarray,
        threshold_equality: float = 1.0e-10,
    ) -> None:
        # %% Attributes

        self.f: cas.Function = f  # Objective function to minimize
        self.h_known: cas.Function = h_known  # Known state equations
        self.h_unknown: cas.Function = h_unknown  # Unknown state equations

        self.n_u: int = self.f.numel_in(0)  # Number of elements in u
        self.n_x: int = self.f.numel_in(1)  # Number of elements in x

        self.n_h_known: int = self.h_known.numel_out(0)  # Number of equations in h_hown
        self.n_h_unknown: int = self.h_unknown.numel_out(0)  # Number of equations in g

        # The following bounds are used
        # - as box constraints in the original problem formulation
        # - as box constraints in the acquisition problem formulation
        self.u_lower_bounds_acquisition: np.ndarray = u_lower_bounds_acquisition.copy()
        self.u_upper_bounds_acquisition: np.ndarray = u_upper_bounds_acquisition.copy()

        # The following bounds are used
        # - as box constraints in the original problem formulation
        # - as box constraints in the acquisition problem formulation
        # - for checking if a solution is valid when evaluating the system at a specific u
        self.x_lower_bounds_acquisition: np.ndarray = x_lower_bounds_acquisition.copy()
        self.x_upper_bounds_acquisition: np.ndarray = x_upper_bounds_acquisition.copy()

        # The following bounds are important when using a method that relies on multiple starting points
        # They can be used to generate starting points
        self.u_lower_bounds_starting_points: np.ndarray = (
            u_lower_bounds_starting_points.copy()
        )
        self.u_upper_bounds_starting_points: np.ndarray = (
            u_upper_bounds_starting_points.copy()
        )

        # The following bounds are important when using a method that relies on multiple starting points
        # They can be used to generate starting points
        self.x_lower_bounds_starting_points: np.ndarray = (
            x_lower_bounds_starting_points.copy()
        )
        self.x_upper_bounds_starting_points: np.ndarray = (
            x_upper_bounds_starting_points.copy()
        )

        self.threshold_equality: float = threshold_equality  # Threshold that defines if two floats are considered equal

        # %% Exceptions

        if self.f.n_in() != 2:
            msg: str = "self.f.n_in() != 2"
            raise Exception(msg)
        if self.f.n_out() != 1:
            msg: str = "self.f.n_out()!= 1"
            raise Exception(msg)

        if self.f.size_in(0) != (self.n_u, 1):
            msg: str = "self.f.size_in(0) != (self.n_u, 1)"
            raise Exception(msg)
        if self.f.size_in(1) != (self.n_x, 1):
            msg: str = "self.f.size_in(1) != (self.n_x, 1)"
            raise Exception(msg)
        if self.f.size_out(0) != (1, 1):
            msg: str = "self.f.size_out(0) != (1, 1)"
            raise Exception(msg)

        if self.h_known.n_in() != 2:  # noqa: PLR2004
            msg: str = "self.h_known.n_in() != 2"
            raise Exception(msg)
        if self.h_known.n_out() != 1:
            msg: str = "self.h_known.n_out()!= 1"
            raise Exception(msg)

        if self.h_known.size_in(0) != (self.n_u, 1):
            msg: str = "self.h_known.size_in(0) != (self.nu, 1)"
            raise Exception(msg)
        if self.h_known.size_in(1) != (self.n_x, 1):
            msg: str = "self.h_known.size_in(1) != (self.n_x, 1)"
            raise Exception(msg)
        if self.h_known.size_out(0) != (self.n_h_known, 1):
            msg: str = "self.h_known.size_out(0) != (self.n_h, 1)"
            raise Exception(msg)

        if self.h_unknown.n_in() != 2:  # noqa: PLR2004
            msg: str = "self.h_unknown.n_in() != 2"
            raise Exception(msg)
        if self.h_unknown.n_out() != 1:
            msg: str = "self.h_unknown.n_out()!= 1"
            raise Exception(msg)

        if self.h_unknown.size_in(0) != (self.n_u, 1):
            msg: str = "self.h_unknown.size_in(0) != (self.nu, 1)"
            raise Exception(msg)
        if self.h_unknown.size_in(1) != (self.n_x, 1):
            msg: str = "self.h_unknown.size_in(1) != (self.n_x, 1)"
            raise Exception(msg)
        if self.h_unknown.size_out(0) != (self.n_h_unknown, 1):
            msg: str = "self.h_unknown.size_out(0) != (self.n_g, 1)"
            raise Exception(msg)

        if self.u_lower_bounds_acquisition.shape != (self.n_u, 1):
            msg: str = "self.u_lower_bounds.shape != (self.n_u, 1)"
            raise Exception(msg)
        if self.u_upper_bounds_acquisition.shape != (self.n_u, 1):
            msg: str = "self.u_upper_bounds.shape != (self.n_u, 1)"
            raise Exception(msg)

        if self.x_lower_bounds_acquisition.shape != (self.n_x, 1):
            msg: str = "self.x_lower_bounds.shape != (self.x_u, 1)"
            raise Exception(msg)
        if self.x_upper_bounds_acquisition.shape != (self.n_x, 1):
            msg: str = "self.x_upper_bounds.shape != (self.n_x, 1)"
            raise Exception(msg)

        if self.u_lower_bounds_starting_points.shape != (self.n_u, 1):
            msg: str = "self.u_lower_bounds_starting_points.shape != (self.n_u, 1)"
            raise Exception(msg)
        if self.u_upper_bounds_starting_points.shape != (self.n_u, 1):
            msg: str = "self.u_upper_bounds_starting_points.shape != (self.n_u, 1)"
            raise Exception(msg)

        if self.x_lower_bounds_starting_points.shape != (self.n_x, 1):
            msg: str = "self.x_lower_bounds_starting_points.shape != (self.x_u, 1)"
            raise Exception(msg)
        if self.x_upper_bounds_starting_points.shape != (self.n_x, 1):
            msg: str = "self.x_upper_bounds_starting_points.shape != (self.n_x, 1)"
            raise Exception(msg)

        if not np.isfinite(self.u_lower_bounds_starting_points).all():
            msg: str = "not np.isfinite(self.u_lower_bounds_starting_points).all()"
            raise Exception(msg)
        if not np.isfinite(self.u_upper_bounds_starting_points).all():
            msg: str = "not np.isfinite(self.u_upper_bounds_starting_points).all()"
            raise Exception(msg)

        if not np.isfinite(self.x_lower_bounds_starting_points).all():
            msg: str = "not np.isfinite(self.x_lower_bounds_starting_points).all()"
            raise Exception(msg)
        if not np.isfinite(self.x_upper_bounds_starting_points).all():
            msg: str = "not np.isfinite(self.x_upper_bounds_starting_points).all()"
            raise Exception(msg)

    @final
    def single_evaluate_with_simulation_original(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        """Returns f and x from u using h_known and h_unknown. No measurement noise is considered."""

        # %% Setup variables to be solved for

        x: SymbolicType = SymbolicType.sym("x", self.n_x, 1)  # pyright: ignore[reportArgumentType]

        # %% Get initial guesses for x

        if rng is None:
            rng = np.random.default_rng()

        sampler: scipy.stats.qmc.LatinHypercube = scipy.stats.qmc.LatinHypercube(
            self.n_x,
            rng=rng,
        )

        samples: np.ndarray = sampler.random(n_starts_max)
        X_0: np.ndarray = (
            self.x_lower_bounds_starting_points.T
            + (
                self.x_upper_bounds_starting_points
                - self.x_lower_bounds_starting_points
            ).T
            * samples
        )

        # %% Setup equations to be solved

        h: cas.Function = cas.Function(
            "h",
            [x],
            [cas.vertcat(self.h_known(u, x), self.h_unknown(u, x))],
        )

        h_jacobian: cas.Function = cas.Function(
            "h_jacobian",
            [x],
            [cas.jacobian(h(x), x)],
        )

        def h_np(x: np.ndarray) -> np.ndarray:
            return h(x[:, np.newaxis]).full().flatten()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        def h_np_jacobian(x: np.ndarray) -> np.ndarray:
            return h_jacobian(x[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # %% Solve for x and get corresponding values of f

        i_start: int = 0
        ready_to_return: bool = False
        while i_start < n_starts_max:
            if use_jacobian:
                roots: np.ndarray = scipy.optimize.fsolve(
                    h_np,
                    X_0[i_start, :],
                    fprime=h_np_jacobian,
                )[
                    :,
                    np.newaxis,
                ]  # pyright: ignore[reportArgumentType, reportCallIssue]
            else:
                roots: np.ndarray = scipy.optimize.fsolve(h_np, X_0[i_start, :])[
                    :,
                    np.newaxis,
                ]  # pyright: ignore[reportArgumentType, reportCallIssue]

            if (
                (np.linalg.norm(h_np(roots.ravel())) < self.threshold_equality)
                and ((roots >= self.x_lower_bounds_acquisition).all())
                and ((roots <= self.x_upper_bounds_acquisition).all())
            ):
                ready_to_return: bool = True

            if ready_to_return:
                f_at_roots: float = self.f(u, roots).full().item()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
                return f_at_roots, roots

            i_start = i_start + 1

        msg: str = "Could not get a solution!"
        raise Exception(msg)

    def single_evaluate_with_simulation(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        return self.single_evaluate_with_simulation_original(
            u,
            n_starts_max,
            use_jacobian,
            rng,
        )

    @final
    def single_evaluate_with_fixed_x_original(
        self,
        u: np.ndarray,
        x_fixed: np.ndarray,
        indices_x_fixed: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        """Returns f and x from u and fixed parts of x using h_known. The fixed part of x is either measured or an output from a GP."""

        if x_fixed.shape != (len(indices_x_fixed), 1):
            msg: str = "x_fixed.shape != (len(indices_x_fixed), 1)"
            raise Exception(msg)

        if x_fixed.shape != (self.n_h_unknown, 1):
            msg: str = "x_fixed.shape != (self.n_h_unknown, 1)"
            raise Exception(msg)

        # %% Get the indices of the elements of x that are not measured

        mask_x_free: np.ndarray = np.ones((self.n_x,), dtype=bool)
        mask_x_free[indices_x_fixed] = False
        indices_x_free: list[int] = np.nonzero(mask_x_free)[0].tolist()

        # %% Setup variables to be solved for

        x_free = SymbolicType.sym("x_free", len(indices_x_free), 1)  # pyright: ignore[reportArgumentType]

        # %% Setup helper variables

        x: SymbolicType = SymbolicType(self.n_x, 1)
        x[indices_x_fixed] = x_fixed
        x[indices_x_free] = x_free

        # %% Get starting points for x_free

        if rng is None:
            rng = np.random.default_rng()

        sampler: scipy.stats.qmc.LatinHypercube = scipy.stats.qmc.LatinHypercube(
            x_fixed.size,
            rng=rng,
        )

        samples: np.ndarray = sampler.random(n_starts_max)
        X_free_0: np.ndarray = (
            self.x_lower_bounds_starting_points[indices_x_free].T
            + (
                self.x_upper_bounds_starting_points[indices_x_free]
                - self.x_lower_bounds_starting_points[indices_x_free]
            ).T
            * samples
        )

        # %% Setup equations to be solved

        h: cas.Function = cas.Function("h", [x_free], [self.h_known(u, x)])
        h_jacobian: cas.Function = cas.Function(
            "h_jacobian",
            [x_free],
            [cas.jacobian(h(x_free), x_free)],
        )

        def h_np(x: np.ndarray) -> np.ndarray:
            return h(x[:, np.newaxis]).full().flatten()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        def h_np_jacobian(x: np.ndarray) -> np.ndarray:
            return h_jacobian(x[:, np.newaxis]).full()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # %% Solve for x_free and get corresponding values of f

        i_start: int = 0
        ready_to_return: bool = False
        while i_start < n_starts_max:
            if use_jacobian:
                roots: np.ndarray = scipy.optimize.fsolve(
                    h_np,
                    X_free_0[i_start, :],
                    fprime=h_np_jacobian,
                )[:, np.newaxis]  # pyright: ignore[reportArgumentType, reportCallIssue]
            else:
                roots: np.ndarray = scipy.optimize.fsolve(h_np, X_free_0[i_start, :])[
                    :,
                    np.newaxis,
                ]  # pyright: ignore[reportArgumentType, reportCallIssue]

            if (
                (np.linalg.norm(h_np(roots.ravel())) < self.threshold_equality)
                and ((roots >= self.x_lower_bounds_acquisition[indices_x_free]).all())
                and ((roots <= self.x_upper_bounds_acquisition[indices_x_free]).all())
            ):
                ready_to_return: bool = True

            if ready_to_return:
                x_solution: np.ndarray = np.empty((self.n_x, 1))
                x_solution[indices_x_fixed] = x_fixed
                x_solution[indices_x_free] = roots
                f_at_solution: float = self.f(u, x_solution).full().item()  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

                return f_at_solution, x_solution

            i_start = i_start + 1

        msg: str = "Could not get a solution!"
        raise Exception(msg)

    def single_evaluate_with_fixed_x(
        self,
        u: np.ndarray,
        x_fixed: np.ndarray,
        indices_x_fixed: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        return self.single_evaluate_with_fixed_x_original(
            u,
            x_fixed,
            indices_x_fixed,
            n_starts_max,
            use_jacobian,
            rng,
        )

    @final
    def single_evaluate_with_noisy_simulation_original(
        self,
        u: np.ndarray,
        measurement_noise: np.ndarray,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Returns f and x (with and without consideration of measurement noise) from u using h_known and h_unknown."""

        f_no_noise: float
        x_no_noise: np.ndarray
        f_no_noise, x_no_noise = self.single_evaluate_with_simulation(
            u,
            n_starts_max,
            use_jacobian,
            rng,
        )

        f: float
        x: np.ndarray
        if (measurement_noise != 0.0).any():
            x_measured: np.ndarray = x_no_noise[indices_x_measured] + measurement_noise

            f, x = self.single_evaluate_with_fixed_x(
                u,
                x_measured,
                indices_x_measured,
                n_starts_max,
                use_jacobian,
                rng,
            )

        else:
            f = f_no_noise
            x = x_no_noise

        return f, x, f_no_noise, x_no_noise

    def single_evaluate_with_noisy_simulation(
        self,
        u: np.ndarray,
        measurement_noise: np.ndarray,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        return self.single_evaluate_with_noisy_simulation_original(
            u,
            measurement_noise,
            indices_x_measured,
            n_starts_max,
            use_jacobian,
            rng,
        )

    def solve(
        self,
        optimizer: Optimizer,
        n_starting_points: int = 10,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float | np.ndarray]:
        # Parameters n_starting_points and rng are only important if type(optimizer) == MultistartOptimizer

        u: SymbolicType = SymbolicType.sym("u", self.n_u, 1)  # pyright: ignore[reportArgumentType]
        x: SymbolicType = SymbolicType.sym("x", self.n_x, 1)  # pyright: ignore[reportArgumentType]
        w: SymbolicType = cas.vertcat(u, x)  # pyright: ignore[reportAssignmentType]

        n_variables: int = self.n_u + self.n_x

        h_expression: SymbolicType = cas.vertcat(
            self.h_known(u, x),
            self.h_unknown(u, x),
        )  # pyright: ignore[reportAssignmentType]
        h: cas.Function = cas.Function("h", [w], [h_expression])

        lower_bounds: np.ndarray = np.vstack(
            (self.u_lower_bounds_acquisition, self.x_lower_bounds_acquisition),
        )
        upper_bounds: np.ndarray = np.vstack(
            (self.u_upper_bounds_acquisition, self.x_upper_bounds_acquisition),
        )

        f_expression: SymbolicType = self.f(u, x)  # pyright: ignore[reportAssignmentType]
        f: cas.Function = cas.Function("f", [w], [f_expression])

        optimizer.set_problem(n_variables, f, None, h, lower_bounds, upper_bounds)

        if isinstance(optimizer, MultiStartOptimizer):
            optimizer.lower_bounds_starting_points = np.vstack(
                (
                    self.u_lower_bounds_starting_points,
                    self.x_lower_bounds_starting_points,
                ),
            )
            optimizer.upper_bounds_starting_points = np.vstack(
                (
                    self.u_upper_bounds_starting_points,
                    self.x_upper_bounds_starting_points,
                ),
            )

            starting_points: np.ndarray = optimizer.create_lhs_samples(
                n_starting_points,
                rng,
            )
            optimizer.X0 = starting_points

        solution: dict[str, float | np.ndarray] | None = optimizer.solve()

        if solution is None:
            msg: str = "solution is None"
            raise Exception(msg)

        solution_formatted: dict[str, float | np.ndarray] = {}
        solution_formatted["f"] = solution["f"]
        solution_formatted["u"] = solution["x"][: self.n_u]  # pyright: ignore[reportIndexIssue]
        solution_formatted["x"] = solution["x"][self.n_u :]  # pyright: ignore[reportIndexIssue]

        return solution_formatted
