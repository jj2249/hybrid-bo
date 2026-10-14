import casadi as cas
import numpy as np
import scipy
import scipy.optimize

from hybrid_bo.sampling import lhs
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

        # Number of equations in h_known and h_unknown
        self.n_h_known: int = self.h_known.numel_out(0)
        self.n_h_unknown: int = self.h_unknown.numel_out(0)

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
            msg = "self.f.n_in() != 2"
            raise Exception(msg)
        if self.f.n_out() != 1:
            msg = "self.f.n_out() != 1"
            raise Exception(msg)

        if self.f.size_in(0) != (self.n_u, 1):
            msg = "self.f.size_in(0) != (self.n_u, 1)"
            raise Exception(msg)
        if self.f.size_in(1) != (self.n_x, 1):
            msg = "self.f.size_in(1) != (self.n_x, 1)"
            raise Exception(msg)
        if self.f.size_out(0) != (1, 1):
            msg = "self.f.size_out(0) != (1, 1)"
            raise Exception(msg)

        if self.h_known.n_in() != 2:
            msg = "self.h_known.n_in() != 2"
            raise Exception(msg)
        if self.h_known.n_out() != 1:
            msg = "self.h_known.n_out() != 1"
            raise Exception(msg)

        if self.h_known.size_in(0) != (self.n_u, 1):
            msg = "self.h_known.size_in(0) != (self.n_u, 1)"
            raise Exception(msg)
        if self.h_known.size_in(1) != (self.n_x, 1):
            msg = "self.h_known.size_in(1) != (self.n_x, 1)"
            raise Exception(msg)
        if self.h_known.size_out(0) != (self.n_h_known, 1):
            msg = "self.h_known.size_out(0) != (self.n_h_known, 1)"
            raise Exception(msg)

        if self.h_unknown.n_in() != 2:
            msg = "self.h_unknown.n_in() != 2"
            raise Exception(msg)
        if self.h_unknown.n_out() != 1:
            msg = "self.h_unknown.n_out() != 1"
            raise Exception(msg)

        if self.h_unknown.size_in(0) != (self.n_u, 1):
            msg = "self.h_unknown.size_in(0) != (self.n_u, 1)"
            raise Exception(msg)
        if self.h_unknown.size_in(1) != (self.n_x, 1):
            msg = "self.h_unknown.size_in(1) != (self.n_x, 1)"
            raise Exception(msg)
        if self.h_unknown.size_out(0) != (self.n_h_unknown, 1):
            msg = "self.h_unknown.size_out(0) != (self.n_h_unknown, 1)"
            raise Exception(msg)

        if self.u_lower_bounds_acquisition.shape != (self.n_u, 1):
            msg = "self.u_lower_bounds_acquisition.shape != (self.n_u, 1)"
            raise Exception(msg)
        if self.u_upper_bounds_acquisition.shape != (self.n_u, 1):
            msg = "self.u_upper_bounds_acquisition.shape != (self.n_u, 1)"
            raise Exception(msg)

        if self.x_lower_bounds_acquisition.shape != (self.n_x, 1):
            msg = "self.x_lower_bounds_acquisition.shape != (self.n_x, 1)"
            raise Exception(msg)
        if self.x_upper_bounds_acquisition.shape != (self.n_x, 1):
            msg = "self.x_upper_bounds_acquisition.shape != (self.n_x, 1)"
            raise Exception(msg)

        if self.u_lower_bounds_starting_points.shape != (self.n_u, 1):
            msg = "self.u_lower_bounds_starting_points.shape != (self.n_u, 1)"
            raise Exception(msg)
        if self.u_upper_bounds_starting_points.shape != (self.n_u, 1):
            msg = "self.u_upper_bounds_starting_points.shape != (self.n_u, 1)"
            raise Exception(msg)

        if self.x_lower_bounds_starting_points.shape != (self.n_x, 1):
            msg = "self.x_lower_bounds_starting_points.shape != (self.n_x, 1)"
            raise Exception(msg)
        if self.x_upper_bounds_starting_points.shape != (self.n_x, 1):
            msg = "self.x_upper_bounds_starting_points.shape != (self.n_x, 1)"
            raise Exception(msg)

        if not np.isfinite(self.u_lower_bounds_starting_points).all():
            msg = "not np.isfinite(self.u_lower_bounds_starting_points).all()"
            raise Exception(msg)
        if not np.isfinite(self.u_upper_bounds_starting_points).all():
            msg = "not np.isfinite(self.u_upper_bounds_starting_points).all()"
            raise Exception(msg)

        if not np.isfinite(self.x_lower_bounds_starting_points).all():
            msg = "not np.isfinite(self.x_lower_bounds_starting_points).all()"
            raise Exception(msg)
        if not np.isfinite(self.x_upper_bounds_starting_points).all():
            msg = "not np.isfinite(self.x_upper_bounds_starting_points).all()"
            raise Exception(msg)

    def evaluate_with_simulation(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
        original: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if u.shape[1] != self.n_u:
            msg = "u.shape[1] != self.n_u"
            raise Exception(msg)

        n_points = u.shape[0]

        f = np.empty((n_points, 1))
        x = np.empty((n_points, self.n_x))

        for i_point in range(n_points):
            if original:
                (
                    f_current,
                    x_current,
                ) = self.single_evaluate_with_simulation_original(
                    u[[i_point], :].T,
                    n_starts_max,
                    rng,
                )
            else:
                (
                    f_current,
                    x_current,
                ) = self.single_evaluate_with_simulation(
                    u[[i_point], :].T,
                    n_starts_max,
                    rng,
                )

            f[i_point] = f_current
            x[[i_point], :] = x_current.T

        return f, x

    def evaluate_with_fixed_x(
        self,
        u: np.ndarray,
        x_fixed: np.ndarray,
        indices_x_fixed: list[int],
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
        original: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

        if u.shape[1] != self.n_u:
            msg = "u.shape[1] != self.n_u"
            raise Exception(msg)

        if x_fixed.shape[1] != len(indices_x_fixed):
            msg = "x_fixed.shape[1] != len(indices_x_fixed)"
            raise Exception(msg)

        if x_fixed.shape[1] != self.n_h_unknown:
            msg = "x_fixed.shape[1] != self.n_h_unknown"
            raise Exception(msg)

        if u.shape[0] != x_fixed.shape[0]:
            msg = "u.shape[0] != x_fixed.shape[0]"
            raise Exception(msg)

        n_points = u.shape[0]

        f = np.empty((n_points, 1))
        x = np.empty((n_points, self.n_x))

        for i_point in range(n_points):
            if original:
                (
                    f_current,
                    x_current,
                ) = self.single_evaluate_with_fixed_x_original(
                    u[[i_point], :].T,
                    x_fixed[[i_point], :].T,
                    indices_x_fixed,
                    n_starts_max,
                    rng,
                )
            else:
                (
                    f_current,
                    x_current,
                ) = self.single_evaluate_with_fixed_x(
                    u[[i_point], :].T,
                    x_fixed[[i_point], :].T,
                    indices_x_fixed,
                    n_starts_max,
                    rng,
                )

            f[i_point] = f_current
            x[[i_point], :] = x_current.T

        return f, x

    def evaluate_with_noisy_simulation(
        self,
        u: np.ndarray,
        noise_std: float,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
        original: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates each row of u in turn; see single_evaluate_with_noisy_simulation_original."""

        if u.shape[1] != self.n_u:
            msg = "u.shape[1] != self.n_u"
            raise Exception(msg)

        if rng is None:
            rng = np.random.default_rng()

        n_points = u.shape[0]

        f = np.empty((n_points, 1))
        x = np.empty((n_points, self.n_x))
        f_no_noise = np.empty((n_points, 1))
        x_no_noise = np.empty((n_points, self.n_x))

        for i_point in range(n_points):
            if original:
                (f_current, x_current, f_no_noise_current, x_no_noise_current) = (
                    self.single_evaluate_with_noisy_simulation_original(
                        u[[i_point], :].T,
                        noise_std,
                        indices_x_measured,
                        n_starts_max,
                        rng,
                    )
                )
            else:
                (f_current, x_current, f_no_noise_current, x_no_noise_current) = (
                    self.single_evaluate_with_noisy_simulation(
                        u[[i_point], :].T,
                        noise_std,
                        indices_x_measured,
                        n_starts_max,
                        rng,
                    )
                )

            f[i_point] = f_current
            x[[i_point], :] = x_current.T
            f_no_noise[i_point] = f_no_noise_current
            x_no_noise[[i_point], :] = x_no_noise_current.T

        return f, x, f_no_noise, x_no_noise

    def single_evaluate_with_simulation_original(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        """Returns f and x from u using h_known and h_unknown. No measurement noise is considered."""

        if u.shape != (self.n_u, 1):
            msg = "u.shape != (self.n_u, 1)"
            raise Exception(msg)

        # %% Setup variables to be solved for

        x = SymbolicType.sym("x", self.n_x, 1)

        # %% Get initial guesses for x

        if rng is None:
            rng = np.random.default_rng()

        X_0 = lhs(
            self.x_lower_bounds_starting_points.ravel(),
            self.x_upper_bounds_starting_points.ravel(),
            n_starts_max,
            rng,
        )

        # %% Setup equations to be solved

        h = cas.Function(
            "h",
            [x],
            [cas.vertcat(self.h_known(u, x), self.h_unknown(u, x))],
        )

        h_jacobian = cas.Function(
            "h_jacobian",
            [x],
            [cas.jacobian(h(x), x)],
        )

        def h_np(x: np.ndarray) -> np.ndarray:
            return h(x[:, np.newaxis]).full().flatten()

        def h_np_jacobian(x: np.ndarray) -> np.ndarray:
            return h_jacobian(x[:, np.newaxis]).full()

        # %% Solve for x and get corresponding values of f

        i_start = 0
        ready_to_return = False
        while i_start < n_starts_max:
            roots = scipy.optimize.fsolve(
                h_np,
                X_0[i_start, :],
                fprime=h_np_jacobian,
            )[:, np.newaxis]

            if (
                (np.linalg.norm(h_np(roots.ravel())) < self.threshold_equality)
                and ((roots >= self.x_lower_bounds_acquisition).all())
                and ((roots <= self.x_upper_bounds_acquisition).all())
            ):
                ready_to_return = True

            if ready_to_return:
                f_at_roots = self.f(u, roots).full().item()
                return f_at_roots, roots

            i_start = i_start + 1

        msg = "Could not get a solution!"
        raise Exception(msg)

    def single_evaluate_with_simulation(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        return self.single_evaluate_with_simulation_original(
            u,
            n_starts_max,
            rng,
        )

    def single_evaluate_with_fixed_x_original(
        self,
        u: np.ndarray,
        x_fixed: np.ndarray,
        indices_x_fixed: list[int],
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        """Returns f and x from u and fixed parts of x using h_known. The fixed part of x is either measured or an output from a GP."""

        if u.shape != (self.n_u, 1):
            msg = "u.shape != (self.n_u, 1)"
            raise Exception(msg)

        if x_fixed.shape != (len(indices_x_fixed), 1):
            msg = "x_fixed.shape != (len(indices_x_fixed), 1)"
            raise Exception(msg)

        if x_fixed.shape != (self.n_h_unknown, 1):
            msg = "x_fixed.shape != (self.n_h_unknown, 1)"
            raise Exception(msg)

        # %% Get the indices of the elements of x that are not measured

        mask_x_free = np.ones((self.n_x,), dtype=bool)
        mask_x_free[indices_x_fixed] = False
        indices_x_free = np.nonzero(mask_x_free)[0].tolist()

        # %% Setup variables to be solved for

        x_free = SymbolicType.sym("x_free", len(indices_x_free), 1)

        # %% Setup helper variables

        x = SymbolicType(self.n_x, 1)
        x[indices_x_fixed] = x_fixed
        x[indices_x_free] = x_free

        # %% Get starting points for x_free

        if rng is None:
            rng = np.random.default_rng()

        sampler = scipy.stats.qmc.LatinHypercube(
            x_fixed.size,
            rng=rng,
        )

        samples = sampler.random(n_starts_max)
        X_free_0 = (
            self.x_lower_bounds_starting_points[indices_x_free].T
            + (
                self.x_upper_bounds_starting_points[indices_x_free]
                - self.x_lower_bounds_starting_points[indices_x_free]
            ).T
            * samples
        )

        # %% Setup equations to be solved

        h = cas.Function("h", [x_free], [self.h_known(u, x)])
        h_jacobian = cas.Function(
            "h_jacobian",
            [x_free],
            [cas.jacobian(h(x_free), x_free)],
        )

        def h_np(x: np.ndarray) -> np.ndarray:
            return h(x[:, np.newaxis]).full().flatten()

        def h_np_jacobian(x: np.ndarray) -> np.ndarray:
            return h_jacobian(x[:, np.newaxis]).full()

        # %% Solve for x_free and get corresponding values of f

        i_start = 0
        ready_to_return = False
        while i_start < n_starts_max:
            roots = scipy.optimize.fsolve(
                h_np,
                X_free_0[i_start, :],
                fprime=h_np_jacobian,
            )[:, np.newaxis]

            if (
                (np.linalg.norm(h_np(roots.ravel())) < self.threshold_equality)
                and ((roots >= self.x_lower_bounds_acquisition[indices_x_free]).all())
                and ((roots <= self.x_upper_bounds_acquisition[indices_x_free]).all())
            ):
                ready_to_return = True

            if ready_to_return:
                x_solution = np.empty((self.n_x, 1))
                x_solution[indices_x_fixed] = x_fixed
                x_solution[indices_x_free] = roots
                f_at_solution = self.f(u, x_solution).full().item()

                return f_at_solution, x_solution

            i_start = i_start + 1

        msg = "Could not get a solution!"
        raise Exception(msg)

    def single_evaluate_with_fixed_x(
        self,
        u: np.ndarray,
        x_fixed: np.ndarray,
        indices_x_fixed: list[int],
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        return self.single_evaluate_with_fixed_x_original(
            u,
            x_fixed,
            indices_x_fixed,
            n_starts_max,
            rng,
        )

    def single_evaluate_with_noisy_simulation_original(
        self,
        u: np.ndarray,
        noise_std: float,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Returns f and x (measured, then oracle values) from u.

        Draws from rng in this order (DESIGN.md, "Order of random draws"):
        1. the oracle simulation solves h_known and h_unknown (fsolve guesses);
        2. noise on the measured states, even when noise_std is 0;
        3. the root-finding problem solves h_known for the other states given the
           measured ones (fsolve guesses). It always runs, also without noise.
        """

        if u.shape != (self.n_u, 1):
            msg = "u.shape != (self.n_u, 1)"
            raise Exception(msg)

        if rng is None:
            rng = np.random.default_rng()

        f_no_noise, x_no_noise = self.single_evaluate_with_simulation(
            u,
            n_starts_max,
            rng,
        )

        noise = rng.normal(0.0, noise_std, size=(len(indices_x_measured), 1))
        x_measured = x_no_noise[indices_x_measured] + noise

        f, x = self.single_evaluate_with_fixed_x(
            u,
            x_measured,
            indices_x_measured,
            n_starts_max,
            rng,
        )

        return f, x, f_no_noise, x_no_noise

    def single_evaluate_with_noisy_simulation(
        self,
        u: np.ndarray,
        noise_std: float,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        return self.single_evaluate_with_noisy_simulation_original(
            u,
            noise_std,
            indices_x_measured,
            n_starts_max,
            rng,
        )
