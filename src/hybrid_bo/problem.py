from dataclasses import dataclass

import casadi as cas
import numpy as np
import scipy
import scipy.optimize

from hybrid_bo.bounds import Bounds, lhs
from hybrid_bo.type_aliases import SymbolicType


@dataclass(frozen=True)
class Problem:
    """A partially known system.

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

    Bounds:
    - u_bounds, x_bounds: box constraints of the acquisition problem. x_bounds also
      decide whether a root found when evaluating the system is accepted; they may be
      infinite (`Bounds.unbounded(n_x)`).
    - u_start_bounds, x_start_bounds: finite ranges for starting points and initial
      guesses.
    """

    f: cas.Function
    h_known: cas.Function
    h_unknown: cas.Function
    u_bounds: Bounds
    x_bounds: Bounds
    u_start_bounds: Bounds
    x_start_bounds: Bounds
    # Residual norm below which an equation system counts as solved
    threshold_equality: float = 1.0e-10

    @property
    def n_u(self) -> int:
        return self.f.numel_in(0)

    @property
    def n_x(self) -> int:
        return self.f.numel_in(1)

    @property
    def n_h_known(self) -> int:
        return self.h_known.numel_out(0)

    @property
    def n_h_unknown(self) -> int:
        return self.h_unknown.numel_out(0)

    def __post_init__(self) -> None:
        n_u, n_x = self.n_u, self.n_x

        for name, function, n_out in (
            ("f", self.f, 1),
            ("h_known", self.h_known, self.n_h_known),
            ("h_unknown", self.h_unknown, self.n_h_unknown),
        ):
            if function.n_in() != 2 or function.n_out() != 1:
                msg = f"{name} must take (u, x) and have one output"
                raise ValueError(msg)
            if function.size_in(0) != (n_u, 1) or function.size_in(1) != (n_x, 1):
                msg = (
                    f"{name} inputs must be columns ({n_u}, 1) and ({n_x}, 1), "
                    f"got {function.size_in(0)} and {function.size_in(1)}"
                )
                raise ValueError(msg)
            if function.size_out(0) != (n_out, 1):
                msg = f"{name} output must be ({n_out}, 1), got {function.size_out(0)}"
                raise ValueError(msg)

        for name, bounds, n in (
            ("u_bounds", self.u_bounds, n_u),
            ("x_bounds", self.x_bounds, n_x),
            ("u_start_bounds", self.u_start_bounds, n_u),
            ("x_start_bounds", self.x_start_bounds, n_x),
        ):
            if bounds.lower.shape != (n,):
                msg = f"{name} must have size {n}, got {bounds.lower.size}"
                raise ValueError(msg)

        for name, bounds in (
            ("u_start_bounds", self.u_start_bounds),
            ("x_start_bounds", self.x_start_bounds),
        ):
            if not (
                np.isfinite(bounds.lower).all() and np.isfinite(bounds.upper).all()
            ):
                msg = f"{name} must be finite"
                raise ValueError(msg)

    def evaluate_with_simulation(
        self,
        u: np.ndarray,
        n_starts_max: int = 100,
        use_jacobian: bool = True,
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
                    use_jacobian,
                    rng,
                )
            else:
                (
                    f_current,
                    x_current,
                ) = self.single_evaluate_with_simulation(
                    u[[i_point], :].T,
                    n_starts_max,
                    use_jacobian,
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
        use_jacobian: bool = True,
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
                    use_jacobian,
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
                    use_jacobian,
                    rng,
                )

            f[i_point] = f_current
            x[[i_point], :] = x_current.T

        return f, x

    def evaluate_with_noisy_simulation(
        self,
        u: np.ndarray,
        measurement_noise: np.ndarray,
        indices_x_measured: list[int],
        n_starts_max: int = 100,
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
        original: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if u.shape[1] != self.n_u:
            msg = "u.shape[1] != self.n_u"
            raise Exception(msg)

        if measurement_noise.shape[1] != len(indices_x_measured):
            msg = "measurement_noise.shape[1] != len(indices_x_measured)"
            raise Exception(msg)

        if u.shape[0] != measurement_noise.shape[0]:
            msg = "u.shape[0] != measurement_noise.shape[0]"
            raise Exception(msg)

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
                        measurement_noise[[i_point], :].T,
                        indices_x_measured,
                        n_starts_max,
                        use_jacobian,
                        rng,
                    )
                )
            else:
                (f_current, x_current, f_no_noise_current, x_no_noise_current) = (
                    self.single_evaluate_with_noisy_simulation(
                        u[[i_point], :].T,
                        measurement_noise[[i_point], :].T,
                        indices_x_measured,
                        n_starts_max,
                        use_jacobian,
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
        use_jacobian: bool = True,
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

        X_0 = lhs(self.x_start_bounds, n_starts_max, rng)

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
            if use_jacobian:
                roots = scipy.optimize.fsolve(
                    h_np,
                    X_0[i_start, :],
                    fprime=h_np_jacobian,
                )[
                    :,
                    np.newaxis,
                ]
            else:
                roots = scipy.optimize.fsolve(h_np, X_0[i_start, :])[
                    :,
                    np.newaxis,
                ]

            if (
                (np.linalg.norm(h_np(roots.ravel())) < self.threshold_equality)
                and ((roots.ravel() >= self.x_bounds.lower).all())
                and ((roots.ravel() <= self.x_bounds.upper).all())
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
        use_jacobian: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, np.ndarray]:
        return self.single_evaluate_with_simulation_original(
            u,
            n_starts_max,
            use_jacobian,
            rng,
        )

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

        # Known issue: the sampler uses x_fixed.size dimensions, not len(indices_x_free)
        samples = sampler.random(n_starts_max)
        lower = self.x_start_bounds.lower[indices_x_free]
        upper = self.x_start_bounds.upper[indices_x_free]
        X_free_0 = lower + (upper - lower) * samples

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
            if use_jacobian:
                roots = scipy.optimize.fsolve(
                    h_np,
                    X_free_0[i_start, :],
                    fprime=h_np_jacobian,
                )[:, np.newaxis]
            else:
                roots = scipy.optimize.fsolve(h_np, X_free_0[i_start, :])[
                    :,
                    np.newaxis,
                ]

            if (
                (np.linalg.norm(h_np(roots.ravel())) < self.threshold_equality)
                and ((roots.ravel() >= self.x_bounds.lower[indices_x_free]).all())
                and ((roots.ravel() <= self.x_bounds.upper[indices_x_free]).all())
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

        if u.shape != (self.n_u, 1):
            msg = "u.shape != (self.n_u, 1)"
            raise Exception(msg)

        if measurement_noise.shape != (len(indices_x_measured), 1):
            msg = "measurement_noise.shape != (len(indices_x_measured), 1)"
            raise Exception(msg)

        f_no_noise, x_no_noise = self.single_evaluate_with_simulation(
            u,
            n_starts_max,
            use_jacobian,
            rng,
        )

        if (measurement_noise != 0.0).any():
            x_measured = x_no_noise[indices_x_measured] + measurement_noise

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
