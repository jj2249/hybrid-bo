"""The hybrid model as an optimization problem: the known equations with the surrogate
model embedded, over scenarios of the GPs' prediction of the unknown states.

`gp_deterministic_equivalent` builds the problem for a given objective, and hybrid BO's
acquisition problem (`hybrid_acquisition`) is its one use. `_unpack_scenarios` reads a
solution.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.dataset import Dataset
from hybrid_bo.gp import GP, FittedGP, fit_gp, predict
from hybrid_bo.problem import Problem, Symbolic, System, Var, _h_inputs
from hybrid_bo.sampling import _start_bounds
from hybrid_bo.solvers import NLP, Ipopt

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenarios:
    """The scenarios of a deterministic equivalent: the scenario draws z (S, P), one row
    per scenario, and the scenario weights w (S,), which sum to 1.

    Raises `ValueError` if the shapes don't match, or a weight is negative, or the
    weights don't sum to 1.
    """

    z: np.ndarray
    w: np.ndarray

    def __post_init__(self) -> None:
        z, w = np.asarray(self.z, dtype=float), np.asarray(self.w, dtype=float)
        if z.ndim != 2 or w.shape != (len(z),):
            msg = (
                f"Scenarios need z of shape (S, P) and w of shape (S,), got {z.shape} "
                f"and {w.shape}"
            )
            raise ValueError(msg)
        if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            msg = f"Scenario weights must be >= 0 and sum to 1, got {w}"
            raise ValueError(msg)

        # Frozen dataclass: set fields through object.__setattr__
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "w", w)


@dataclass(frozen=True)
class DeterministicEquivalent:
    """A deterministic equivalent: the NLP over the decision vector w, the start bounds
    `(lower, upper)` of w for its multistart, and the order of w's variables.

    w holds the N decision variables u, then the S scenario copies of the M mechanistic
    states x (S, M), then those of the unknown states y (S, P) if they are variables.
    `_pack` and `_unpack` are the only code that knows this order. The defaults are
    those of a problem over u alone, such as standard BO's acquisition problem.

    `scenario_values` maps w to f under each scenario (S, 1) and the scenarios' unknown
    states y (S, P), read from w or predicted by the GPs.
    `log_ei` says the objective is -log EI rather than -EI.
    """

    nlp: NLP
    start_bounds: tuple[np.ndarray, np.ndarray]
    n_u: int
    n_x: int = 0
    n_scenarios: int = 0
    unknown_states_as_variables: bool = False
    scenario_values: cas.Function | None = None
    log_ei: bool = False


def _pack(u: Any, x: Any = None, y: Any = None) -> Any:
    """The decision vector w of u (N,), the scenario states x (S, M) and, if they are
    variables, y (S, P): numpy arrays give a numpy array, CasADi matrices a column.

    The states are stacked column by column, so each state has its S scenario copies
    in a row.
    """
    states = [block for block in (x, y) if block is not None]
    if isinstance(u, np.ndarray):
        return np.concatenate([u, *(block.ravel(order="F") for block in states)])
    return cas.vertcat(u, *(cas.vec(block) for block in states))


def _unpack(
    problem: DeterministicEquivalent, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The u (N,), x (S, M) and y (S, P) (None unless they are variables) held in a
    numeric decision vector w of the problem; undoes `_pack`.

    Raises `ValueError` if w has the wrong length for the problem.
    """
    # One column per state of each scenario; a problem over u alone has no scenarios
    w = np.asarray(w, dtype=float).ravel()
    n_u, n_x, n_scenarios = problem.n_u, problem.n_x, problem.n_scenarios
    n_states = (len(w) - n_u) // n_scenarios if n_scenarios else 0
    n_y = n_states - n_x
    fits = len(w) == n_u + n_scenarios * n_states and n_y >= 0
    if not fits or (n_y > 0) != problem.unknown_states_as_variables:
        msg = f"A decision vector of length {len(w)} doesn't fit this problem"
        raise ValueError(msg)
    states = w[n_u:].reshape((n_scenarios, n_states), order="F")
    y = states[:, n_x:] if problem.unknown_states_as_variables else None
    return w[:n_u], states[:, :n_x], y


def _unpack_scenarios(
    problem: DeterministicEquivalent, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The u (N,), x (S, M) and y (S, P) of a numeric decision vector w of a problem
    over scenarios: `_unpack`, with y from `scenario_values`, so the GPs' prediction when
    it was substituted for the unknown states."""
    u, x, _ = _unpack(problem, w)
    assert problem.scenario_values is not None  # a problem over scenarios has it
    return u, x, problem.scenario_values(w)[1].full()


def _scenario_unknown_states(
    system: System,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    u: Any,
    x: Any,
) -> Any:
    """The scenarios' unknown states (S, P), symbolic: unknown state j of scenario s
    is GP j's prediction at the scenario's arguments of h, mean_j + std_j * z_sj, for
    u (N, 1) and x (S, M). `fitted` holds one fitted GP per unknown state, in the
    order of `system.y`.
    """
    # One column of mean and std per unknown state, matching the columns of z
    n_scenarios = len(scenarios.w)
    inputs = cas.horzcat(*_h_inputs(system, cas.repmat(u.T, n_scenarios, 1), x))
    mean, std = zip(*(predict(gp, inputs) for gp in fitted), strict=True)
    return cas.horzcat(*mean) + cas.horzcat(*std) * Symbolic(scenarios.z)


def gp_deterministic_equivalent(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    objective: Callable[[Any], Any],
    unknown_states_as_variables: bool,
) -> DeterministicEquivalent:
    """The hybrid model over the scenarios of the surrogate model, minimizing `objective`
    of f under each scenario (S, 1).

    `fitted` is the surrogate model: one fitted GP per unknown state, in the order of
    `problem.system.y`. Each scenario s has its own copies x_s and y_s of the states, and
    the known equations g(x_s, y_s, u) = 0 are constraints. The unknown states y_s are
    the GPs' prediction at the scenario's arguments of h, mean + std * z_s
    (`_scenario_unknown_states`). With `unknown_states_as_variables` the y_s are variables
    too, tied by y_s - prediction = 0 (paper, Problem 8), which helps IPOPT. Otherwise the
    prediction is substituted for them, so they add no variables and have no bounds.

    The variables are u, the x_s and, if variables, the y_s, in `_pack`'s order, each
    boxed by its bounds and started within its start bounds.

    Raises `ValueError` for scenarios or fitted GPs without one column or one GP per
    unknown state.
    """
    # The checks
    system = problem.system
    if scenarios.z.shape[1] != system.n_y:
        msg = (
            f"The scenarios need one column of z per unknown state ({system.n_y}), "
            f"got {scenarios.z.shape[1]}"
        )
        raise ValueError(msg)
    if len(fitted) != system.n_y:
        msg = (
            "The surrogate model needs one fitted GP per unknown state "
            f"({system.n_y}), got {len(fitted)}"
        )
        raise ValueError(msg)

    # The symbols for u and each scenario's mechanistic states (row s: scenario s), and
    # the unknown states: variables tied to the GP, or the GP itself
    n_scenarios = len(scenarios.w)
    u = Symbolic.sym("u", system.n_u)
    x = Symbolic.sym("x", n_scenarios, system.n_x)
    prediction = _scenario_unknown_states(system, fitted, scenarios, u, x)
    y = prediction
    if unknown_states_as_variables:
        y = Symbolic.sym("y", n_scenarios, system.n_y)

    # Each scenario: its known equations, its tie to the GP, and its f
    constraints, f_parts = [], []
    for s in range(n_scenarios):
        x_s, y_s = x[s, :].T, y[s, :].T
        constraints.append(system.g(x_s, y_s, u))
        if unknown_states_as_variables:
            constraints.append((y - prediction)[s, :].T)
        f_parts.append(problem.f(x_s, y_s, u))

    # The Var of each entry of w, each state once per scenario, for the bounds
    def per_scenario(variables: Sequence[Var]) -> np.ndarray:
        """The variables repeated in a row per scenario, (S, len(variables))."""
        return np.tile(np.array(variables, dtype=object), (n_scenarios, 1))

    unknown = per_scenario(system.y) if unknown_states_as_variables else None
    variables = _pack(np.array(system.u, dtype=object), per_scenario(system.x), unknown)

    f = cas.vertcat(*f_parts)
    w = _pack(u, x, y if unknown_states_as_variables else None)
    nlp = NLP(
        cas.Function("acquisition", [w], [objective(f)]),
        np.array([var.lower for var in variables]),
        np.array([var.upper for var in variables]),
        cas.Function("g", [w], [cas.vertcat(*constraints)]),
    )
    return DeterministicEquivalent(
        nlp,
        _start_bounds(variables),
        system.n_u,
        system.n_x,
        n_scenarios,
        unknown_states_as_variables,
        cas.Function("scenario_values", [w], [f, y]),
    )


def fit_surrogate(
    system: System,
    dataset: Dataset,
    *,
    gp: GP,
    rng: np.random.Generator,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> tuple[FittedGP, ...]:
    """The surrogate model of the system: the prior GP `gp` fitted to each unknown state
    of `dataset` over the arguments of h, one fitted GP per unknown state, in the order
    of `system.y`.

    Each GP is fitted by `fit_gp` from `n_starts` starts.
    """
    X = np.column_stack(_h_inputs(system, dataset.u, dataset.x))
    fitted = []
    for var, y in zip(system.y, dataset.y.T, strict=True):
        _logger.info("Fitting the GP of %s", var.name)
        fitted.append(fit_gp(gp, X, y, rng, n_starts, solver))
    return tuple(fitted)
