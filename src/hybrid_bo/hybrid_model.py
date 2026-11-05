"""The hybrid model as an optimization problem: the known equations with the surrogate
model embedded.

`build_deterministic_equivalent` builds the problem for given unknown equations and a
given objective; the caller solves it by `solve_with_multistart`, and
`_unpack_scenarios` reads the solution. The unknown equations are the seam, with two
adapters: the GPs' scenarios (`gp_deterministic_equivalent`) and the oracle
(`oracle_optimum`).

Everything that solves the hybrid model is one use of the GPs' adapter.
`optimize_hybrid_model` is the simplest: the nominal scenario, minimizing the objective
f itself, with no Bayesian optimization. Hybrid BO's acquisition problem
(`hybrid_acquisition`) supplies an acquisition function as the objective, and
`hybrid_bo` is the loop around it: fit, solve, evaluate.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as np

from hybrid_bo.gp import GP, FittedGP, fit_gp, predict
from hybrid_bo.history import History
from hybrid_bo.problem import Problem, Symbolic, System, Var, _h_inputs
from hybrid_bo.sampling import _start_bounds, lhs
from hybrid_bo.solvers import NLP, Ipopt, solve_with_multistart

# What a deterministic equivalent minimizes, from CasADi expressions of f under each
# scenario (S, 1), the scenario states x (S, M) and y (S, P), and u (N, 1): the
# objective (1, 1), and the std of f (1, 1) that `scenario_values` reports
ScenarioObjective = Callable[[Any, Any, Any, Any], tuple[Any, Any]]

# What ties the unknown states of a deterministic equivalent, from CasADi expressions
# of u (N, 1) and the scenario states x (S, M) and y (S, P): the residuals of the
# unknown equations (S, P), zero where they hold
UnknownEquations = Callable[[Any, Any, Any], Any]


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


def nominal_scenario(n_y: int) -> Scenarios:
    """The nominal scenario, the only scenario of linearization and of the nominal
    optimum: zero scenario draws z (1, n_y) and weight 1."""
    return Scenarios(np.zeros((1, n_y)), np.ones(1))


@dataclass(frozen=True)
class DeterministicEquivalent:
    """A deterministic equivalent: the NLP over the decision vector w, the start bounds
    `(lower, upper)` of w for its multistart, and the order of w's variables.

    w holds the N decision variables u, then the S scenario copies of the M mechanistic
    states x (S, M), then those of the unknown states y (S, P) if they are variables.
    `_pack` and `_unpack` are the only code that knows this order. The defaults are
    those of a problem over u alone, such as standard BO's acquisition problem.

    `scenario_values` maps w to f under each scenario (S, 1), the scenarios' unknown
    states y (S, P), read from w or predicted by the GPs, and the std of f (1, 1), as
    the objective states it.
    """

    nlp: NLP
    start_bounds: tuple[np.ndarray, np.ndarray]
    n_u: int
    n_x: int = 0
    n_scenarios: int = 0
    unknown_states_as_variables: bool = False
    scenario_values: cas.Function | None = None


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
    over scenarios: `_unpack`, with y from `scenario_values`, so the GPs' prediction
    when it was substituted for the unknown states."""
    u, x, _ = _unpack(problem, w)
    assert problem.scenario_values is not None  # a problem over scenarios has them
    _, y, _ = problem.scenario_values(w)
    return u, x, y.full()


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


def _minimize_f(f: Any, _x: Any, _y: Any, _u: Any) -> tuple[Any, Any]:
    """The objective f itself, of one scenario, so f has no spread."""
    return f, Symbolic(0)


def _assemble(
    problem: Problem,
    u: Any,
    x: Any,
    y: Any,
    residuals: Any | None,
    objective: ScenarioObjective,
    objective_name: str,
) -> DeterministicEquivalent:
    """The deterministic equivalent over the symbols u (N, 1) and x (S, M), and the
    unknown states y (S, P).

    `residuals` holds the unknown equations (S, P), and then y are symbols, variables of
    the problem. With None, y is an expression of u and x, so no variables and no
    equations. The variables are in `_pack`'s order; `build_deterministic_equivalent`
    says the rest.
    """
    # Each scenario: its known equations, its unknown equations, and its f
    system = problem.system
    n_scenarios = x.size1()
    unknown_states_as_variables = residuals is not None
    constraints, f_parts = [], []
    for s in range(n_scenarios):
        x_s, y_s = x[s, :].T, y[s, :].T
        constraints.append(system.g(x_s, y_s, u))
        if unknown_states_as_variables:
            constraints.append(residuals[s, :].T)
        f_parts.append(problem.f(x_s, y_s, u))
    f = cas.vertcat(*f_parts)
    minimized, f_std = objective(f, x, y, u)

    # The Var of each entry of w, each state once per scenario, for the bounds
    def per_scenario(variables: Sequence[Var]) -> np.ndarray:
        """The variables repeated in a row per scenario, (S, len(variables))."""
        return np.tile(np.array(variables, dtype=object), (n_scenarios, 1))

    unknown = per_scenario(system.y) if unknown_states_as_variables else None
    variables = _pack(np.array(system.u, dtype=object), per_scenario(system.x), unknown)

    w = _pack(u, x, y if unknown_states_as_variables else None)
    nlp = NLP(
        cas.Function(objective_name, [w], [minimized]),
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
        cas.Function("scenario_values", [w], [f, y, f_std]),
    )


def build_deterministic_equivalent(
    problem: Problem,
    n_scenarios: int,
    unknown_equations: UnknownEquations,
    objective: ScenarioObjective,
    objective_name: str,
) -> DeterministicEquivalent:
    """The hybrid model over `n_scenarios` scenarios, minimizing `objective`.

    Each scenario s has its own copies x_s and y_s of the states, and the known
    equations g(x_s, y_s, u) = 0 are constraints, followed by its row of
    `unknown_equations`, which ties the unknown states and may read them.

    The variables are u, the x_s and the y_s, in `_pack`'s order. Each variable is boxed
    by its bounds and started within its start bounds. `objective_name` names the
    objective's function, which the solver's log and warnings give.
    """
    system = problem.system
    u = Symbolic.sym("u", system.n_u)
    # Row s: the states of scenario s
    x = Symbolic.sym("x", n_scenarios, system.n_x)
    y = Symbolic.sym("y", n_scenarios, system.n_y)
    residuals = unknown_equations(u, x, y)
    return _assemble(problem, u, x, y, residuals, objective, objective_name)


def gp_deterministic_equivalent(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    scenarios: Scenarios,
    objective: ScenarioObjective,
    unknown_states_as_variables: bool,
    objective_name: str,
) -> DeterministicEquivalent:
    """The hybrid model over the scenarios of the surrogate model, minimizing
    `objective`.

    `fitted` is the surrogate model: one fitted GP per unknown state, in the order of
    `problem.system.y`. The unknown states y_s of scenario s are the GPs' prediction at
    the scenario's arguments of h, mean + std * z_s (`_scenario_unknown_states`). With
    `unknown_states_as_variables` the y_s are variables too, and the unknown equations
    of `build_deterministic_equivalent` are y_s - prediction (paper, Problem 8), which
    helps IPOPT.

    Otherwise the prediction is substituted for them, which only an explicit y allows,
    so this stays here. Substituted unknown states are no variables, so they have no
    bounds.

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
    n_scenarios = len(scenarios.w)

    # The unknown states as variables, tied to the prediction
    if unknown_states_as_variables:

        def unknown_equations(u: Any, x: Any, y: Any) -> Any:
            """The unknown states' distance from the GPs' prediction."""
            return y - _scenario_unknown_states(system, fitted, scenarios, u, x)

        return build_deterministic_equivalent(
            problem, n_scenarios, unknown_equations, objective, objective_name
        )

    # The prediction in their place
    u = Symbolic.sym("u", system.n_u)
    x = Symbolic.sym("x", n_scenarios, system.n_x)
    y = _scenario_unknown_states(system, fitted, scenarios, u, x)
    return _assemble(problem, u, x, y, None, objective, objective_name)


def fit_surrogate(
    system: System,
    history: History,
    *,
    gp: GP,
    rng: np.random.Generator,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> tuple[FittedGP, ...]:
    """The surrogate model of the system: the prior GP `gp` fitted to each unknown state
    of `history` over the arguments of h, one fitted GP per unknown state, in the order
    of `system.y`.

    The history is a run's, or `reconstruct`'s for measurements from elsewhere. Each GP
    is fitted by `fit_gp` from `n_starts` starts.
    """
    X = np.column_stack(_h_inputs(system, history.u, history.x))
    return tuple(fit_gp(gp, X, y, rng, n_starts, solver) for y in history.y.T)


@dataclass(frozen=True)
class NominalOptimum:
    """The nominal optimum: the decision variables u (N,), the nominal states x (M,) and
    y (P,) there, and the objective f."""

    u: np.ndarray
    x: np.ndarray
    y: np.ndarray
    f: float


def optimize_hybrid_model(
    problem: Problem,
    fitted: tuple[FittedGP, ...],
    rng: np.random.Generator,
    *,
    n_starts: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> NominalOptimum:
    """The nominal optimum: the lowest objective the hybrid model allows with the GPs
    replaced by their mean, min f(x, y, u) subject to g(x, y, u) = 0 and y = the GPs'
    mean at (x, u), within the bounds.

    `fitted` is the surrogate model (`fit_surrogate`). This is the deterministic
    equivalent of the nominal scenario, minimizing f, solved once: no acquisition
    function, no measurements. It has no uncertainty in it, so it trusts the GPs' mean
    even far from their training data, and the optimum can sit where the model only
    extrapolates.

    Solved by `solve_with_multistart` from `n_starts` Latin hypercube starts over the
    start bounds of the variables, so it is the best local optimum found: raise
    `n_starts` if the objective has several. The unknown states are variables, so their
    bounds hold. Raises `ValueError` unless `fitted` has one GP per unknown state, and
    `NoSolutionError` if every start fails.
    """
    nominal = gp_deterministic_equivalent(
        problem,
        fitted,
        nominal_scenario(problem.system.n_y),
        _minimize_f,
        unknown_states_as_variables=True,
        objective_name="nominal_optimum",
    )
    starts = lhs(*nominal.start_bounds, n_starts, rng)
    solution = solve_with_multistart(nominal.nlp, starts, solver)
    u, x, y = _unpack_scenarios(nominal, solution.w)
    return NominalOptimum(u, x[0], y[0], solution.f)
