"""Bayesian optimization: the optimization loop shared by the BO methods, standard BO
and hybrid BO.

A BO method is a settings dataclass plus a function that runs `optimization_loop` with
its own trial-point function. The loop evaluates the initial points, then in each
iteration asks the trial-point function for the next trial point and evaluates it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import assert_never

import numpy as np

from hybrid_bo.acquisition import (
    EI,
    LCB,
    SAA,
    AcquisitionSolution,
    Linearization,
    _check_acquisition,
    _check_approximation,
    _scenario_states,
    _unpack,
    hybrid_acquisition,
    nominal_scenario,
    saa_scenarios,
    standard_acquisition,
)
from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Measure, evaluate
from hybrid_bo.gp import FittedGP, PriorGP, _gp_record, default_gp, fit_gp
from hybrid_bo.history import History, _concatenate, incumbents
from hybrid_bo.problem import Problem, Var, _h_inputs
from hybrid_bo.run import Run, run_record
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import Ipopt, multistart

# Chooses the next trial point (N,) from the points evaluated so far, drawing from the
# run's generator, and returns it with the fitted surrogate model behind it and the
# solution of the acquisition problem it came from
TrialPoint = Callable[
    [Problem, History, np.random.Generator],
    tuple[np.ndarray, FittedGP, AcquisitionSolution],
]


def optimization_loop(
    problem: Problem,
    measure: Measure,
    initial_points: np.ndarray,
    n_iterations: int,
    trial_point: TrialPoint,
    rng: np.random.Generator,
    record: dict[str, str],
) -> Run:
    """Evaluates the initial points (n, N), then runs `n_iterations` iterations: each
    calls `trial_point` and evaluates the trial point it returns.

    Returns the run with `record`, labeled with the method it names, and per
    iteration the surrogate and the acquisition problem's solution behind that
    iteration's trial point. For writing new methods: `record` comes from
    `run_record`, built before the method draws anything.

    Raises what `evaluate` and `trial_point` raise; a `HybridBOError` carries the
    partial run as `error.run`, and its history (every point evaluated before the
    error) as `error.history`. If evaluating a trial point fails, its surrogate and
    acquisition solution are dropped with it, so the run keeps one of each per
    evaluated trial point.
    """
    history: History | None = None
    surrogates: list[FittedGP] = []
    acquisitions: list[AcquisitionSolution] = []
    try:
        history = evaluate(problem, measure, initial_points, rng)
        for _ in range(n_iterations):
            u, surrogate, acquisition = trial_point(problem, history, rng)
            later = evaluate(problem, measure, u[np.newaxis], rng)
            history = _concatenate(history, later)
            surrogates.append(surrogate)
            acquisitions.append(acquisition)
    except HybridBOError as error:
        # Before the loop, the initial points evaluated so far are the partial run
        partial = error.history if history is None else history
        if partial is not None:
            error.history = partial
            error.run = Run(
                partial,
                record,
                record["method"],
                surrogates=surrogates,
                acquisitions=acquisitions,
            )
        raise
    return Run(
        history,
        record,
        record["method"],
        surrogates=surrogates,
        acquisitions=acquisitions,
    )


@dataclass(frozen=True)
class StandardBOSettings:
    """Settings of standard BO: the GP of u -> f (None: `default_gp` over the
    decision variables), the acquisition function, the number of multistart starts
    for the acquisition problem and for fitting the GP, and the solver for both.

    Raises `TypeError` unless `acquisition` is an `EI` or an `LCB`.
    """

    gp: PriorGP | None = None
    acquisition: EI | LCB = field(default_factory=EI)
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = field(default_factory=Ipopt)

    def __post_init__(self) -> None:
        _check_acquisition(self.acquisition)


@dataclass(frozen=True)
class HybridBOSettings:
    """Settings of hybrid BO: the GP of the unknown states over the arguments of h
    (None: `default_gp` over `h_inputs`), the acquisition function (what is
    computed), its approximation (how it is computed, with its own settings), the
    number of multistart starts for the acquisition problem and for fitting the GP,
    and the solver for both.

    Raises `TypeError` unless `acquisition` is an `EI` or an `LCB` and
    `approximation` an `SAA` or a `Linearization`, and `NotImplementedError` for
    `LCB` with `Linearization`.
    """

    gp: PriorGP | None = None
    acquisition: EI | LCB = field(default_factory=EI)
    approximation: SAA | Linearization = field(default_factory=SAA)
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = field(default_factory=Ipopt)

    def __post_init__(self) -> None:
        _check_acquisition(self.acquisition)
        _check_approximation(self.acquisition, self.approximation)


def _method_record(
    problem: Problem,
    measure: Measure,
    rng: np.random.Generator,
    method: str,
    n_iterations: int,
    settings: StandardBOSettings | HybridBOSettings,
    gp: PriorGP,
    gp_inputs: Sequence[Var],
) -> dict[str, str]:
    """The record of a BO run: `run_record` with the settings except the prior GP,
    which is recorded as text in its own entries (its repr would only name the
    kernel's function), over the names of its inputs."""
    settings_text = ", ".join(
        [f"n_iterations={n_iterations}"]
        + [
            f"{f.name}={getattr(settings, f.name)!r}"
            for f in fields(settings)
            if f.name != "gp"
        ]
    )
    record = run_record(problem, measure, rng, method, settings_text)
    return record | _gp_record(gp, [var.name for var in gp_inputs])


def standard_bo(
    problem: Problem,
    measure: Measure,
    initial_points: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
    *,
    settings: StandardBOSettings = StandardBOSettings(),
) -> Run:
    """Standard BO: a GP of the objective over the decision variables, u -> f, with
    the acquisition function computed analytically.

    Evaluates the initial points (n, N), then in each iteration fits the GP to the
    measured objective so far, solves the acquisition problem by multistart from
    Latin hypercube starts over the start bounds of `u`, and evaluates the solution.
    Each iteration's acquisition solution holds the acquisition value only.
    `settings` (keyword-only) defaults to `StandardBOSettings()`.

    Raises what `evaluate` and `multistart` raise; a `HybridBOError` carries the
    partial run as `error.run`.
    """
    # The GP and the record
    u_vars = problem.system.u
    gp = default_gp(len(u_vars)) if settings.gp is None else settings.gp
    record = _method_record(
        problem, measure, rng, "standard_bo", n_iterations, settings, gp, u_vars
    )

    def trial_point(
        problem: Problem, history: History, rng: np.random.Generator
    ) -> tuple[np.ndarray, FittedGP, AcquisitionSolution]:
        """Fits the GP to the points so far and solves the acquisition problem."""
        fitted = fit_gp(
            gp, history.u, history.f, rng, settings.n_starts_gp, settings.solver
        )
        incumbent = float(incumbents(history)[-1])
        acquisition_problem = standard_acquisition(
            problem, gp, fitted, incumbent, settings.acquisition
        )
        starts = lhs(*acquisition_problem.start_bounds, settings.n_starts, rng)
        solution = multistart(acquisition_problem.nlp, starts, settings.solver)
        u, _, _ = _unpack(acquisition_problem, solution.w)
        return u, fitted, AcquisitionSolution(solution.f)

    return optimization_loop(
        problem, measure, initial_points, n_iterations, trial_point, rng, record
    )


def hybrid_bo(
    problem: Problem,
    measure: Measure,
    initial_points: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
    *,
    settings: HybridBOSettings = HybridBOSettings(),
) -> Run:
    """Hybrid BO: a GP of the unknown states over the arguments of h, combined with
    the known equations in the acquisition problem (paper, Problems 8 and 9).

    Draws the scenarios (SAA: `settings.approximation.n_scenarios` standard normal
    draws, equal weights; linearization draws nothing and has the nominal scenario
    alone), the same in every iteration. Then evaluates the initial
    points (n, N), and in each iteration fits the GP to the measured and reconstructed
    states so far, solves `hybrid_acquisition` by multistart from Latin hypercube
    starts over the start bounds of its variables, and evaluates the decision
    variables of the solution. The run holds the scenarios, and each iteration's
    acquisition solution holds the scenario states at its trial point. `settings`
    (keyword-only) defaults to `HybridBOSettings()`.

    Raises what `evaluate` and `multistart` raise; a `HybridBOError` carries the
    partial run, with the scenarios, as `error.run`.
    """
    # The GP and the record, before the scenarios are drawn
    system = problem.system
    gp = default_gp(len(system.h_inputs)) if settings.gp is None else settings.gp
    record = _method_record(
        problem,
        measure,
        rng,
        "hybrid_bo",
        n_iterations,
        settings,
        gp,
        system.h_inputs,
    )
    # The scenarios: SAA draws them, linearization has the nominal scenario alone
    approximation = settings.approximation
    match approximation:
        case SAA(n_scenarios=n_scenarios):
            scenarios = saa_scenarios(n_scenarios, system.n_y, rng)
        case Linearization():
            scenarios = nominal_scenario(system.n_y)
        case _:
            assert_never(approximation)

    def trial_point(
        problem: Problem, history: History, rng: np.random.Generator
    ) -> tuple[np.ndarray, FittedGP, AcquisitionSolution]:
        """Fits the GP to the points so far and solves the acquisition problem."""
        # One unknown state (P = 1), so the GP has one output
        X = np.column_stack(_h_inputs(system, history.u, history.x))
        fitted = fit_gp(
            gp, X, history.y[:, 0], rng, settings.n_starts_gp, settings.solver
        )
        incumbent = float(incumbents(history)[-1])
        acquisition_problem = hybrid_acquisition(
            problem,
            gp,
            fitted,
            scenarios,
            incumbent,
            settings.acquisition,
            approximation,
        )
        starts = lhs(*acquisition_problem.start_bounds, settings.n_starts, rng)
        solution = multistart(acquisition_problem.nlp, starts, settings.solver)

        # The trial point, and the scenario states at the solution
        u, _, _ = _unpack(acquisition_problem, solution.w)
        x, y = _scenario_states(
            system, gp, fitted, scenarios, acquisition_problem, solution.w
        )
        return u, fitted, AcquisitionSolution(solution.f, x, y)

    try:
        run = optimization_loop(
            problem, measure, initial_points, n_iterations, trial_point, rng, record
        )
    except HybridBOError as error:
        if error.run is not None:
            error.run = replace(error.run, scenarios=scenarios)
        raise
    return replace(run, scenarios=scenarios)
