"""Bayesian optimization: the optimization loop shared by the BO methods, standard BO
and hybrid BO.

A BO method is a function that runs `optimization_loop` with its own trial-point
function. The loop evaluates the initial points, then in each
iteration asks the trial-point function for the next trial point and evaluates it.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import fields
from typing import assert_never

import numpy as np

from hybrid_bo.acquisition import (
    EI,
    LCB,
    SAA,
    AcquisitionProblem,
    AcquisitionSolution,
    Linearization,
    SoftSAA,
    hybrid_acquisition,
    saa_scenarios,
    standard_acquisition,
)
from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Measure, evaluate
from hybrid_bo.gp import GP, FittedGP, fit_gp
from hybrid_bo.history import History, _concatenate, incumbents
from hybrid_bo.hybrid_model import (
    Scenarios,
    _unpack_scenarios,
    fit_surrogate,
    nominal_scenario,
)
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, run_artifact
from hybrid_bo.sampling import _starting_points, lhs
from hybrid_bo.settings import HybridBOSettings, MethodSettings, StandardBOSettings
from hybrid_bo.solvers import Ipopt, multistart

_logger = logging.getLogger(__name__)

# Chooses the next trial point (N,) from the points evaluated so far, drawing from the
# run's generator, and returns it with the fitted surrogate model behind it (one
# fitted GP per unknown state, or the one GP of f) and the solution of the
# acquisition problem it came from
TrialPoint = Callable[
    [Problem, History, np.random.Generator],
    tuple[np.ndarray, tuple[FittedGP, ...], AcquisitionSolution],
]


def optimization_loop(
    problem: Problem,
    measure: Measure,
    initial_points: np.ndarray,
    n_iterations: int,
    trial_point: TrialPoint,
    rng: np.random.Generator,
    artifact: dict[str, str],
    *,
    settings: MethodSettings | None,
    scenarios: Scenarios | None,
) -> Run:
    """Evaluates the initial points (n, N), then runs `n_iterations` iterations: each
    calls `trial_point` and evaluates the trial point it returns.

    Returns the run with `artifact`, `settings` and `scenarios`, labeled with the method
    the artifact names, and per iteration the surrogate and the acquisition problem's
    solution behind that iteration's trial point. For writing new methods: `artifact`
    comes from `run_artifact`, built before the method draws anything, `settings` are
    the method's, which the iteration plots replay (None for a method they can't
    replay), and `scenarios` are the ones its trial points use (None for a method
    without them).

    Raises what `evaluate` and `trial_point` raise; a `HybridBOError` carries the
    partial run as `error.run`, and its history (every point evaluated before the
    error) as `error.history`. If evaluating a trial point fails, its surrogate and
    acquisition solution are dropped with it, so the run keeps one of each per
    evaluated trial point.

    Logs the time taken by the initial points, and each iteration's time and
    incumbent, at info level.
    """
    history: History | None = None
    surrogates: list[tuple[FittedGP, ...]] = []
    acquisitions: list[AcquisitionSolution] = []
    method = artifact["method"]

    def run_of(evaluated: History) -> Run:
        """The run of the points evaluated so far, complete or partial."""
        return Run(
            evaluated,
            artifact,
            method,
            surrogates=surrogates,
            acquisitions=acquisitions,
            scenarios=scenarios,
            settings=settings,
        )

    try:
        start = time.perf_counter()
        history = evaluate(problem, measure, initial_points, rng)
        _logger.info(
            "%s: evaluated %d initial points in %.2fs",
            method,
            len(initial_points),
            time.perf_counter() - start,
        )
        for i in range(n_iterations):
            start = time.perf_counter()
            u, surrogate, acquisition = trial_point(problem, history, rng)
            later = evaluate(problem, measure, u[np.newaxis], rng)
            history = _concatenate(history, later)
            surrogates.append(surrogate)
            acquisitions.append(acquisition)
            _logger.info(
                "%s iteration %d of %d: %.2fs, incumbent %.6g",
                method,
                i + 1,
                n_iterations,
                time.perf_counter() - start,
                incumbents(history)[-1],
            )
    except HybridBOError as error:
        # Before the loop, the initial points evaluated so far are the partial run
        partial = error.history if history is None else history
        if partial is not None:
            error.history = partial
            error.run = run_of(partial)
        raise
    return run_of(history)


def _method_artifact(
    problem: Problem,
    measure: Measure,
    rng: np.random.Generator,
    method: str,
    n_iterations: int,
    settings: MethodSettings,
) -> dict[str, str]:
    """The artifact of a BO run: `run_artifact` with the settings except the prior GP,
    whose repr is its own entry, `gp`."""
    settings_text = ", ".join(
        [f"n_iterations={n_iterations}"]
        + [
            f"{f.name}={getattr(settings, f.name)!r}"
            for f in fields(settings)
            if f.name != "gp"
        ]
    )
    artifact = run_artifact(problem, measure, rng, method, settings_text)
    return artifact | {"gp": repr(settings.gp)}


def _acquisition_problem(
    problem: Problem,
    history: History,
    *,
    n_evaluated: int,
    surrogate: tuple[FittedGP, ...],
    scenarios: Scenarios | None,
    settings: MethodSettings,
) -> AcquisitionProblem:
    """The acquisition problem that follows the first `n_evaluated` points of `history`:
    the method's, as `settings` says, from its surrogate, its scenarios (None for
    standard BO) and the incumbent after those points.

    The loop and `iteration_problem` both build it here, so the iteration figures show
    the function the method minimized (ADR 0005).
    """
    incumbent = float(incumbents(history)[n_evaluated - 1])
    match settings:
        case StandardBOSettings(acquisition=acquisition):
            return standard_acquisition(problem, surrogate[0], incumbent, acquisition)
        case HybridBOSettings(acquisition=acquisition, approximation=approximation):
            assert scenarios is not None  # hybrid BO has its scenarios
            return hybrid_acquisition(
                problem, surrogate, scenarios, incumbent, acquisition, approximation
            )
        case _:
            assert_never(settings)


def standard_bo(
    problem: Problem,
    measure: Measure,
    *,
    n_iterations: int,
    rng: np.random.Generator,
    gp: GP,
    acquisition: EI | LCB,
    n_initial: int | None = None,
    initial_points: np.ndarray | None = None,
    n_starts: int = 10,
    n_starts_gp: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> Run:
    """Standard BO: a GP of the objective over the decision variables, u -> f, with
    the acquisition function computed analytically.

    Evaluates `initial_points` (n, N), or `n_initial` points drawn from `rng`, then in
    each iteration fits the prior GP `gp` to the measured objective so far, solves the
    acquisition problem by multistart from `n_starts` Latin hypercube starts over the
    start bounds of `u`, and evaluates the solution. Each iteration's acquisition
    solution holds the acquisition value only.

    Raises `ValueError` unless exactly one of `n_initial` and `initial_points` is
    given, `TypeError` for an `acquisition` that isn't `EI` or `LCB`, and what
    `evaluate` and `multistart` raise; a `HybridBOError` carries the partial run as
    `error.run`.
    """
    # The settings the run keeps, the artifact, then the initial points
    settings = StandardBOSettings(
        gp=gp,
        acquisition=acquisition,
        n_starts=n_starts,
        n_starts_gp=n_starts_gp,
        solver=solver,
    )
    artifact = _method_artifact(
        problem, measure, rng, "standard_bo", n_iterations, settings
    )
    start = _starting_points(problem.system.u, n_initial, initial_points, rng)

    def trial_point(
        problem: Problem, history: History, rng: np.random.Generator
    ) -> tuple[np.ndarray, tuple[FittedGP, ...], AcquisitionSolution]:
        """Fits the GP to the points so far and solves the acquisition problem."""
        fitted = (fit_gp(gp, history.u, history.f, rng, n_starts_gp, solver),)
        acquisition_problem = _acquisition_problem(
            problem,
            history,
            n_evaluated=len(history.u),
            surrogate=fitted,
            scenarios=None,
            settings=settings,
        )
        starts = lhs(*acquisition_problem.start_bounds, n_starts, rng)
        solution = multistart(acquisition_problem.nlp, starts, solver)
        return solution.w, fitted, AcquisitionSolution(solution.f)

    return optimization_loop(
        problem,
        measure,
        start,
        n_iterations,
        trial_point,
        rng,
        artifact,
        settings=settings,
        scenarios=None,
    )


def hybrid_bo(
    problem: Problem,
    measure: Measure,
    *,
    n_iterations: int,
    rng: np.random.Generator,
    gp: GP,
    acquisition: EI | LCB,
    approximation: SAA | SoftSAA | Linearization,
    n_initial: int | None = None,
    initial_points: np.ndarray | None = None,
    n_starts: int = 10,
    n_starts_gp: int = 10,
    solver: Ipopt = Ipopt(),  # noqa: B008  (frozen, never mutated)
) -> Run:
    """Hybrid BO: one GP per unknown state over the arguments of h, combined with the
    known equations in the acquisition problem (paper, Problems 8 and 9).

    Starts from `initial_points` (n, N), or `n_initial` points drawn from `rng`. Draws
    the scenarios (SAA and SoftSAA: `approximation.n_scenarios` standard normal draws,
    equal weights; linearization draws nothing and has the nominal scenario alone), the
    same in every iteration. Then evaluates the initial points, and in each iteration
    fits the prior GP `gp` to each unknown state's
    measured and reconstructed values so far, in the order of `system.y`, solves
    `hybrid_acquisition` by multistart from Latin hypercube starts over the start
    bounds of its variables, and evaluates the decision variables of the solution.
    The run holds the scenarios, each iteration's surrogate holds one fitted GP per
    unknown state, and each iteration's acquisition solution holds the scenario
    states at its trial point.

    Raises `ValueError` unless exactly one of `n_initial` and `initial_points` is
    given, what `HybridBOSettings` raises for the acquisition function and its
    approximation, and what `evaluate` and `multistart` raise; a `HybridBOError`
    carries the partial run, with the scenarios, as `error.run`.
    """
    # The settings the run keeps, the artifact, then the initial points
    system = problem.system
    settings = HybridBOSettings(
        gp=gp,
        acquisition=acquisition,
        approximation=approximation,
        n_starts=n_starts,
        n_starts_gp=n_starts_gp,
        solver=solver,
    )
    artifact = _method_artifact(
        problem, measure, rng, "hybrid_bo", n_iterations, settings
    )
    start = _starting_points(system.u, n_initial, initial_points, rng)

    # The scenarios: SAA and SoftSAA draw them, linearization has the nominal
    # scenario alone
    match approximation:
        case SAA(n_scenarios=n_scenarios) | SoftSAA(n_scenarios=n_scenarios):
            scenarios = saa_scenarios(n_scenarios, system.n_y, rng)
        case Linearization():
            scenarios = nominal_scenario(system.n_y)
        case _:
            assert_never(approximation)

    def trial_point(
        problem: Problem, history: History, rng: np.random.Generator
    ) -> tuple[np.ndarray, tuple[FittedGP, ...], AcquisitionSolution]:
        """Fits the GPs to the points so far and solves the acquisition problem."""
        fitted = fit_surrogate(
            system, history, gp=gp, rng=rng, n_starts=n_starts_gp, solver=solver
        )
        acquisition_problem = _acquisition_problem(
            problem,
            history,
            n_evaluated=len(history.u),
            surrogate=fitted,
            scenarios=scenarios,
            settings=settings,
        )
        starts = lhs(*acquisition_problem.start_bounds, n_starts, rng)
        solution = multistart(acquisition_problem.nlp, starts, solver)
        u, x, y = _unpack_scenarios(acquisition_problem, solution.w)
        return u, fitted, AcquisitionSolution(solution.f, x, y)

    return optimization_loop(
        problem,
        measure,
        start,
        n_iterations,
        trial_point,
        rng,
        artifact,
        settings=settings,
        scenarios=scenarios,
    )


def iteration_problem(run: Run, problem: Problem, i: int) -> AcquisitionProblem:
    """Iteration i's acquisition problem (from 0), rebuilt as the run's method built
    it: from the iteration's stored surrogate, the run's scenarios, the incumbent
    before trial point i and the run's settings.

    Raises `ValueError` for a baseline's run, which solves no acquisition problem.
    """
    surrogate = run.surrogates[i]
    if run.settings is None:
        msg = f"{run.method} is a baseline, which solves no acquisition problem"
        raise ValueError(msg)
    return _acquisition_problem(
        problem,
        run.history,
        n_evaluated=run.history.n_initial + i,
        surrogate=surrogate,
        scenarios=run.scenarios,
        settings=run.settings,
    )
