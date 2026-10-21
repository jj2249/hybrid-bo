"""Bayesian optimization: the optimization loop shared by the BO methods, standard BO
and hybrid BO.

A BO method is a settings dataclass plus a function that runs `optimization_loop` with
its own trial-point function. The loop evaluates the initial points, then in each
iteration asks the trial-point function for the next trial point and evaluates it.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import fields, replace
from typing import assert_never

import numpy as np

from hybrid_bo.acquisition import (
    SAA,
    AcquisitionSolution,
    Linearization,
    SoftSAA,
    _scenario_states,
    _unpack,
    hybrid_acquisition,
    nominal_scenario,
    saa_scenarios,
    standard_acquisition,
)
from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Measure, evaluate
from hybrid_bo.gp import FittedGP, fit_gp
from hybrid_bo.history import History, _concatenate, incumbents
from hybrid_bo.problem import Problem, _h_inputs
from hybrid_bo.run import Run, run_artifact
from hybrid_bo.sampling import lhs
from hybrid_bo.settings import HybridBOSettings, MethodSettings, StandardBOSettings
from hybrid_bo.solvers import multistart

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
) -> Run:
    """Evaluates the initial points (n, N), then runs `n_iterations` iterations: each
    calls `trial_point` and evaluates the trial point it returns.

    Returns the run with `artifact` and `settings`, labeled with the method the artifact
    names, and per iteration the surrogate and the acquisition problem's solution
    behind that iteration's trial point. For writing new methods: `artifact` comes
    from `run_artifact`, built before the method draws anything, and `settings` are
    the method's, which the iteration plots replay (None for a method they can't
    replay).

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
            error.run = Run(
                partial,
                artifact,
                artifact["method"],
                surrogates=surrogates,
                acquisitions=acquisitions,
                settings=settings,
            )
        raise
    return Run(
        history,
        artifact,
        artifact["method"],
        surrogates=surrogates,
        acquisitions=acquisitions,
        settings=settings,
    )


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


def standard_bo(
    problem: Problem,
    measure: Measure,
    initial_points: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
    *,
    settings: StandardBOSettings,
) -> Run:
    """Standard BO: a GP of the objective over the decision variables, u -> f, with
    the acquisition function computed analytically.

    Evaluates the initial points (n, N), then in each iteration fits the GP to the
    measured objective so far, solves the acquisition problem by multistart from
    Latin hypercube starts over the start bounds of `u`, and evaluates the solution.
    Each iteration's acquisition solution holds the acquisition value only.
    `settings` are keyword-only.

    Raises what `evaluate` and `multistart` raise; a `HybridBOError` carries the
    partial run as `error.run`.
    """
    # The artifact
    gp = settings.gp
    artifact = _method_artifact(
        problem, measure, rng, "standard_bo", n_iterations, settings
    )

    def trial_point(
        problem: Problem, history: History, rng: np.random.Generator
    ) -> tuple[np.ndarray, tuple[FittedGP, ...], AcquisitionSolution]:
        """Fits the GP to the points so far and solves the acquisition problem."""
        fitted = fit_gp(
            gp, history.u, history.f, rng, settings.n_starts_gp, settings.solver
        )
        incumbent = float(incumbents(history)[-1])
        acquisition_problem = standard_acquisition(
            problem, fitted, incumbent, settings.acquisition
        )
        starts = lhs(*acquisition_problem.start_bounds, settings.n_starts, rng)
        solution = multistart(acquisition_problem.nlp, starts, settings.solver)
        u, _, _ = _unpack(acquisition_problem, solution.w)
        return u, (fitted,), AcquisitionSolution(solution.f)

    return optimization_loop(
        problem,
        measure,
        initial_points,
        n_iterations,
        trial_point,
        rng,
        artifact,
        settings=settings,
    )


def hybrid_bo(
    problem: Problem,
    measure: Measure,
    initial_points: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
    *,
    settings: HybridBOSettings,
) -> Run:
    """Hybrid BO: one GP per unknown state over the arguments of h, combined with the
    known equations in the acquisition problem (paper, Problems 8 and 9).

    Draws the scenarios (SAA and SoftSAA: `settings.approximation.n_scenarios`
    standard normal draws, equal weights; linearization draws nothing and has the
    nominal scenario alone), the same in every iteration. Then evaluates the initial
    points (n, N), and in each iteration fits the prior GP to each unknown state's
    measured and reconstructed values so far, in the order of `system.y`, solves
    `hybrid_acquisition` by multistart from Latin hypercube starts over the start
    bounds of its variables, and evaluates the decision variables of the solution.
    The run holds the scenarios, each iteration's surrogate holds one fitted GP per
    unknown state, and each iteration's acquisition solution holds the scenario
    states at its trial point. `settings` are keyword-only.

    Raises what `evaluate` and `multistart` raise; a `HybridBOError` carries the
    partial run, with the scenarios, as `error.run`.
    """
    # The artifact, before the scenarios are drawn
    system, gp = problem.system, settings.gp
    artifact = _method_artifact(
        problem, measure, rng, "hybrid_bo", n_iterations, settings
    )
    # The scenarios: SAA and SoftSAA draw them, linearization has the nominal
    # scenario alone
    approximation = settings.approximation
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
        # One GP per unknown state, in the order of y, on the same arguments of h
        X = np.column_stack(_h_inputs(system, history.u, history.x))
        fitted = tuple(
            fit_gp(gp, X, y, rng, settings.n_starts_gp, settings.solver)
            for y in history.y.T
        )
        incumbent = float(incumbents(history)[-1])
        acquisition_problem = hybrid_acquisition(
            problem,
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
            system, fitted, scenarios, acquisition_problem, solution.w
        )
        return u, fitted, AcquisitionSolution(solution.f, x, y)

    try:
        run = optimization_loop(
            problem,
            measure,
            initial_points,
            n_iterations,
            trial_point,
            rng,
            artifact,
            settings=settings,
        )
    except HybridBOError as error:
        if error.run is not None:
            error.run = replace(error.run, scenarios=scenarios)
        raise
    return replace(run, scenarios=scenarios)
