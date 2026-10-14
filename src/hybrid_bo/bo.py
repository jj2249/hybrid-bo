"""Bayesian optimization: the optimization loop shared by the BO methods, and standard
BO.

A BO method is a settings dataclass plus a function that runs `optimization_loop` with
its own trial-point function. The loop evaluates the initial points, then in each
iteration asks the trial-point function for the next trial point and evaluates it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, fields

import numpy as np

from hybrid_bo.acquisition import EI, LCB, _check_acquisition, standard_acquisition
from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Experiment, evaluate
from hybrid_bo.gp import GP, FittedGP, _gp_record, default_gp, fit_gp
from hybrid_bo.history import History, _concatenate, incumbents
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, run_record
from hybrid_bo.sampling import lhs
from hybrid_bo.solvers import Ipopt, multistart

# Chooses the next trial point (N,) from the points evaluated so far, drawing from the
# run's generator, and returns it with the fitted surrogate model behind it
TrialPoint = Callable[
    [Problem, History, np.random.Generator], tuple[np.ndarray, FittedGP]
]


def optimization_loop(
    problem: Problem,
    experiment: Experiment,
    initial_points: np.ndarray,
    n_iterations: int,
    trial_point: TrialPoint,
    rng: np.random.Generator,
    record: dict[str, str],
) -> Run:
    """Evaluates the initial points (n, N), then runs `n_iterations` iterations: each
    calls `trial_point` and evaluates the trial point it returns.

    Returns the run with `record`, and one surrogate per iteration (the one behind
    that iteration's trial point). For writing new methods: `record` comes from
    `run_record`, built before the method draws anything.

    Raises what `evaluate` and `trial_point` raise; a `HybridBOError` carries the
    partial run, every point evaluated before it, as `error.run`.
    """
    history: History | None = None
    surrogates: list[FittedGP] = []
    try:
        history = evaluate(problem, experiment, initial_points, rng)
        for _ in range(n_iterations):
            u, surrogate = trial_point(problem, history, rng)
            later = evaluate(problem, experiment, u[np.newaxis], rng)
            history = _concatenate(history, later)
            surrogates.append(surrogate)
    except HybridBOError as error:
        # Before the loop, the initial points evaluated so far are the partial run
        partial = error.history if history is None else history
        if partial is not None:
            error.run = Run(partial, record, surrogates)
        raise
    return Run(history, record, surrogates)


@dataclass(frozen=True)
class StandardBOSettings:
    """Settings of standard BO: the GP of u -> f (None: `default_gp` over the
    decision variables), the acquisition function, the number of multistart starts
    for the acquisition problem and for fitting the GP, and the solver for both.

    Raises `TypeError` unless `acquisition` is an `EI` or an `LCB`.
    """

    gp: GP | None = None
    acquisition: EI | LCB = field(default_factory=EI)
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = field(default_factory=Ipopt)

    def __post_init__(self) -> None:
        _check_acquisition(self.acquisition)


def standard_bo(
    problem: Problem,
    experiment: Experiment,
    initial_points: np.ndarray,
    n_iterations: int,
    settings: StandardBOSettings,
    rng: np.random.Generator,
) -> Run:
    """Standard BO: a GP of the objective over the decision variables, u -> f, with
    the acquisition function computed analytically.

    Evaluates the initial points (n, N), then in each iteration fits the GP to the
    measured objective so far, solves the acquisition problem by multistart from
    Latin hypercube starts over the start bounds of `u`, and evaluates the solution.

    Raises what `evaluate` and `multistart` raise; a `HybridBOError` carries the
    partial run as `error.run`.
    """
    # The GP and the record. The GP is recorded as text in its own entries, since its
    # repr would only name the kernel's function
    u_names = [var.name for var in problem.system.u]
    gp = default_gp(len(u_names)) if settings.gp is None else settings.gp
    settings_text = ", ".join(
        [f"n_iterations={n_iterations}"]
        + [
            f"{f.name}={getattr(settings, f.name)!r}"
            for f in fields(settings)
            if f.name != "gp"
        ]
    )
    record = run_record(problem, experiment, rng, "standard_bo", settings_text)
    record |= _gp_record(gp, u_names)

    def trial_point(
        problem: Problem, history: History, rng: np.random.Generator
    ) -> tuple[np.ndarray, FittedGP]:
        """Fits the GP to the points so far and solves the acquisition problem."""
        fitted = fit_gp(
            gp, history.u, history.f, rng, settings.n_starts_gp, settings.solver
        )
        incumbent = float(incumbents(history)[-1])
        nlp, start_bounds = standard_acquisition(
            problem, gp, fitted, incumbent, settings.acquisition
        )
        starts = lhs(*start_bounds, settings.n_starts, rng)
        return multistart(nlp, starts, settings.solver).w, fitted

    return optimization_loop(
        problem, experiment, initial_points, n_iterations, trial_point, rng, record
    )
