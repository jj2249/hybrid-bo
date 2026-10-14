"""The non-adaptive baselines: Latin hypercube and uniform sampling.

Neither is sequential. Each draws its whole design over the start bounds of the
decision variables, then evaluates the initial points followed by the design.
"""

from dataclasses import replace

import numpy as np

from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Experiment, evaluate
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, run_record
from hybrid_bo.sampling import _start_bounds, lhs, uniform


def _run_design(
    problem: Problem,
    experiment: Experiment,
    initial_points: np.ndarray,
    design: np.ndarray,
    rng: np.random.Generator,
    record: dict[str, str],
) -> Run:
    """Evaluates the initial points (n_initial, N), then the design (n_points, N).

    Raises what `evaluate` raises; a `HybridBOError` carries the partial run as
    `error.run`.
    """
    n_initial = len(initial_points)
    try:
        history = evaluate(
            problem, experiment, np.concatenate([initial_points, design]), rng
        )
    except HybridBOError as error:
        if error.history is not None:
            n_evaluated = len(error.history.u)
            partial = replace(error.history, n_initial=min(n_initial, n_evaluated))
            error.run = Run(partial, record)
        raise
    return Run(replace(history, n_initial=n_initial), record)


def latin_hypercube_sampling(
    problem: Problem,
    experiment: Experiment,
    initial_points: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Run:
    """Evaluates the initial points (n, N), then `n_points` points from a Latin
    hypercube over the start bounds of `u`, drawn before any evaluation.

    Raises what `evaluate` raises; a `HybridBOError` carries the partial run as
    `error.run`.
    """
    record = run_record(
        problem, experiment, rng, "latin_hypercube_sampling", f"n_points={n_points}"
    )
    design = lhs(*_start_bounds(problem.system.u), n_points, rng)
    return _run_design(problem, experiment, initial_points, design, rng, record)


def uniform_sampling(
    problem: Problem,
    experiment: Experiment,
    initial_points: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Run:
    """Evaluates the initial points (n, N), then `n_points` points drawn uniformly
    from the start bounds of `u`, drawn before any evaluation.

    Raises what `evaluate` raises; a `HybridBOError` carries the partial run as
    `error.run`.
    """
    record = run_record(
        problem, experiment, rng, "uniform_sampling", f"n_points={n_points}"
    )
    design = uniform(*_start_bounds(problem.system.u), n_points, rng)
    return _run_design(problem, experiment, initial_points, design, rng, record)
