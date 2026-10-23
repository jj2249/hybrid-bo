"""Studies: runs of one or more configurations on one problem, where run i of every
configuration with the same number of initial points starts from the same initial
points.

A configuration is a method with its own evaluation budget. `repeat` runs one
configuration `n_runs` times from generators spawned from one seed; `study` runs every
configuration from the same seed, and its artifact records the problem, the methods and
the stated oracle optimum once. Save and load a study with `hb.save` and `hb.load`.
"""

import logging
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from hybrid_bo.errors import HybridBOError, OracleOptimumWarning
from hybrid_bo.evaluation import Measure
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, _problem_artifact, _versions_artifact
from hybrid_bo.sampling import initial_points

_logger = logging.getLogger(__name__)

# A method, called as `method(problem, measure, initial_points, n, rng=rng)`, where
# n is the number of iterations (of design points for the baselines). Bind a BO
# method's settings with `functools.partial(hb.hybrid_bo, settings=...)`
Method = Callable[..., Run]


@dataclass(frozen=True, kw_only=True)
class Configuration:
    """A method with its own evaluation budget: `n_evaluations` evaluations in all,
    the first `n_initial` of them initial points, so it runs `n_evaluations -
    n_initial` iterations (design points for the baselines).

    Raises `ValueError` unless 1 <= `n_initial` <= `n_evaluations`.
    """

    method: Method
    n_evaluations: int
    n_initial: int

    def __post_init__(self) -> None:
        if not 1 <= self.n_initial <= self.n_evaluations:
            msg = (
                "A configuration needs 1 <= n_initial <= n_evaluations, not "
                f"n_initial={self.n_initial}, n_evaluations={self.n_evaluations}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class StudySettings:
    """How many runs of each configuration, and the seed they come from. Both are
    the same for every configuration, so run i of one is paired with run i of the
    others.

    `seed` is where every random draw of the study comes from: the same seed gives
    the same runs. None draws a fresh seed, which `study` keeps in its settings and
    every run records.
    """

    n_runs: int
    seed: int | None = None


@dataclass(frozen=True)
class Study:
    """Runs of one or more configurations on one problem, as a flat collection: each
    run names its configuration (`run.method`, the configuration's label) and its
    position among that configuration's runs (`run.index`), and run i of every
    configuration with the same number of initial points starts from the same
    initial points.

    Each run records its configuration's budget (`run.n_evaluations`). `artifact`
    says, as text, what the study ran and how: the problem, the measurement, each method with
    its settings (a BO method's prior GP included, never the fitted GPs, which are in
    the runs), the oracle optimum (if stated) and the versions. The seed is in
    `settings`.
    """

    runs: tuple[Run, ...]
    settings: StudySettings
    name: str | None
    artifact: dict[str, str]


def repeat(
    configuration: Configuration,
    problem: Problem,
    measure: Measure,
    settings: StudySettings,
) -> tuple[Run, ...]:
    """Runs the configuration `settings.n_runs` times and returns the runs with their
    indices.

    Run i draws its initial points and runs the method with the ith generator spawned
    from `settings.seed` (`SeedSequence(seed).spawn(n_runs)[i]`), so the same seed
    gives the same runs, and run i of any configuration with the same `n_initial`
    starts from the same initial points. Seed None draws a fresh seed, which every
    run records, as it records the configuration's `n_evaluations`. A run that
    raises a `HybridBOError` is logged as a warning and kept as far as it got
    (`error.run`); the next run goes on. Each finished run is logged
    at info level with its time.
    """
    runs = []
    n_iterations = configuration.n_evaluations - configuration.n_initial
    children = np.random.SeedSequence(settings.seed).spawn(settings.n_runs)
    for index, child in enumerate(children):
        start = time.perf_counter()
        rng = np.random.default_rng(child)
        u = initial_points(problem.system.u, configuration.n_initial, rng)
        try:
            run = configuration.method(problem, measure, u, n_iterations, rng=rng)
        except HybridBOError as error:
            _logger.warning(
                "Run %d of %d failed after %.2fs: %r",
                index + 1,
                settings.n_runs,
                time.perf_counter() - start,
                error,
            )
            if error.run is None:
                continue
            run = error.run
        else:
            _logger.info(
                "Run %d of %d finished in %.2fs",
                index + 1,
                settings.n_runs,
                time.perf_counter() - start,
            )
        runs.append(
            replace(run, index=index, n_evaluations=configuration.n_evaluations)
        )
    return tuple(runs)


def study(
    configurations: Mapping[str, Configuration],
    problem: Problem,
    measure: Measure,
    settings: StudySettings,
    name: str | None = None,
    optimum: float | None = None,
) -> Study:
    """Runs every configuration with `repeat` from the same seed, so run i of every
    configuration with the same `n_initial` starts from the same initial points, and
    the same seed gives the same study. Each run is labeled with its configuration's
    key, so one method can appear under several labels, with different settings or
    budgets. Logs each label at info level as its runs start.

    Seed None draws a fresh seed first, and the study's settings hold it, so the
    saved study can be run again exactly. `optimum` is the oracle optimum of a test
    case, which regret is measured from. It is stated, never computed, since a search
    can miss it (`oracle_optimum` helps find it once); the artifact records it. Warns
    with `OracleOptimumWarning` if an evaluated point's oracle objective is below it.
    """
    # One seed for every configuration; a fresh one is kept in the settings
    seed = np.random.SeedSequence(settings.seed).entropy
    settings = replace(settings, seed=seed)

    # Every configuration's runs, labeled with their key
    runs: list[Run] = []
    for label, configuration in configurations.items():
        _logger.info(
            "Configuration %s: %d runs of %d evaluations, %d of them initial points",
            label,
            settings.n_runs,
            configuration.n_evaluations,
            configuration.n_initial,
        )
        configuration_runs = repeat(configuration, problem, measure, settings)
        runs += [replace(run, method=label) for run in configuration_runs]

    # Each method with its settings, from the artifact of one of its runs; a BO
    # method's prior GP follows on its own line
    artifact = _problem_artifact(problem, measure)
    run_artifacts = {run.method: run.artifact for run in runs}
    method_lines = []
    for label in configurations:
        if label in run_artifacts:
            run_artifact = run_artifacts[label]
            text = f"{run_artifact['method']}({run_artifact['settings']})"
            if "gp" in run_artifact:
                text += f"\n  gp: {run_artifact['gp']}"
        else:
            text = "no runs"
        method_lines.append(f"{label}: {text}")
    artifact["methods"] = "\n".join(method_lines)

    # The stated oracle optimum, checked against the points the runs evaluated
    if optimum is not None:
        artifact["oracle_optimum"] = repr(float(optimum))
        _check_optimum(runs, float(optimum))
    return Study(tuple(runs), settings, name, artifact | _versions_artifact())


def _check_optimum(runs: list[Run], optimum: float) -> None:
    """Warns with `OracleOptimumWarning` if a run evaluated a point whose oracle
    objective is below `optimum`, beyond the solvers' tolerance."""
    # The oracle objectives of every evaluated point; none without an oracle
    oracle_values = [
        run.history.f_oracle
        for run in runs
        if run.history.f_oracle is not None and len(run.history.f_oracle) > 0
    ]
    if not oracle_values:
        return

    # Below the optimum by more than a relative 1e-6
    lowest = float(np.min(np.concatenate(oracle_values)))
    tolerance = 1e-6 * max(1.0, abs(optimum))
    beats_optimum = lowest < optimum - tolerance
    if beats_optimum:
        msg = (
            f"An evaluated point has oracle objective {lowest!r}, below the stated "
            f"oracle optimum {optimum!r}, so the regret is wrong"
        )
        warnings.warn(msg, OracleOptimumWarning, stacklevel=3)
