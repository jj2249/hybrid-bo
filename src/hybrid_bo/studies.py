"""Studies: runs of one or more methods on one problem, where run i of every method
starts from the same initial points.

`repeat` runs one method `n_runs` times from generators spawned from one seed; `study`
runs every method from the same seed, and its artifact records the problem, the
methods and the stated oracle optimum once. Save and load a study with `hb.save` and
`hb.load`.
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
# n is the number of iterations (of design points for the baselines). A BO method
# runs with its default settings; bind others with
# `functools.partial(hb.hybrid_bo, settings=...)`
Method = Callable[..., Run]


@dataclass(frozen=True)
class StudySettings:
    """How many runs of each method, with how many initial points and iterations.

    `seed` is where every random draw of the study comes from: the same seed gives
    the same runs. None draws a fresh seed, which `study` keeps in its settings and
    every run records.
    """

    n_runs: int
    n_initial: int
    n_iterations: int
    seed: int | None = None


@dataclass(frozen=True)
class Study:
    """Runs of one or more methods on one problem, as a flat collection: each run
    names its method (`run.method`) and its position among that method's runs
    (`run.index`), and run i of every method starts from the same initial points.

    `artifact` says, as text, what the study ran and how: the problem, the
    measurement, each method with its settings (a BO method's prior GP included,
    never the fitted GPs, which are in the runs), the oracle optimum (if stated) and
    the versions. The seed is in `settings`.
    """

    runs: tuple[Run, ...]
    settings: StudySettings
    name: str | None
    artifact: dict[str, str]


def repeat(
    method: Method,
    problem: Problem,
    measure: Measure,
    settings: StudySettings,
) -> tuple[Run, ...]:
    """Runs `method` `settings.n_runs` times and returns the runs with their indices.

    Run i draws its initial points and runs the method with the ith generator spawned
    from `settings.seed` (`SeedSequence(seed).spawn(n_runs)[i]`), so the same seed
    gives the same runs, and run i of any method starts from the same initial points.
    Seed None draws a fresh seed, which every run records. A run that raises a
    `HybridBOError` is logged as a warning and kept as far as it got (`error.run`);
    the next run goes on. Each finished run is logged at info level with its time.
    """
    runs = []
    children = np.random.SeedSequence(settings.seed).spawn(settings.n_runs)
    for index, child in enumerate(children):
        start = time.perf_counter()
        rng = np.random.default_rng(child)
        u = initial_points(problem.system.u, settings.n_initial, rng)
        try:
            run = method(problem, measure, u, settings.n_iterations, rng=rng)
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
        runs.append(replace(run, index=index))
    return tuple(runs)


def study(
    methods: Mapping[str, Method],
    problem: Problem,
    measure: Measure,
    settings: StudySettings,
    name: str | None = None,
    optimum: float | None = None,
) -> Study:
    """Runs every method with `repeat` from the same seed, so run i of every method
    starts from the same initial points, and the same seed gives the same study.
    Each run is labeled with its key in `methods`, so two configurations of one
    method can share a study. Logs each key at info level as its runs start.

    Seed None draws a fresh seed first, and the study's settings hold it, so the
    saved study can be run again exactly. `optimum` is the oracle optimum of a test
    case, which regret is measured from. It is stated, never computed, since a search
    can miss it (`oracle_optimum` helps find it once); the artifact records it. Warns
    with `OracleOptimumWarning` if an evaluated point's oracle objective is below it.
    """
    # One seed for every method; a fresh one is kept in the settings
    seed = np.random.SeedSequence(settings.seed).entropy
    settings = replace(settings, seed=seed)

    # Every method's runs, labeled with their key
    runs: list[Run] = []
    for label, method in methods.items():
        _logger.info("Method %s: %d runs", label, settings.n_runs)
        method_runs = repeat(method, problem, measure, settings)
        runs += [replace(run, method=label) for run in method_runs]

    # Each method with its settings, from the artifact of one of its runs; a BO
    # method's prior GP follows on its own line
    artifact = _problem_artifact(problem, measure)
    run_artifacts = {run.method: run.artifact for run in runs}
    method_lines = []
    for label in methods:
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
