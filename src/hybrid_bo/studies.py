"""Studies: runs of one or more methods on one problem, where run i of every method
starts from the same initial points.

`repeat` runs one method `n_runs` times from generators spawned from one seed; `study`
runs every method from the same seed and records the problem, the methods and the
oracle optimum once. Save and load a study with `hb.save` and
`hb.load`.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Experiment, oracle_optimum
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, _problem_record, _versions_record
from hybrid_bo.sampling import initial_points

_logger = logging.getLogger(__name__)

# A method, called as `method(problem, experiment, initial_points, n, rng=rng)`, where
# n is the number of iterations (of design points for the baselines). Bind a BO
# method's settings with `functools.partial(hb.hybrid_bo, settings=...)`
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

    `record` holds, as text, what the runs share: the problem, the experiment, each
    method with its settings, the oracle optimum (if any) with whether it was stated
    or computed, and the versions. The seed is in `settings`.
    """

    runs: tuple[Run, ...]
    settings: StudySettings
    name: str | None
    record: dict[str, str]


def repeat(
    method: Method,
    problem: Problem,
    experiment: Experiment,
    settings: StudySettings,
) -> tuple[Run, ...]:
    """Runs `method` `settings.n_runs` times and returns the runs with their indices.

    Run i draws its initial points and runs the method with the ith generator spawned
    from `settings.seed` (`SeedSequence(seed).spawn(n_runs)[i]`), so the same seed
    gives the same runs, and run i of any method starts from the same initial points.
    Seed None draws a fresh seed, which every run records. A run that raises a
    `HybridBOError` is logged as a warning and kept as far as it got (`error.run`);
    the next run goes on.
    """
    runs = []
    children = np.random.SeedSequence(settings.seed).spawn(settings.n_runs)
    for index, child in enumerate(children):
        rng = np.random.default_rng(child)
        u = initial_points(problem.system.u, settings.n_initial, rng)
        try:
            run = method(problem, experiment, u, settings.n_iterations, rng=rng)
        except HybridBOError as error:
            _logger.warning("Run %d failed: %r", index, error)
            if error.run is None:
                continue
            run = error.run
        runs.append(replace(run, index=index))
    return tuple(runs)


def study(
    methods: Mapping[str, Method],
    problem: Problem,
    experiment: Experiment,
    settings: StudySettings,
    name: str | None = None,
    optimum: float | None = None,
    oracle: Callable[[Any], Any] | None = None,
) -> Study:
    """Runs every method with `repeat` from the same seed, so run i of every method
    starts from the same initial points, and the same seed gives the same study.
    Each run is labeled with its key in `methods`, so two configurations of one
    method can share a study.

    Seed None draws a fresh seed first, and the study's settings hold it, so the
    saved study can be run again exactly. The oracle optimum is `optimum` when given;
    otherwise, when `oracle` (written as for `oracle_experiment`) is given, it is
    computed with `oracle_optimum` from the generator spawned after the runs' (child
    `n_runs` of the seed). The study records it and whether it was stated or
    computed.
    """
    # One seed for every method; a fresh one is kept in the settings
    seed = np.random.SeedSequence(settings.seed).entropy
    settings = replace(settings, seed=seed)

    # Every method's runs, labeled with their key
    runs: list[Run] = []
    for label, method in methods.items():
        method_runs = repeat(method, problem, experiment, settings)
        runs += [replace(run, method=label) for run in method_runs]

    # Each method with its settings, from the record of one of its runs
    record = _problem_record(problem, experiment)
    records = {run.method: run.record for run in runs}
    method_lines = []
    for label in methods:
        if label in records:
            text = f"{records[label]['method']}({records[label]['settings']})"
        else:
            text = "no runs"
        method_lines.append(f"{label}: {text}")
    record["methods"] = "\n".join(method_lines)

    # The oracle optimum, stated or computed from the oracle
    if optimum is not None:
        record["oracle_optimum"] = repr(float(optimum))
        record["oracle_optimum_source"] = "stated"
    elif oracle is not None:
        child = np.random.SeedSequence(seed, spawn_key=(settings.n_runs,))
        value = oracle_optimum(problem, oracle, np.random.default_rng(child))
        record["oracle_optimum"] = repr(value)
        record["oracle_optimum_source"] = "computed"

    return Study(tuple(runs), settings, name, record | _versions_record())
