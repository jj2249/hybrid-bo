"""Studies: runs of one or more methods on one problem, where run i of every method
starts from the same initial points.

`repeat` runs one method `n_runs` times from generators spawned from one generator;
`study` runs every method from the same spawned generators and records the problem,
the methods and the oracle optimum once. Save and load a study with `hb.save` and
`hb.load`.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Experiment, oracle_optimum
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, _generator_record, _problem_record, _versions_record
from hybrid_bo.sampling import initial_points

_logger = logging.getLogger(__name__)

# A method, called as `method(problem, experiment, initial_points, n, rng=rng)`, where
# n is the number of iterations (of design points for the baselines). Bind a BO
# method's settings with `functools.partial(hb.hybrid_bo, settings=...)`
Method = Callable[..., Run]


@dataclass(frozen=True)
class StudySettings:
    """How many runs of each method, with how many initial points and iterations.

    `seed` is what the caller builds the study's generator from
    (`np.random.default_rng(seed)`); None draws a fresh seed. The study and every
    run record the seed they used.
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
    method with its settings, the study's generator, the oracle optimum (if any)
    with whether it was stated or computed, and the versions.
    """

    runs: tuple[Run, ...]
    settings: StudySettings
    name: str | None
    record: dict[str, str]


def _copy(rng: np.random.Generator) -> np.random.Generator:
    """A new generator whose seed sequence is in the state of `rng`'s, so it spawns
    the same children `rng` would spawn next.

    Raises `TypeError` unless `rng` was seeded through a `SeedSequence`, as
    `np.random.default_rng` seeds it.
    """
    seed_sequence = rng.bit_generator.seed_seq
    if not isinstance(seed_sequence, np.random.SeedSequence):
        msg = f"A study needs a generator seeded through a SeedSequence, got {rng!r}"
        raise TypeError(msg)
    copy = np.random.SeedSequence(
        seed_sequence.entropy,
        spawn_key=seed_sequence.spawn_key,
        n_children_spawned=seed_sequence.n_children_spawned,
    )
    return np.random.Generator(type(rng.bit_generator)(copy))


def repeat(
    method: Method,
    problem: Problem,
    experiment: Experiment,
    settings: StudySettings,
    rng: np.random.Generator,
) -> tuple[Run, ...]:
    """Runs `method` `settings.n_runs` times, run i with the ith generator spawned
    from `rng` and initial points drawn from it, and returns the runs with their
    indices.

    A run that raises a `HybridBOError` is logged as a warning and kept as far as it
    got (`error.run`); the next run goes on.
    """
    runs = []
    for index, run_rng in enumerate(rng.spawn(settings.n_runs)):
        u = initial_points(problem.system.u, settings.n_initial, run_rng)
        try:
            run = method(problem, experiment, u, settings.n_iterations, rng=run_rng)
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
    rng: np.random.Generator,
    name: str | None = None,
    optimum: float | None = None,
) -> Study:
    """Runs every method with `repeat`, each from the same generators spawned from
    `rng`, so run i of every method starts from the same initial points. Each run is
    labeled with its key in `methods`, so two configurations of one method can share
    a study.

    The oracle optimum is `optimum` when given; otherwise, when the experiment is an
    oracle simulation, it is computed with `oracle_optimum` from the next generator
    spawned from `rng`. The study records it and whether it was stated or computed.
    Raises `TypeError` unless `rng` was seeded through a `SeedSequence`.
    """
    # The record of the study's generator, before anything is spawned from it
    record = _problem_record(problem, experiment) | _generator_record(rng)

    # Every method from a copy of rng, so each spawns the same generators; then
    # spawn them from rng itself, so it moves on as one `repeat` moves it
    runs: list[Run] = []
    for label, method in methods.items():
        method_runs = repeat(method, problem, experiment, settings, _copy(rng))
        runs += [replace(run, method=label) for run in method_runs]
    rng.spawn(settings.n_runs)

    # Each method with its settings, from its first run's record
    first_runs = {run.method: run.record for run in reversed(runs)}
    record["methods"] = "\n".join(
        f"{label}: {first_runs[label]['method']}({first_runs[label]['settings']})"
        if label in first_runs
        else f"{label}: no runs"
        for label in methods
    )

    # The oracle optimum, stated or computed from the oracle
    oracle = getattr(experiment, "oracle", None)
    if optimum is not None:
        record["oracle_optimum"] = repr(float(optimum))
        record["oracle_optimum_source"] = "stated"
    elif oracle is not None:
        record["oracle_optimum"] = repr(oracle_optimum(problem, oracle, rng))
        record["oracle_optimum_source"] = "computed"

    return Study(tuple(runs), settings, name, record | _versions_record())
