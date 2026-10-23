"""Studies: runs of one or more configurations on one problem, where run i of every
configuration with the same number of initial points starts from the same initial
points.

A configuration is a method with its own evaluation budget. `repeat` runs one
configuration `n_runs` times from generators spawned from one seed; `study` runs every
configuration from the same seed, in this process or in worker processes, and its
artifact records the problem, the methods and the stated oracle optimum once. Save and load a study with `hb.save` and `hb.load`.
"""

import logging
import logging.handlers
import multiprocessing
import os
import pickle
import threading
import time
import warnings
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
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

# The study run this context is running, as (label, index), for `run_of`
_current_run: ContextVar[tuple[str, int] | None] = ContextVar(
    "_current_run", default=None
)

# The attribute that carries a worker's record's run to the parent
_RUN_ATTRIBUTE = "hybrid_bo_run"

# The environment variables that set the threads of BLAS and OpenMP, read when they
# load, so workers get them from the environment they start in
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


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
    tasks = _tasks({"": configuration}, settings)
    runs = [_run(task, problem, measure, settings) for task in tasks]
    return tuple(run for run in runs if run is not None)


@dataclass(frozen=True)
class _Task:
    """One run of a study: run `index` of the configuration labeled `label`, drawn
    from `child`, its spawned seed."""

    label: str
    configuration: Configuration
    index: int
    child: np.random.SeedSequence


def _tasks(
    configurations: Mapping[str, Configuration], settings: StudySettings
) -> list[_Task]:
    """The runs of every configuration, by label, then index. Each configuration
    spawns its own seeds from `settings.seed`, since `lhs` spawns from them."""
    return [
        _Task(label, configuration, index, child)
        for label, configuration in configurations.items()
        for index, child in enumerate(
            np.random.SeedSequence(settings.seed).spawn(settings.n_runs)
        )
    ]


def _run(
    task: _Task, problem: Problem, measure: Measure, settings: StudySettings
) -> Run | None:
    """The task's run, as `repeat` describes; None if it failed before evaluating
    anything."""
    configuration, index = task.configuration, task.index
    start = time.perf_counter()
    rng = np.random.default_rng(task.child)
    n_iterations = configuration.n_evaluations - configuration.n_initial
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
            return None
        run = error.run
    else:
        _logger.info(
            "Run %d of %d finished in %.2fs",
            index + 1,
            settings.n_runs,
            time.perf_counter() - start,
        )
    return replace(run, index=index, n_evaluations=configuration.n_evaluations)


def study(
    configurations: Mapping[str, Configuration],
    problem: Problem,
    measure: Measure,
    settings: StudySettings,
    name: str | None = None,
    optimum: float | None = None,
    *,
    n_workers: int = 1,
) -> Study:
    """Runs every configuration `settings.n_runs` times, as `repeat` does, from the
    same seed, so run i of every configuration with the same `n_initial` starts from
    the same initial points, and the same seed gives the same study. Each run is
    labeled with its configuration's key, so one method can appear under several
    labels, with different settings or budgets. Logs each label at info level before
    the runs start.

    Seed None draws a fresh seed first, and the study's settings hold it, so the
    saved study can be run again exactly. `optimum` is the oracle optimum of a test
    case, which regret is measured from. It is stated, never computed, since a search
    can miss it (`oracle_optimum` helps find it once); the artifact records it. Warns
    with `OracleOptimumWarning` if an evaluated point's oracle objective is below it.

    `n_workers` > 1 runs the runs in that many worker processes, one task per run,
    each with single-threaded BLAS and OpenMP; the study is the same as with 1 (this
    process), runs in the same order. The problem, the measurement and the methods
    are pickled to the workers, so they must be module-level functions or data. The
    workers' records are handled by this process's loggers at their levels, and their
    warnings raised again here after each run. While a run of the study runs, `run_of`
    gives the label and index of the run that logged a record, in this process or a
    worker. Raises `ValueError` if `n_workers` < 1, and `TypeError` naming what can't
    be pickled.
    """
    if n_workers < 1:
        msg = f"A study needs n_workers >= 1, not {n_workers}"
        raise ValueError(msg)

    # One seed for every configuration; a fresh one is kept in the settings
    seed = np.random.SeedSequence(settings.seed).entropy
    settings = replace(settings, seed=seed)

    # One task per run of each configuration, here or in workers, each run's records
    # tagged with it; the runs keep the tasks' order and are labeled with their key
    for label, configuration in configurations.items():
        _log_configuration(label, configuration, settings)
    tasks = _tasks(configurations, settings)
    if n_workers == 1:
        results = []
        for task in tasks:
            with _tagged(task.label, task.index):
                results.append(_run(task, problem, measure, settings))
    else:
        results = _parallel_results(tasks, problem, measure, settings, n_workers)
    runs = [
        replace(run, method=task.label)
        for task, run in zip(tasks, results, strict=True)
        if run is not None
    ]

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


def _log_configuration(
    label: str, configuration: Configuration, settings: StudySettings
) -> None:
    """Logs at info level that the configuration's runs start."""
    _logger.info(
        "Configuration %s: %d runs of %d evaluations, %d of them initial points",
        label,
        settings.n_runs,
        configuration.n_evaluations,
        configuration.n_initial,
    )


def _parallel_results(
    tasks: list[_Task],
    problem: Problem,
    measure: Measure,
    settings: StudySettings,
    n_workers: int,
) -> list[Run | None]:
    """The tasks' runs (`_run`), from `n_workers` worker processes, in the tasks'
    order.

    Raises `TypeError` if the problem, the measurement or a method can't be pickled,
    which sending it to a worker needs.
    """
    # Everything a worker gets must pickle; say which object doesn't, before running
    _check_picklable(problem, "problem")
    _check_picklable(measure, "measurement")
    for task in tasks:
        _check_picklable(task.configuration.method, f"method of {task.label}")

    # A thread that handles the workers' records here, until the pool is done
    context = multiprocessing.get_context("spawn")
    records = context.Queue()
    listener = threading.Thread(target=_reemit, args=(records,))
    listener.start()
    try:
        with ProcessPoolExecutor(
            n_workers,
            mp_context=context,
            initializer=_start_worker,
            initargs=(records, _levels()),
        ) as executor:
            # The pool starts its workers as tasks are submitted, so they inherit
            # single-threaded BLAS and OpenMP
            with _single_threaded_children():
                futures = [
                    executor.submit(_worker_run, task, problem, measure, settings)
                    for task in tasks
                ]

            # Each run's warnings raised again here, tagged with the run; a
            # registry per file shows a warning once per place, as serially. An
            # error cancels the runs not started yet
            results = []
            registries = {}
            try:
                for task, future in zip(tasks, futures, strict=True):
                    run, caught = future.result()
                    with _tagged(task.label, task.index):
                        for message, category, filename, lineno in caught:
                            warnings.warn_explicit(
                                message,
                                category,
                                filename,
                                lineno,
                                registry=registries.setdefault(filename, {}),
                            )
                    results.append(run)
            except BaseException:
                executor.shutdown(cancel_futures=True)
                raise
    finally:
        # The workers have exited, so their records are all queued before this
        records.put(None)
        listener.join()
    return results


def _check_picklable(value: object, what: str) -> None:
    """Raises `TypeError`, naming `what`, if `value` can't be pickled."""
    try:
        pickle.dumps(value)
    except (pickle.PicklingError, AttributeError, TypeError) as error:
        msg = (
            f"With n_workers > 1, the {what} is sent to worker processes, so it must "
            f"pickle, and {value!r} doesn't ({error}). Define it as a module-level "
            "function, not a lambda or a nested function"
        )
        raise TypeError(msg) from error


def _levels() -> dict[str, int]:
    """The levels set on this process's loggers, by name ("" for the root), for a
    worker to log at the same levels."""
    loggers = logging.root.manager.loggerDict.items()
    levels = {
        name: logger.level
        for name, logger in loggers
        if isinstance(logger, logging.Logger) and logger.level != logging.NOTSET
    }
    return levels | {"": logging.root.level}


def _start_worker(
    records: "multiprocessing.Queue[logging.LogRecord | None]", levels: dict[str, int]
) -> None:
    """Sets up a worker process: its loggers at the parent's levels, with every
    record sent to the parent through `records` instead of handled here."""
    for name, level in levels.items():
        logging.getLogger(name or None).setLevel(level)
    logging.root.handlers = [_Forward(records)]


def _reemit(records: "multiprocessing.Queue[logging.LogRecord | None]") -> None:
    """Handles the workers' records with this process's loggers, as if logged here,
    until it takes None."""
    while (record := records.get()) is not None:
        logger = logging.getLogger(None if record.name == "root" else record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


def run_of(record: logging.LogRecord) -> tuple[str, int] | None:
    """The configuration label and run index of the study run that logged `record`,
    or None for a record logged outside the runs of a study (`repeat` has no labels,
    so its runs have none either).

    Call it while handling the record, in a handler or a filter: a record from a
    worker process carries its run, and a record logged in this process gets it from
    the run that is logging.
    """
    return getattr(record, _RUN_ATTRIBUTE, None) or _current_run.get()


@contextmanager
def _tagged(label: str, index: int) -> Generator[None]:
    """Tags the records logged in the block with the run `(label, index)`."""
    token = _current_run.set((label, index))
    try:
        yield
    finally:
        _current_run.reset(token)


def _worker_run(
    task: _Task, problem: Problem, measure: Measure, settings: StudySettings
) -> tuple[Run | None, list[tuple[Warning | str, type[Warning], str, int]]]:
    """`_run` in a worker process, its records tagged with the run, and every
    warning it raised (message, category, file and line), for the parent to raise
    again."""
    with (
        _tagged(task.label, task.index),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run = _run(task, problem, measure, settings)
    return run, [(w.message, w.category, w.filename, w.lineno) for w in caught]


class _Forward(logging.handlers.QueueHandler):
    """Sends a worker's records to the parent process, each tagged with its run."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """The record as `QueueHandler` sends it (message formatted, nothing that
        might not pickle), carrying its run."""
        prepared = super().prepare(record)
        setattr(prepared, _RUN_ATTRIBUTE, _current_run.get())
        return prepared


@contextmanager
def _single_threaded_children() -> Generator[None]:
    """Sets one thread for BLAS and OpenMP in the environment of the processes
    started in the block, then puts this process's environment back as it was."""
    before = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(_THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
