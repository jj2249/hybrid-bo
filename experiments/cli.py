"""The command line of an experiment: run its methods as a study, resume a study, run
one task of a study, or plot a saved study (`docs/running_large_studies.rst`).

An experiment ends with a call that passes everything it defines, its name being its
module's:

    if __name__ == "__main__":
        cli.main("illustrative_example", problem, measure, configurations,
                 study_settings, optimum=optimum, true_curves=True)

and the pixi tasks run it as a module from this folder (`python -m <name> ...`).
`run` (the default command) is `prepare` (reserve the study folder and write its task
list), then the launcher (`run_commands`) running each task as
`python -m <name> run-task`, then assembling the study from the run files. A study is
finished exactly when its `artifact.json` exists; `resume` runs the tasks left of one
that isn't, then assembles it.
"""

import argparse
import dataclasses
import importlib.util
import json
import logging
import os
import secrets
import signal
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import hybrid_bo as hb
from hybrid_bo.run import code_version, is_clean, problem_artifact
from hybrid_bo.saving import ARTIFACT_FILE, OLD_ARTIFACT_FILE, save_artifact

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_ROOT = EXPERIMENTS_DIR.parent / "results"
ARTIFACTS = (ARTIFACT_FILE, OLD_ARTIFACT_FILE)
TASK_LIST = "tasks.json"
STUDY_LOG = "study.log"
SUMMARY = "summary.csv"
PER_RUN_LOGS = "logs"
RUN_FILES = "runs"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
COMMANDS = ("run", "prepare", "run-task", "resume", "plot")

# The environment variables that set the threads of BLAS and OpenMP, read when they
# load, so a command gets them from the environment it starts in
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# How long stopped commands get to exit before they are killed, and how often the
# launcher checks on them, in seconds
STOP_TIMEOUT = 5.0
POLL_INTERVAL = 0.05

# The study figures `plot` draws, by file name
FIGURES: Mapping[str, Callable[[hb.Study], object]] = {
    "convergence": hb.plots.convergence,
    "regret": hb.plots.regret,
}

_logger = logging.getLogger(__name__)


class UsageError(Exception):
    """The command line asks for something that can't be done; `main` reports it as
    a usage error (exit status 2)."""


class CommandFailedError(Exception):
    """A command the launcher ran exited with a status other than 0."""

    def __init__(self, name: str, status: int) -> None:
        """The command `name` exited with `status`."""
        super().__init__(f"{name} failed with exit status {status}")
        self.name = name
        self.status = status


@dataclass(frozen=True, kw_only=True)
class Experiment:
    """What an experiment defines that its studies depend on: its name (its module's
    too), problem, measurement, configurations and stated oracle optimum."""

    name: str
    problem: hb.Problem
    measure: hb.Measure
    configurations: Mapping[str, hb.Configuration]
    optimum: float | None


@dataclass(frozen=True, kw_only=True)
class TaskList:
    """What a study runs, saved as `tasks.json` before anything runs: the seed (never
    None), the number of runs, each chosen configuration's budget, the oracle optimum,
    the code version, the problem's artifact text and the tasks as (label, index), in
    order."""

    name: str
    seed: int
    n_runs: int
    configurations: dict[str, dict[str, int]]
    optimum: float | None
    code_version: str
    problem: dict[str, str]
    tasks: tuple[tuple[str, int], ...]


def main(
    name: str,
    problem: hb.Problem,
    measure: hb.Measure,
    configurations: Mapping[str, hb.Configuration],
    study_settings: hb.StudySettings,
    *,
    optimum: float | None = None,
    true_curves: bool = False,
    argv: Sequence[str] | None = None,
) -> None:
    """Runs, prepares or resumes a study of the experiment, runs one of its tasks, or
    plots a saved study, as the command line (`argv`, default `sys.argv`) says.

    `name` names the experiment's module, its studies and their folder.
    `study_settings` are the defaults the flags override. `optimum` is the stated
    oracle optimum, which regret is measured from. `true_curves` draws the
    measurement's values in the iteration figures, which only an oracle can afford.

    A study that doesn't finish exits with status 130 (Ctrl-C) or 1, after saying how
    to resume it.
    """
    # The command: the first argument if it names one, else a run
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = (
        (args[0], args[1:]) if args[:1] and args[0] in COMMANDS else ("run", args)
    )
    parser = _parsers(name, configurations)[command]
    parsed = parser.parse_args(rest)
    experiment = Experiment(
        name=name,
        problem=problem,
        measure=measure,
        configurations=configurations,
        optimum=optimum,
    )
    experiment_dir = _from_init_cwd(parsed.results_root) / name

    try:
        # Plot: the study given, else the newest finished one
        if command == "plot":
            study_dir = _study_to_plot(experiment_dir, parsed.study)

        # One task, in this process, as a launcher started it
        elif command == "run-task":
            study_dir = experiment_dir / parsed.study
            sys.exit(run_one_task(experiment, study_dir, parsed.task_index))

        # Prepare, run or resume
        else:
            study_dir = _run_study(
                command, experiment, experiment_dir, study_settings, parsed
            )
    except UsageError as error:
        parser.error(str(error))

    # Plot, also straight after a run with --plot or --plot-runs
    plot_after_run = command == "run" and (parsed.plot or parsed.plot_runs is not None)
    if command == "plot" or plot_after_run:
        # Imported here, as in `plot`: tasks never plot, so they start without it
        import matplotlib.pyplot as plt

        runs = _plot_runs(parser, parsed.plot_runs)
        with plt.style.context(hb.plots.STYLE):
            plot(name, problem, measure if true_curves else None, study_dir, runs=runs)


def _parsers(
    name: str, configurations: Mapping[str, hb.Configuration]
) -> dict[str, argparse.ArgumentParser]:
    """The parsers of the commands, by name."""
    parsers = {
        "run": argparse.ArgumentParser(
            prog=f"pixi run experiment {name}",
            description=f"Run the configurations of {name} as a study and save it.",
            epilog=(
                f"Other commands: pixi run experiment {name} prepare | run-task | "
                f"resume | plot. To plot a saved study: pixi run plot {name} [STUDY]"
            ),
        ),
        "prepare": argparse.ArgumentParser(
            prog=f"pixi run experiment {name} prepare",
            description="Reserve a study and write its task list, running nothing.",
        ),
        "run-task": argparse.ArgumentParser(
            prog=f"pixi run experiment {name} run-task",
            description="Run one task of a study; exit status 0 once it is done.",
        ),
        "resume": argparse.ArgumentParser(
            prog=f"pixi run experiment {name} resume",
            description="Run the tasks of a study without a saved run, then save it.",
        ),
        "plot": argparse.ArgumentParser(
            prog=f"pixi run plot {name}",
            description=f"Plot a saved study of {name} into its figures/ folder.",
        ),
    }

    # The study's choices, when it starts
    for command in ("run", "prepare"):
        parser = parsers[command]
        parser.add_argument(
            "--method",
            action="append",
            dest="methods",
            choices=list(configurations),
            metavar="METHOD",
            help=f"a configuration (repeatable): {', '.join(configurations)}; "
            "default every one",
        )
        parser.add_argument("--n-runs", type=int, help="default: the experiment's")
        parser.add_argument("--seed", type=int, help="default: the experiment's")
        parser.add_argument("--name", help="the study id; default: the UTC time")

    # The study, for the commands on one
    parsers["run-task"].add_argument("study", help="the study id")
    parsers["run-task"].add_argument("--task-index", type=int, required=True)
    parsers["resume"].add_argument("study", help="the study id")
    parsers["plot"].add_argument("study", nargs="?", help="default: the newest")

    # Running tasks, plotting, and the flags of every command
    for command in ("run", "prepare", "resume"):
        parsers[command].add_argument(
            "--require-clean",
            action="store_true",
            help="refuse to run with uncommitted or untracked files",
        )
    for command in ("run", "resume"):
        parsers[command].add_argument(
            "--n-workers", type=int, default=1, help="tasks at once (default 1)"
        )
    parsers["run"].add_argument("--plot", action="store_true", help="plot after")
    for command in ("run", "plot"):
        parsers[command].add_argument(
            "--plot-runs",
            nargs="+",
            type=_plot_run,
            metavar="RUN",
            help="the runs with iteration figures: indices or all; default 0",
        )
    for parser in parsers.values():
        parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parsers


def _plot_run(text: str) -> int | str:
    """One `--plot-runs` value: a run index (from 0) or `all`.

    Raises `argparse.ArgumentTypeError` otherwise.
    """
    if text == "all":
        return text
    if not text.isdigit():
        msg = f"not a run index or all: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(text)


def _plot_runs(
    parser: argparse.ArgumentParser, values: list[int | str] | None
) -> list[int] | None:
    """The runs `--plot-runs` chose: its indices, run 0 if it wasn't given, or None
    (every run) for `all`. `all` among indices is a usage error (`parser.error`)."""
    if values is None:
        return [0]
    indices = [value for value in values if isinstance(value, int)]
    if len(indices) == len(values):
        return indices
    if len(values) > 1:
        parser.error("--plot-runs takes run indices or all, not both")
    return None


def _from_init_cwd(path: Path) -> Path:
    """`path`, absolute, relative to the folder pixi was started from (pixi runs the
    experiment from `experiments/` and keeps the starting folder in `INIT_CWD`)."""
    return (Path(os.environ.get("INIT_CWD", ".")) / path).resolve()


def _command(name: str, arguments: str, results_root: Path) -> str:
    """The command line of the experiment with `arguments`, and the results root as
    the user gave it, if it isn't the default."""
    line = f"pixi run experiment {name} {arguments}"
    if results_root != RESULTS_ROOT:
        line += f" --results-root {results_root}"
    return line


# Studies


def _run_study(
    command: str,
    experiment: Experiment,
    experiment_dir: Path,
    study_settings: hb.StudySettings,
    parsed: argparse.Namespace,
) -> Path:
    """Prepares a study (`command` is `run` or `prepare`) or takes the one to resume
    (`resume`) as the command line `parsed` says, then, unless preparing, runs its tasks left and
    assembles it; returns its folder.

    Raises `UsageError` for a tree that isn't clean with `--require-clean`, an
    experiment that isn't a module, a taken name, or a study that can't be resumed.
    Exits with status 130 (Ctrl-C) or 1 if the study doesn't finish.
    """
    # The code version, and the experiment's module, which tasks run as
    version = code_version(Path(hb.__file__).parent)
    if not is_clean(version) and parsed.require_clean:
        msg = f"the code version {version} isn't clean"
        raise UsageError(msg)
    if importlib.util.find_spec(experiment.name) is None:
        msg = f"tasks run `python -m {experiment.name}`, so it must be a module"
        raise UsageError(msg)

    # The task list: the study's to resume, else a new one, as the flags choose
    if command == "resume":
        study_dir = experiment_dir / parsed.study
        task_list = _resumable_task_list(experiment, study_dir, version)
    else:
        n_runs = study_settings.n_runs if parsed.n_runs is None else parsed.n_runs
        # A fresh seed when none is given, as `hb.study` draws one (128 random bits)
        seed = study_settings.seed if parsed.seed is None else parsed.seed
        seed = secrets.randbits(128) if seed is None else seed
        labels = list(dict.fromkeys(parsed.methods or experiment.configurations))
        task_list = task_list_of(
            experiment,
            labels,
            n_runs=n_runs,
            seed=seed,
            version=version,
        )
        default_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        study_dir = reserve(
            experiment_dir, parsed.name or default_id, suffix_if_taken=not parsed.name
        )
        write_task_list(study_dir, task_list)

    # Run the tasks left and assemble the study, logging to its study log
    with logging_to(study_dir / STUDY_LOG, mode="a", console_format=CONSOLE_FORMAT):
        if not is_clean(version):
            _logger.warning(
                "The code version %s isn't clean (uncommitted or untracked files), "
                "so this study won't be reproducible",
                version,
            )
        if command == "prepare":
            _print_tasks(experiment.name, study_dir, task_list, parsed.results_root)
            return study_dir
        if command == "run":
            _log_configurations(task_list)
        resume = _command(
            experiment.name, f"resume {study_dir.name}", parsed.results_root
        )
        try:
            run_tasks(
                experiment,
                study_dir,
                task_list,
                n_workers=parsed.n_workers,
                results_root=_from_init_cwd(parsed.results_root),
            )
            assemble_study(experiment, study_dir, task_list)

        # A study that doesn't finish says how to finish it. Ctrl-C ends it, so
        # another one is ignored (pixi passes each one on, so it comes twice)
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            _logger.error("The study didn't finish. To finish it: %s", resume)
            sys.exit(130)
        except CommandFailedError as error:
            per_run_log = study_dir / PER_RUN_LOGS / f"{error.name}.log"
            _logger.error("%s; its per-run log is %s", error, per_run_log)
            _logger.error("The study didn't finish. To finish it: %s", resume)
            sys.exit(1)
        except Exception:
            _logger.exception("The study didn't finish. To finish it: %s", resume)
            sys.exit(1)
    return study_dir


def task_list_of(
    experiment: Experiment,
    labels: Sequence[str],
    *,
    n_runs: int,
    seed: int,
    version: str,
) -> TaskList:
    """The task list of a study of the experiment's configurations `labels`, `n_runs`
    runs each from `seed`, at the code version `version`."""
    chosen = {label: experiment.configurations[label] for label in labels}
    settings = hb.StudySettings(n_runs=n_runs, seed=seed)
    return TaskList(
        name=experiment.name,
        seed=seed,
        n_runs=n_runs,
        configurations={
            label: {
                "n_evaluations": configuration.n_evaluations,
                "n_initial": configuration.n_initial,
            }
            for label, configuration in chosen.items()
        },
        optimum=None if experiment.optimum is None else float(experiment.optimum),
        code_version=version,
        problem=problem_artifact(experiment.problem, experiment.measure),
        tasks=tuple(
            (task.label, task.index) for task in hb.study_tasks(chosen, settings)
        ),
    )


def write_task_list(study_dir: Path, task_list: TaskList) -> None:
    """Writes the task list to the study's folder, so it appears only when
    complete."""
    text = json.dumps(dataclasses.asdict(task_list), indent=1) + "\n"
    _write_atomically(study_dir / TASK_LIST, lambda path: path.write_text(text))


def read_task_list(study_dir: Path) -> TaskList:
    """The task list in the study's folder.

    Raises `UsageError` if there is none.
    """
    path = study_dir / TASK_LIST
    if not path.exists():
        msg = f"{study_dir} holds no unfinished study (no {TASK_LIST})"
        raise UsageError(msg)
    data = json.loads(path.read_text())
    tasks = tuple((label, index) for label, index in data.pop("tasks"))
    return TaskList(**data, tasks=tasks)


def task_list_differences(
    task_list: TaskList, experiment: Experiment, version: str
) -> list[str]:
    """What differs between the task list and the experiment at the code version
    `version`, as the task list's fields, each with both values if it is one value;
    empty if nothing does."""
    if any(
        label not in experiment.configurations for label in task_list.configurations
    ):
        return ["configurations"]
    now = task_list_of(
        experiment,
        list(task_list.configurations),
        n_runs=task_list.n_runs,
        seed=task_list.seed,
        version=version,
    )
    different = []
    for field in dataclasses.fields(TaskList):
        then_value, now_value = getattr(task_list, field.name), getattr(now, field.name)
        if then_value == now_value:
            continue
        is_one_value = isinstance(now_value, str | int | float | None)
        different.append(
            f"{field.name} ({then_value!r} in the study, {now_value!r} now)"
            if is_one_value
            else field.name
        )
    return different


def _resumable_task_list(
    experiment: Experiment, study_dir: Path, version: str
) -> TaskList:
    """The task list of the study to resume.

    Raises `UsageError` if the study is finished, or its task list differs from the
    experiment.
    """
    if _is_finished(study_dir):
        msg = f"the study {study_dir} is finished, so there is nothing to resume"
        raise UsageError(msg)
    task_list = read_task_list(study_dir)
    _check_task_list(task_list, experiment, version)
    return task_list


def _check_task_list(task_list: TaskList, experiment: Experiment, version: str) -> None:
    """Raises `UsageError`, naming what differs, if the task list differs from the
    experiment at the code version `version`."""
    different = task_list_differences(task_list, experiment, version)
    if different:
        msg = f"the study's task list differs from the experiment in {', '.join(different)}"
        raise UsageError(msg)


def _settings(task_list: TaskList) -> hb.StudySettings:
    """The study's settings, as its task list holds them."""
    return hb.StudySettings(n_runs=task_list.n_runs, seed=task_list.seed)


def _tasks(task_list: TaskList, experiment: Experiment) -> tuple[hb.Task, ...]:
    """The study's tasks, in the order of its task list."""
    chosen = {
        label: experiment.configurations[label] for label in task_list.configurations
    }
    return hb.study_tasks(chosen, _settings(task_list))


def _log_configurations(task_list: TaskList) -> None:
    """Logs at info level each configuration's runs and budget."""
    for label, budget in task_list.configurations.items():
        _logger.info(
            "Configuration %s: %d runs of %d evaluations, %d of them initial points",
            label,
            task_list.n_runs,
            budget["n_evaluations"],
            budget["n_initial"],
        )


def _print_tasks(
    name: str, study_dir: Path, task_list: TaskList, results_root: Path
) -> None:
    """Prints the prepared study's number of tasks and how to run task K."""
    n_tasks = len(task_list.tasks)
    line = _command(name, f"run-task {study_dir.name} --task-index K", results_root)
    print(
        f"Prepared {study_dir} with {n_tasks} tasks. Run task K (0 to {n_tasks - 1}):"
    )
    print(f"  {line}")


# The study folder


def reserve(experiment_dir: Path, study_id: str, *, suffix_if_taken: bool) -> Path:
    """Reserves a study id in `experiment_dir` by creating its folder, and returns
    the folder. Creating it fails if it exists, so studies started at once never
    share an id, and an unfinished study keeps its id.

    A taken id gets `-2`, `-3`, ... appended if `suffix_if_taken`; otherwise raises
    `UsageError`.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate, n = study_id, 2
    while True:
        try:
            (experiment_dir / candidate).mkdir()
        except FileExistsError:
            if not suffix_if_taken:
                msg = f"the study id {candidate} is taken: {experiment_dir / candidate}"
                raise UsageError(msg) from None
            candidate, n = f"{study_id}-{n}", n + 1
        else:
            return experiment_dir / candidate


def _is_finished(study_dir: Path) -> bool:
    """Whether the study in `study_dir` is finished: its artifact is saved."""
    return any((study_dir / name).exists() for name in ARTIFACTS)


def _study_to_plot(experiment_dir: Path, study_id: str | None) -> Path:
    """The finished study `study_id`, else the finished study saved last.

    Raises `UsageError` if there is none, or the study named is unfinished.
    """
    # The newest, by when its artifact was written
    if study_id is None:
        saved = [
            path for name in ARTIFACTS for path in experiment_dir.glob(f"*/{name}")
        ]
        if not saved:
            msg = f"{experiment_dir} holds no study"
            raise UsageError(msg)
        return max(saved, key=lambda path: path.stat().st_mtime).parent

    # The one named, if it is finished
    study_dir = experiment_dir / study_id
    if not _is_finished(study_dir):
        unfinished = (study_dir / TASK_LIST).exists()
        msg = (
            f"the study {study_id} didn't finish: resume it first"
            if unfinished
            else f"{study_dir} holds no study"
        )
        raise UsageError(msg)
    return study_dir


def run_tag(task: hb.Task) -> str:
    """The task's tag, `label/run-003`, as its lines in the study log start and its
    files are named."""
    return f"{task.label}/run-{task.index:03d}"


def _run_file(study_dir: Path, task: hb.Task) -> Path:
    """The task's run file, where `hb.save` puts it."""
    return study_dir / RUN_FILES / f"{run_tag(task)}.json"


def _failure_marker(study_dir: Path, task: hb.Task) -> Path:
    """The file that says the task's run failed before evaluating anything."""
    return _run_file(study_dir, task).with_suffix(".failed")


def _per_run_log(study_dir: Path, task: hb.Task) -> Path:
    """The per-run log of the task's run."""
    return study_dir / PER_RUN_LOGS / f"{run_tag(task)}.log"


def _is_done(study_dir: Path, task: hb.Task) -> bool:
    """Whether the task is done: its run is saved, or it failed before evaluating
    anything."""
    return (
        _run_file(study_dir, task).exists() or _failure_marker(study_dir, task).exists()
    )


def _write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Writes the file `path` with `write`, which writes to the path it is given, so
    the file appears only when complete: to a temporary file beside it, then
    renamed."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    write(temporary)
    temporary.replace(path)


# Tasks


def run_one_task(experiment: Experiment, study_dir: Path, task_index: int) -> int:
    """Runs the study's task `task_index` (`hb.run_task`) unless it is done, and
    saves its run file, or its failure marker, so it appears only when complete.
    Logs to the run's per-run log (DEBUG and above from the library) and to the
    console (INFO and above, tagged with the run), warnings included.

    Returns the exit status: 0 once the task is done, 1 if the run raised an
    exception (logged with its traceback), 130 on Ctrl-C. Raises `UsageError` if the
    study has no task list, the experiment differs from it, or there is no such task.
    """
    # The task, from a task list the experiment matches
    task_list = read_task_list(study_dir)
    _check_task_list(task_list, experiment, code_version(Path(hb.__file__).parent))
    tasks = _tasks(task_list, experiment)
    if not 0 <= task_index < len(tasks):
        msg = f"the study has tasks 0 to {len(tasks) - 1}, not {task_index}"
        raise UsageError(msg)
    task = tasks[task_index]
    if _is_done(study_dir, task):
        return 0

    # The run, logged to its per-run log, saved as it finishes
    per_run_log = _per_run_log(study_dir, task)
    per_run_log.parent.mkdir(parents=True, exist_ok=True)
    _run_file(study_dir, task).parent.mkdir(parents=True, exist_ok=True)
    console_format = f"{run_tag(task)} {CONSOLE_FORMAT}"
    with logging_to(per_run_log, mode="w", console_format=console_format):
        try:
            run = hb.run_task(
                task, experiment.problem, experiment.measure, _settings(task_list)
            )
            if run is None:
                _write_atomically(
                    _failure_marker(study_dir, task),
                    lambda path: path.write_text("failed before evaluating anything\n"),
                )
            else:
                labeled = replace(run, method=task.label)
                _write_atomically(
                    _run_file(study_dir, task),
                    lambda path: hb.save_run(labeled, path),
                )
        except KeyboardInterrupt:
            _logger.exception("Task %d (%s) was interrupted", task_index, run_tag(task))
            return 130
        except Exception:
            _logger.exception("Task %d (%s) failed", task_index, run_tag(task))
            return 1
    return 0


def run_tasks(
    experiment: Experiment,
    study_dir: Path,
    task_list: TaskList,
    *,
    n_workers: int,
    results_root: Path,
) -> None:
    """Runs the study's tasks that aren't done, each as `python -m <experiment>
    run-task`, `n_workers` at a time (`run_commands`), named by their tags, with this
    process's warning options.

    Raises `CommandFailedError`, naming the task, if one fails, and
    `KeyboardInterrupt` on Ctrl-C, after stopping the tasks running.
    """
    commands = {
        run_tag(task): [
            sys.executable,
            *(f"-W{option}" for option in sys.warnoptions),
            "-m",
            experiment.name,
            "run-task",
            study_dir.name,
            "--task-index",
            str(index),
            "--results-root",
            str(results_root),
        ]
        for index, task in enumerate(_tasks(task_list, experiment))
        if not _is_done(study_dir, task)
    }
    run_commands(commands, n_workers=n_workers, cwd=EXPERIMENTS_DIR)


def run_commands(
    commands: Mapping[str, Sequence[str]], *, n_workers: int, cwd: Path
) -> None:
    """Runs the commands, named by their keys, from the folder `cwd`, keeping
    `n_workers` running, in order, and logs each as it starts and finishes.

    Each runs in its own process group, so a Ctrl-C in the terminal reaches only this
    process. With more than one at a time, they run BLAS and OpenMP on one thread
    each, unless their variables are set here.

    Raises `ValueError` if `n_workers` < 1, `CommandFailedError` when a command exits
    with a status other than 0, and `KeyboardInterrupt` on Ctrl-C, after stopping the
    commands running; after Ctrl-C this process ignores Ctrl-C.
    """
    # The commands' environment
    if n_workers < 1:
        msg = f"n_workers must be at least 1, not {n_workers}"
        raise ValueError(msg)
    environment = dict(os.environ)
    if n_workers > 1:
        for variable in THREAD_VARIABLES:
            environment.setdefault(variable, "1")

    # Ctrl-C only marks the commands interrupted, checked once per poll, so it never
    # lands between starting a command and keeping track of it
    interrupts = []
    previous = signal.signal(signal.SIGINT, lambda signum, _: interrupts.append(signum))

    # Keep n_workers running; one that fails stops the rest
    left = list(commands)
    running = {}
    try:
        while left or running:
            if interrupts:
                raise KeyboardInterrupt
            while left and len(running) < n_workers:
                name = left.pop(0)
                running[name] = subprocess.Popen(
                    commands[name],
                    cwd=cwd,
                    env=environment,
                    start_new_session=sys.platform != "win32",
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
                _logger.info("Started %s", name)
            for name, process in list(running.items()):
                status = process.poll()
                if status is None:
                    continue
                del running[name]
                if status != 0:
                    raise CommandFailedError(name, status)
                n_done = len(commands) - len(left) - len(running)
                _logger.info("Finished %s: %d of %d done", name, n_done, len(commands))
            time.sleep(POLL_INTERVAL)

    # On a failure or Ctrl-C, the commands still running are stopped at once, as
    # they may never see the interrupt (CasADi catches it inside IPOPT). Ctrl-C ends
    # the commands, so another one is ignored from then on: pixi passes a Ctrl-C on
    # to the process it runs, which so gets it twice
    finally:
        _stop(list(running.values()))
        signal.signal(signal.SIGINT, signal.SIG_IGN if interrupts else previous)


def _stop(processes: Sequence["subprocess.Popen[bytes]"]) -> None:
    """Terminates the processes and waits for them, killing any still running after
    `STOP_TIMEOUT`. Ctrl-C meanwhile is ignored, so the stop always finishes."""
    if not processes:
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for process in processes:
            process.terminate()
        deadline = time.monotonic() + STOP_TIMEOUT
        for process in processes:
            try:
                process.wait(timeout=max(deadline - time.monotonic(), 0))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


# Assembling the study


def assemble_study(
    experiment: Experiment, study_dir: Path, task_list: TaskList
) -> hb.Study:
    """The study, from its run files in the order of its task list (`hb.assemble`).
    Rebuilds the study log with every per-run log and saves the summary, then the
    artifact, which finishes the study; logs how many runs of each configuration were
    saved."""
    # Each task's run, None for a run that failed before evaluating anything
    tasks = _tasks(task_list, experiment)
    runs = [
        None
        if _failure_marker(study_dir, task).exists()
        else hb.load_run(_run_file(study_dir, task))
        for task in tasks
    ]
    study = hb.assemble(
        tasks,
        runs,
        experiment.problem,
        experiment.measure,
        _settings(task_list),
        name=task_list.name,
        optimum=task_list.optimum,
    )

    # The logs and the summary, then the artifact
    _rebuild_study_log(study_dir, tasks)
    save_summary(study, study_dir)
    save_artifact(study, study_dir)
    for label in task_list.configurations:
        n_saved = sum(run.method == label for run in study.runs)
        _logger.info(
            "%s: saved %d of %d runs to %s", label, n_saved, task_list.n_runs, study_dir
        )
    return study


def _rebuild_study_log(study_dir: Path, tasks: Sequence[hb.Task]) -> None:
    """Rewrites the study log as the launcher's lines, then each per-run log in task
    order, every line starting with its run's tag. Lines tagged by an earlier,
    interrupted assembly are dropped first, so they are never there twice."""
    tags = tuple(f"{run_tag(task)} " for task in tasks)
    study_log = study_dir / STUDY_LOG
    lines = [
        line for line in study_log.read_text().splitlines() if not line.startswith(tags)
    ]
    for task in tasks:
        per_run_log = _per_run_log(study_dir, task)
        if per_run_log.exists():
            tag = run_tag(task)
            lines += [f"{tag} {line}" for line in per_run_log.read_text().splitlines()]
    text = "".join(f"{line}\n" for line in lines)

    # Close the handlers logging to the study log, which reopen it on their next
    # record (in append mode): Windows can't replace an open file, and elsewhere
    # they would go on writing to the replaced one
    for handler in logging.getLogger().handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == study_log.resolve()
        ):
            handler.close()
    _write_atomically(study_log, lambda path: path.write_text(text))


@contextmanager
def logging_to(path: Path, *, mode: str, console_format: str) -> Generator[None]:
    """Logs to the file at `path`, opened with `mode` (DEBUG and above from the
    library, INFO and above from the rest), and to the console (INFO and above, in
    `console_format`), with warnings logged too, until the block ends; then puts
    logging back as it was."""
    # The handlers, on the root logger so the experiment's own records are kept
    file_handler = logging.FileHandler(path, mode=mode)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(console_format))

    # Attach them, saving what to put back: the levels, and whether warnings were
    # logged already (logging's `showwarning` replaces the warnings module's)
    root, library = logging.getLogger(), logging.getLogger("hybrid_bo")
    root_level, library_level = root.level, library.level
    captured = getattr(warnings.showwarning, "__module__", None) == "logging"
    root.addHandler(file_handler)
    root.addHandler(console)
    root.setLevel(logging.INFO)
    library.setLevel(logging.DEBUG)
    logging.captureWarnings(True)

    try:
        yield
    finally:
        logging.captureWarnings(captured)
        root.setLevel(root_level)
        library.setLevel(library_level)
        root.removeHandler(console)
        root.removeHandler(file_handler)
        file_handler.close()


# Plotting


def save_summary(study: hb.Study, study_dir: Path) -> None:
    """Prints `hb.summary(study)` and saves it to `summary.csv` in `study_dir`."""
    # Imported here, so tasks, which never summarize, start without it
    import polars as pl

    table = hb.summary(study)
    table.write_csv(study_dir / SUMMARY)
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(table)


def plot(
    name: str,
    problem: hb.Problem,
    measure: hb.Measure | None,
    study_dir: Path,
    *,
    runs: Sequence[int] | None,
) -> None:
    """Saves the figures of the study in `study_dir` to its `figures/` folder:
    `<name>.png` for each of `FIGURES` (`convergence.png`, and `regret.png` when the
    study has an oracle optimum) and
    `<method>/run-000/iteration-00.png` for every iteration of each method's runs
    with these indices (None: every run), with the oracle values from `measure` if
    given; hybrid BO's are drawn without the GP panels when it has more than one
    unknown state. A figure that can't be drawn is skipped with a message. Saves
    `summary.csv` too if the study has none (a study saved before the CLI wrote
    summaries).

    Raises `ValueError` if the study was run on another experiment.
    """
    # Imported here, so tasks, which never plot, start without it
    import matplotlib.pyplot as plt

    study = hb.load(study_dir)
    if study.name != name:
        msg = f"{study_dir} holds a study of {study.name}, not {name}"
        raise ValueError(msg)
    if not (study_dir / SUMMARY).exists():
        save_summary(study, study_dir)
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # The study figures, each skipped if the study can't give it (regret without
    # an oracle optimum)
    for figure_name, figure in FIGURES.items():
        try:
            figure(study)
        except ValueError as error:
            print(f"no {figure_name} figure: {error}")
        else:
            plt.savefig(figures_dir / f"{figure_name}.png")
        plt.close("all")

    # Every iteration of the runs chosen
    for run in study.runs:
        if runs is not None and run.index not in runs:
            continue
        run_dir = figures_dir / run.method / f"run-{run.index:03d}"
        # Hybrid BO's GP panels draw one unknown state, so with more it goes without
        hybrid_with_several_unknown_states = (
            isinstance(run.settings, hb.HybridBOSettings) and problem.system.n_y > 1
        )
        for i in range(hb.history.n_iterations(run.history)):
            try:
                figure = hb.plots.iteration(
                    run,
                    problem,
                    i,
                    measure=measure,
                    gp=not hybrid_with_several_unknown_states,
                )
            except (ValueError, hb.HybridBOError) as error:
                print(f"{run.method} run {run.index}: no iteration figures: {error}")
                break
            run_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(run_dir / f"iteration-{i:02d}.png")
            plt.close(figure)
    print(f"Saved the figures to {figures_dir}")
