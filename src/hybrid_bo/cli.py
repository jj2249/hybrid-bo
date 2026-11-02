"""The command line of a study script: run its configurations as a study, resume a
study, run one task of a study, or plot a saved study
(`docs/running_large_studies.rst`).

A script (an experiment in `experiments/`, or a study script anywhere) ends with a call
that passes everything it defines:

    if __name__ == "__main__":
        hb.cli.main(problem, measure, configurations, study_settings,
                    optimum=optimum, true_curves=True)

and runs as `python path/to/script.py [command] [flags]`. The script's file name,
without `.py`, names its studies. `run` (the default command) is `prepare` (reserve
the study folder and write its task list), then the launcher (`_run_commands`) running
each task as `python <script> run-task`, then assembling the study from the run files.
A study is finished exactly when its `artifact.json` exists; `resume` runs the tasks
left of one that isn't, then assembles it. The task list records what the script
passes to `main`, so a study resumes from a script that moved, as long as it passes
the same.

Running a study needs Polars (`hybrid-bo[analysis]`), for its summary; plotting needs
matplotlib too (`hybrid-bo[plots]`). Both are imported on first use, so tasks start
without them.
"""

import argparse
import dataclasses
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
from types import ModuleType
from typing import TYPE_CHECKING

from hybrid_bo import plots
from hybrid_bo.errors import HybridBOError
from hybrid_bo.evaluation import Measure
from hybrid_bo.history import n_iterations
from hybrid_bo.problem import Problem
from hybrid_bo.run import code_version, is_clean, problem_artifact
from hybrid_bo.saving import (
    ARTIFACT_FILE,
    OLD_ARTIFACT_FILE,
    load,
    load_run,
    save_artifact,
    save_run,
)
from hybrid_bo.settings import HybridBOSettings
from hybrid_bo.studies import (
    Configuration,
    Study,
    StudySettings,
    Task,
    assemble,
    run_task,
    study_tasks,
)
from hybrid_bo.views import _polars, summary

if TYPE_CHECKING:
    import polars as pl

_RESULTS_ROOT = Path("results")
_ARTIFACTS = (ARTIFACT_FILE, OLD_ARTIFACT_FILE)
_TASK_LIST = "tasks.json"
_STUDY_LOG = "study.log"
_SUMMARY = "summary.csv"
_PER_RUN_LOGS = "logs"
_RUN_FILES = "runs"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
_COMMANDS = ("run", "prepare", "run-task", "resume", "plot")

# The environment variables that set the threads of BLAS and OpenMP, read when they
# load, so a command gets them from the environment it starts in
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# How long stopped commands get to exit before they are killed, and how often the
# launcher checks on them, in seconds
_STOP_TIMEOUT = 5.0
_POLL_INTERVAL = 0.05

# The study figures `plot` draws, by file name
_FIGURES: Mapping[str, Callable[[Study], object]] = {
    "convergence": plots.convergence,
    "regret": plots.regret,
}

_logger = logging.getLogger(__name__)


class _UsageError(Exception):
    """The command line asks for something that can't be done; `main` reports it as
    a usage error (exit status 2)."""


class _CommandFailedError(Exception):
    """A command the launcher ran exited with a status other than 0."""

    def __init__(self, name: str, status: int) -> None:
        """The command `name` exited with `status`."""
        super().__init__(f"{name} failed with exit status {status}")
        self.name = name
        self.status = status


@dataclass(frozen=True, kw_only=True)
class _Experiment:
    """What a study script defines that its studies depend on: the script, whose file
    name (without `.py`) and arguments name its studies, and the problem,
    measurement, configurations, stated oracle optimum and notes it passes to
    `main`."""

    script: Path
    arguments: tuple[str, ...]
    problem: Problem
    measure: Measure
    configurations: Mapping[str, Configuration]
    optimum: float | None
    notes: dict[str, str]

    @property
    def name(self) -> str:
        """The name of the script's studies: its file name without `.py`, then each
        of its arguments, joined by `/` (`s2mpj/HS27`)."""
        return "/".join((self.script.stem, *self.arguments))


@dataclass(frozen=True, kw_only=True)
class _TaskList:
    """What a study runs, saved as `tasks.json` before anything runs: the seed (never
    None), the number of runs, each chosen configuration's budget, the oracle optimum,
    the code version, the problem's artifact text, the tasks as (label, index), in
    order, and the script's notes (none in task lists written before notes)."""

    name: str
    seed: int
    n_runs: int
    configurations: dict[str, dict[str, int]]
    optimum: float | None
    code_version: str
    problem: dict[str, str]
    tasks: tuple[tuple[str, int], ...]
    notes: dict[str, str] = dataclasses.field(default_factory=dict)


def main(
    problem: Problem,
    measure: Measure,
    configurations: Mapping[str, Configuration],
    study_settings: StudySettings,
    *,
    optimum: float | None = None,
    true_curves: bool = False,
    arguments: Sequence[str] = (),
    notes: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Runs, prepares or resumes a study of the script calling it, runs one of its
    tasks, or plots a saved study, as the command line (`argv`, default `sys.argv`)
    says.

    The script is the one Python runs as `__main__`, and its file name without `.py`
    names its studies and their folder; `main` called from no script file (a
    notebook, `python -c`) is a usage error, since tasks run the script again. `study_settings` are the defaults the
    flags override. `optimum` is the stated oracle optimum, which regret is measured
    from. `true_curves` draws the measurement's values in the iteration figures,
    which only an oracle can afford.

    `arguments` are the script's own arguments, which it read from the start of the
    command line to choose what it builds (the S2MPJ benchmark's problem name):
    `argv` defaults to `sys.argv` after them, each task runs as `python <script>
    <arguments> run-task`, and the studies are saved under `<script>/<arguments>/`
    (`results/s2mpj/HS27/`). `notes` are text that describes the experiment beyond
    what it passes here (the S2MPJ commit), recorded in the task list and, as
    `notes`, in the study's artifact.

    A study that doesn't finish exits with status 130 (Ctrl-C) or 1, after saying how
    to resume it. Raises `ImportError`, naming the extra, when running a study without
    Polars or plotting without matplotlib.
    """
    # The experiment, and the command line after the script's own arguments
    experiment = _Experiment(
        script=_script(),
        arguments=tuple(arguments),
        problem=problem,
        measure=measure,
        configurations=configurations,
        optimum=optimum,
        notes=dict(notes or {}),
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if argv is None:
        _check_arguments(experiment.arguments, args)
        args = args[len(experiment.arguments) :]

    # The command: the first argument if it names one, else a run
    command, rest = (
        (args[0], args[1:]) if args[:1] and args[0] in _COMMANDS else ("run", args)
    )
    parser = _parsers(experiment)[command]
    parsed = parser.parse_args(rest)
    experiment_dir = _from_init_cwd(parsed.results_root) / experiment.name

    try:
        # Plot: the study given, else the newest finished one
        if command == "plot":
            study_dir = _study_to_plot(experiment_dir, parsed.study)

        # One task, in this process, as a launcher started it
        elif command == "run-task":
            study_dir = experiment_dir / parsed.study
            sys.exit(_run_one_task(experiment, study_dir, parsed.task_index))

        # Prepare, run or resume
        else:
            study_dir = _run_study(
                command, experiment, experiment_dir, study_settings, parsed
            )
    except _UsageError as error:
        parser.error(str(error))

    # Plot, also straight after a run or a resume with --plot or --plot-runs
    plot_after_run = command in ("run", "resume") and (
        parsed.plot or parsed.plot_runs is not None
    )
    if command == "plot" or plot_after_run:
        plt = _pyplot()
        runs = _plot_runs(parser, parsed.plot_runs)
        with plt.style.context(plots.STYLE):
            _plot(
                experiment.name,
                problem,
                measure if true_curves else None,
                study_dir,
                runs=runs,
            )


def _script() -> Path:
    """The script Python runs as `__main__`, resolved. A usage error (exit status 2)
    if there is none: a notebook or `python -c` runs no file."""
    main_file = getattr(sys.modules["__main__"], "__file__", None)
    if main_file is None:
        msg = (
            "tasks run the script again (python <script> run-task), so hb.cli.main "
            "must be called from a script file, not from a notebook or python -c"
        )
        argparse.ArgumentParser(prog="hb.cli.main").error(msg)
    return Path(main_file).resolve()


def _check_arguments(arguments: tuple[str, ...], command_line: list[str]) -> None:
    """Checks that the script's `arguments` start its command line, and that each
    can name a folder: tasks get them there, and the studies are saved under them.

    Exits with a usage error (status 2) otherwise.
    """
    usage = argparse.ArgumentParser(prog="hb.cli.main").error
    if command_line[: len(arguments)] != list(arguments):
        usage(
            f"the script's arguments {list(arguments)} must start the command line, "
            f"which is {command_line}"
        )
    for argument in arguments:
        is_folder_name = (
            argument not in ("", ".", "..")
            and not argument.startswith("-")
            and not any(separator in argument for separator in "/\\")
        )
        if not is_folder_name:
            usage(f"the script's argument {argument!r} can't name a folder")


def _pyplot() -> ModuleType:
    """`matplotlib.pyplot`, imported here so tasks, which never plot, start without
    it. Raises `ImportError`, naming the extra, when it isn't installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        msg = (
            "plotting a study needs matplotlib: install hybrid-bo[plots] "
            "(or matplotlib)"
        )
        raise ImportError(msg) from error
    return plt


def _parsers(experiment: _Experiment) -> dict[str, argparse.ArgumentParser]:
    """The parsers of the commands of the experiment's script, by name."""
    program = _program(experiment)
    configurations = experiment.configurations
    parsers = {
        "run": argparse.ArgumentParser(
            prog=program,
            description=f"Run the configurations of {experiment.name} as a study and "
            "save it.",
            epilog=(
                f"Other commands: {program} prepare | run-task | resume | plot. "
                f"To plot a saved study: {program} plot [STUDY]"
            ),
        ),
        "prepare": argparse.ArgumentParser(
            prog=f"{program} prepare",
            description="Reserve a study and write its task list, running nothing.",
        ),
        "run-task": argparse.ArgumentParser(
            prog=f"{program} run-task",
            description="Run one task of a study; exit status 0 once it is done.",
        ),
        "resume": argparse.ArgumentParser(
            prog=f"{program} resume",
            description="Run the tasks of a study without a saved run, then save it.",
        ),
        "plot": argparse.ArgumentParser(
            prog=f"{program} plot",
            description=f"Plot a saved study of {experiment.name} into its figures/ "
            "folder.",
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
        parser.add_argument("--n-runs", type=int, help="default: the script's")
        parser.add_argument("--seed", type=int, help="default: the script's")
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
        parsers[command].add_argument("--plot", action="store_true", help="plot after")
    for command in ("run", "resume", "plot"):
        parsers[command].add_argument(
            "--plot-runs",
            nargs="+",
            type=_plot_run,
            metavar="RUN",
            help="the runs with iteration figures: indices or all; default 0",
        )
    for parser in parsers.values():
        parser.add_argument(
            "--results-root",
            type=Path,
            default=_RESULTS_ROOT,
            help="where studies are saved (default: results/ where you start)",
        )
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
    """`path`, absolute, relative to the folder the command was started from: pixi's
    `INIT_CWD` (pixi runs the experiments from `experiments/` and keeps the starting
    folder there), else the working folder."""
    return (Path(os.environ.get("INIT_CWD", ".")) / path).resolve()


def _program(experiment: _Experiment) -> str:
    """The command that runs the experiment's script from the folder the command was
    started from: `python <script> <arguments>`, in pixi's environment if this runs
    in one, with the script's path relative to that folder when it is inside it."""
    script = experiment.script
    start = _from_init_cwd(Path())
    shown = script.relative_to(start) if script.is_relative_to(start) else script
    environment = os.environ.get("PIXI_ENVIRONMENT_NAME")
    python = "python" if environment is None else f"pixi run -e {environment} python"
    return " ".join((python, shown.as_posix(), *experiment.arguments))


def _command(experiment: _Experiment, command: str, results_root: Path) -> str:
    """The command line of the experiment's script with `command` after its own
    arguments, and the results root as the user gave it, if it isn't the default."""
    line = f"{_program(experiment)} {command}"
    if results_root != _RESULTS_ROOT:
        line += f" --results-root {results_root}"
    return line


# Studies


def _run_study(
    command: str,
    experiment: _Experiment,
    experiment_dir: Path,
    study_settings: StudySettings,
    parsed: argparse.Namespace,
) -> Path:
    """Prepares a study (`command` is `run` or `prepare`) or takes the one to resume
    (`resume`) as the command line `parsed` says, then, unless preparing, runs its
    tasks left and assembles it; returns its folder.

    Raises `_UsageError` for a tree that isn't clean with `--require-clean`, a
    taken name, or a study that can't be resumed, and `ImportError` without Polars, which assembling needs. Exits with
    status 130 (Ctrl-C) or 1 if the study doesn't finish.
    """
    # The code version, and Polars, checked before anything runs
    version = code_version(Path(__file__).parent)
    if not is_clean(version) and parsed.require_clean:
        msg = f"the code version {version} isn't clean"
        raise _UsageError(msg)
    _polars()

    # The task list: the study's to resume, else a new one, as the flags choose
    if command == "resume":
        study_dir = experiment_dir / parsed.study
        task_list = _resumable_task_list(experiment, study_dir, version)
    else:
        n_runs = study_settings.n_runs if parsed.n_runs is None else parsed.n_runs
        # A fresh seed when none is given, as `study` draws one (128 random bits)
        seed = study_settings.seed if parsed.seed is None else parsed.seed
        seed = secrets.randbits(128) if seed is None else seed
        labels = list(dict.fromkeys(parsed.methods or experiment.configurations))
        task_list = _task_list_of(
            experiment,
            labels,
            n_runs=n_runs,
            seed=seed,
            version=version,
        )
        default_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        study_dir = _reserve(
            experiment_dir, parsed.name or default_id, suffix_if_taken=not parsed.name
        )
        _write_task_list(study_dir, task_list)

    # Run the tasks left and assemble the study, logging to its study log
    with _logging_to(study_dir / _STUDY_LOG, mode="a", console_format=_CONSOLE_FORMAT):
        if not is_clean(version):
            _logger.warning(
                "The code version %s isn't clean (uncommitted or untracked files), "
                "so this study won't be reproducible",
                version,
            )
        if command == "prepare":
            _print_tasks(experiment, study_dir, task_list, parsed.results_root)
            return study_dir
        if command == "run":
            _log_configurations(task_list)
        resume = _command(experiment, f"resume {study_dir.name}", parsed.results_root)
        try:
            _run_tasks(
                experiment,
                study_dir,
                task_list,
                n_workers=parsed.n_workers,
                results_root=_from_init_cwd(parsed.results_root),
            )
            _assemble_study(experiment, study_dir, task_list)

        # A study that doesn't finish says how to finish it. Ctrl-C ends it, so
        # another one is ignored (pixi passes each one on, so it comes twice)
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            _logger.error("The study didn't finish. To finish it: %s", resume)
            sys.exit(130)
        except _CommandFailedError as error:
            per_run_log = study_dir / _PER_RUN_LOGS / f"{error.name}.log"
            _logger.error("%s; its per-run log is %s", error, per_run_log)
            _logger.error("The study didn't finish. To finish it: %s", resume)
            sys.exit(1)
        except Exception:
            _logger.exception("The study didn't finish. To finish it: %s", resume)
            sys.exit(1)
    return study_dir


def _task_list_of(
    experiment: _Experiment,
    labels: Sequence[str],
    *,
    n_runs: int,
    seed: int,
    version: str,
) -> _TaskList:
    """The task list of a study of the experiment's configurations `labels`, `n_runs`
    runs each from `seed`, at the code version `version`."""
    chosen = {label: experiment.configurations[label] for label in labels}
    settings = StudySettings(n_runs=n_runs, seed=seed)
    return _TaskList(
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
        tasks=tuple((task.label, task.index) for task in study_tasks(chosen, settings)),
        notes=experiment.notes,
    )


def _write_task_list(study_dir: Path, task_list: _TaskList) -> None:
    """Writes the task list to the study's folder, so it appears only when
    complete."""
    text = json.dumps(dataclasses.asdict(task_list), indent=1) + "\n"
    _write_atomically(study_dir / _TASK_LIST, lambda path: path.write_text(text))


def _read_task_list(study_dir: Path) -> _TaskList:
    """The task list in the study's folder.

    Raises `_UsageError` if there is none.
    """
    path = study_dir / _TASK_LIST
    if not path.exists():
        msg = f"{study_dir} holds no unfinished study (no {_TASK_LIST})"
        raise _UsageError(msg)
    data = json.loads(path.read_text())
    tasks = tuple((label, index) for label, index in data.pop("tasks"))
    return _TaskList(**data, tasks=tasks)


def _task_list_differences(
    task_list: _TaskList, experiment: _Experiment, version: str
) -> list[str]:
    """What differs between the task list and the experiment at the code version
    `version`, as the task list's fields, each with both values if it is one value;
    empty if nothing does."""
    if any(
        label not in experiment.configurations for label in task_list.configurations
    ):
        return ["configurations"]
    now = _task_list_of(
        experiment,
        list(task_list.configurations),
        n_runs=task_list.n_runs,
        seed=task_list.seed,
        version=version,
    )
    different = []
    for field in dataclasses.fields(_TaskList):
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
    experiment: _Experiment, study_dir: Path, version: str
) -> _TaskList:
    """The task list of the study to resume.

    Raises `_UsageError` if the study is finished, or its task list differs from the
    experiment.
    """
    if _is_finished(study_dir):
        msg = f"the study {study_dir} is finished, so there is nothing to resume"
        raise _UsageError(msg)
    task_list = _read_task_list(study_dir)
    _check_task_list(task_list, experiment, version)
    return task_list


def _check_task_list(
    task_list: _TaskList, experiment: _Experiment, version: str
) -> None:
    """Raises `_UsageError`, naming what differs, if the task list differs from the
    experiment at the code version `version`."""
    different = _task_list_differences(task_list, experiment, version)
    if different:
        msg = f"the study's task list differs from the experiment in {', '.join(different)}"
        raise _UsageError(msg)


def _settings(task_list: _TaskList) -> StudySettings:
    """The study's settings, as its task list holds them."""
    return StudySettings(n_runs=task_list.n_runs, seed=task_list.seed)


def _tasks(task_list: _TaskList, experiment: _Experiment) -> tuple[Task, ...]:
    """The study's tasks, in the order of its task list."""
    chosen = {
        label: experiment.configurations[label] for label in task_list.configurations
    }
    return study_tasks(chosen, _settings(task_list))


def _log_configurations(task_list: _TaskList) -> None:
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
    experiment: _Experiment, study_dir: Path, task_list: _TaskList, results_root: Path
) -> None:
    """Prints the prepared study's number of tasks and how to run task K."""
    n_tasks = len(task_list.tasks)
    line = _command(
        experiment, f"run-task {study_dir.name} --task-index K", results_root
    )
    print(
        f"Prepared {study_dir} with {n_tasks} tasks. Run task K (0 to {n_tasks - 1}):"
    )
    print(f"  {line}")


# The study folder


def _reserve(experiment_dir: Path, study_id: str, *, suffix_if_taken: bool) -> Path:
    """Reserves a study id in `experiment_dir` by creating its folder, and returns
    the folder. Creating it fails if it exists, so studies started at once never
    share an id, and an unfinished study keeps its id.

    A taken id gets `-2`, `-3`, ... appended if `suffix_if_taken`; otherwise raises
    `_UsageError`.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate, n = study_id, 2
    while True:
        try:
            (experiment_dir / candidate).mkdir()
        except FileExistsError:
            if not suffix_if_taken:
                msg = f"the study id {candidate} is taken: {experiment_dir / candidate}"
                raise _UsageError(msg) from None
            candidate, n = f"{study_id}-{n}", n + 1
        else:
            return experiment_dir / candidate


def _is_finished(study_dir: Path) -> bool:
    """Whether the study in `study_dir` is finished: its artifact is saved."""
    return any((study_dir / name).exists() for name in _ARTIFACTS)


def _study_to_plot(experiment_dir: Path, study_id: str | None) -> Path:
    """The finished study `study_id`, else the finished study saved last.

    Raises `_UsageError` if there is none, or the study named is unfinished.
    """
    # The newest, by when its artifact was written
    if study_id is None:
        saved = [
            path for name in _ARTIFACTS for path in experiment_dir.glob(f"*/{name}")
        ]
        if not saved:
            msg = f"{experiment_dir} holds no study"
            raise _UsageError(msg)
        return max(saved, key=lambda path: path.stat().st_mtime).parent

    # The one named, if it is finished
    study_dir = experiment_dir / study_id
    if not _is_finished(study_dir):
        unfinished = (study_dir / _TASK_LIST).exists()
        msg = (
            f"the study {study_id} didn't finish: resume it first"
            if unfinished
            else f"{study_dir} holds no study"
        )
        raise _UsageError(msg)
    return study_dir


def _run_tag(task: Task) -> str:
    """The task's tag, `label/run-003`, as its lines in the study log start and its
    files are named."""
    return f"{task.label}/run-{task.index:03d}"


def _run_file(study_dir: Path, task: Task) -> Path:
    """The task's run file, where `save` puts it."""
    return study_dir / _RUN_FILES / f"{_run_tag(task)}.json"


def _failure_marker(study_dir: Path, task: Task) -> Path:
    """The file that says the task's run failed before evaluating anything."""
    return _run_file(study_dir, task).with_suffix(".failed")


def _per_run_log(study_dir: Path, task: Task) -> Path:
    """The per-run log of the task's run."""
    return study_dir / _PER_RUN_LOGS / f"{_run_tag(task)}.log"


def _is_done(study_dir: Path, task: Task) -> bool:
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


def _run_one_task(experiment: _Experiment, study_dir: Path, task_index: int) -> int:
    """Runs the study's task `task_index` (`run_task`) unless it is done, and
    saves its run file, or its failure marker, so it appears only when complete.
    Logs to the run's per-run log (DEBUG and above from the library) and to the
    console (INFO and above, tagged with the run), warnings included.

    Returns the exit status: 0 once the task is done, 1 if the run raised an
    exception (logged with its traceback), 130 on Ctrl-C. Raises `_UsageError` if the
    study has no task list, the experiment differs from it, or there is no such task.
    """
    # The task, from a task list the experiment matches
    task_list = _read_task_list(study_dir)
    _check_task_list(task_list, experiment, code_version(Path(__file__).parent))
    tasks = _tasks(task_list, experiment)
    if not 0 <= task_index < len(tasks):
        msg = f"the study has tasks 0 to {len(tasks) - 1}, not {task_index}"
        raise _UsageError(msg)
    task = tasks[task_index]
    if _is_done(study_dir, task):
        return 0

    # The run, logged to its per-run log, saved as it finishes
    per_run_log = _per_run_log(study_dir, task)
    per_run_log.parent.mkdir(parents=True, exist_ok=True)
    _run_file(study_dir, task).parent.mkdir(parents=True, exist_ok=True)
    console_format = f"{_run_tag(task)} {_CONSOLE_FORMAT}"
    with _logging_to(per_run_log, mode="w", console_format=console_format):
        try:
            run = run_task(
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
                    lambda path: save_run(labeled, path),
                )
        except KeyboardInterrupt:
            _logger.exception(
                "Task %d (%s) was interrupted", task_index, _run_tag(task)
            )
            return 130
        except Exception:
            _logger.exception("Task %d (%s) failed", task_index, _run_tag(task))
            return 1
    return 0


def _run_tasks(
    experiment: _Experiment,
    study_dir: Path,
    task_list: _TaskList,
    *,
    n_workers: int,
    results_root: Path,
) -> None:
    """Runs the study's tasks that aren't done, each as `python <script>
    <arguments> run-task` from this process's folder, `n_workers` at a time
    (`_run_commands`), named by their tags, with this process's warning options.

    Raises `_CommandFailedError`, naming the task, if one fails, and
    `KeyboardInterrupt` on Ctrl-C, after stopping the tasks running.
    """
    commands = {
        _run_tag(task): [
            sys.executable,
            *(f"-W{option}" for option in sys.warnoptions),
            str(experiment.script),
            *experiment.arguments,
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
    _run_commands(commands, n_workers=n_workers)


def _run_commands(commands: Mapping[str, Sequence[str]], *, n_workers: int) -> None:
    """Runs the commands, named by their keys, from this process's folder, keeping
    `n_workers` running, in order, and logs each as it starts and finishes.

    Each runs in its own process group, so a Ctrl-C in the terminal reaches only this
    process. With more than one at a time, they run BLAS and OpenMP on one thread
    each, unless their variables are set here.

    Raises `ValueError` if `n_workers` < 1, `_CommandFailedError` when a command exits
    with a status other than 0, and `KeyboardInterrupt` on Ctrl-C, after stopping the
    commands running; after Ctrl-C this process ignores Ctrl-C.
    """
    # The commands' environment
    if n_workers < 1:
        msg = f"n_workers must be at least 1, not {n_workers}"
        raise ValueError(msg)
    environment = dict(os.environ)
    if n_workers > 1:
        for variable in _THREAD_VARIABLES:
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
                    raise _CommandFailedError(name, status)
                n_done = len(commands) - len(left) - len(running)
                _logger.info("Finished %s: %d of %d done", name, n_done, len(commands))
            time.sleep(_POLL_INTERVAL)

    # On a failure or Ctrl-C, the commands still running are stopped at once, as
    # they may never see the interrupt (CasADi catches it inside IPOPT). Ctrl-C ends
    # the commands, so another one is ignored from then on: pixi passes a Ctrl-C on
    # to the process it runs, which so gets it twice
    finally:
        _stop(list(running.values()))
        signal.signal(signal.SIGINT, signal.SIG_IGN if interrupts else previous)


def _stop(processes: Sequence["subprocess.Popen[bytes]"]) -> None:
    """Terminates the processes and waits for them, killing any still running after
    `_STOP_TIMEOUT`. Ctrl-C meanwhile is ignored, so the stop always finishes."""
    if not processes:
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for process in processes:
            process.terminate()
        deadline = time.monotonic() + _STOP_TIMEOUT
        for process in processes:
            try:
                process.wait(timeout=max(deadline - time.monotonic(), 0))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


# Assembling the study


def _assemble_study(
    experiment: _Experiment, study_dir: Path, task_list: _TaskList
) -> Study:
    """The study, from its run files in the order of its task list (`assemble`),
    with the task list's notes in its artifact. Rebuilds the study log with every
    per-run log and saves the summary, then the artifact, which finishes the study;
    then prints the summary and logs how many runs of each configuration were
    saved."""
    # Each task's run, None for a run that failed before evaluating anything
    tasks = _tasks(task_list, experiment)
    runs = [
        None
        if _failure_marker(study_dir, task).exists()
        else load_run(_run_file(study_dir, task))
        for task in tasks
    ]
    study = assemble(
        tasks,
        runs,
        experiment.problem,
        experiment.measure,
        _settings(task_list),
        name=task_list.name,
        optimum=task_list.optimum,
    )
    if task_list.notes:
        notes = "\n".join(f"{key}: {value}" for key, value in task_list.notes.items())
        study = replace(study, artifact=study.artifact | {"notes": notes})

    # The logs and the summary, then the artifact, and only then what is printed,
    # which so can't stop the study from finishing
    _rebuild_study_log(study_dir, tasks)
    table = _save_summary(study, study_dir)
    save_artifact(study, study_dir)
    _print_summary(table)
    for label in task_list.configurations:
        n_saved = sum(run.method == label for run in study.runs)
        _logger.info(
            "%s: saved %d of %d runs to %s", label, n_saved, task_list.n_runs, study_dir
        )
    return study


def _rebuild_study_log(study_dir: Path, tasks: Sequence[Task]) -> None:
    """Rewrites the study log as the launcher's lines, then each per-run log in task
    order, every line starting with its run's tag. Lines tagged by an earlier,
    interrupted assembly are dropped first, so they are never there twice."""
    tags = tuple(f"{_run_tag(task)} " for task in tasks)
    study_log = study_dir / _STUDY_LOG
    lines = [
        line for line in study_log.read_text().splitlines() if not line.startswith(tags)
    ]
    for task in tasks:
        per_run_log = _per_run_log(study_dir, task)
        if per_run_log.exists():
            tag = _run_tag(task)
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
def _logging_to(path: Path, *, mode: str, console_format: str) -> Generator[None]:
    """Logs to the file at `path`, opened with `mode` (DEBUG and above from the
    library, INFO and above from the rest), and to the console (INFO and above, in
    `console_format`), with warnings logged too, until the block ends; then puts
    logging back as it was."""
    # The handlers, on the root logger so the experiment's own records are kept
    file_handler = logging.FileHandler(path, mode=mode)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
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


def _save_summary(study: Study, study_dir: Path) -> "pl.DataFrame":
    """Saves `summary(study)` to `summary.csv` in `study_dir`, and returns it."""
    table = summary(study)
    table.write_csv(study_dir / _SUMMARY)
    return table


def _print_summary(table: "pl.DataFrame") -> None:
    """Prints the summary table with ASCII borders, which any console encoding can
    print (a Windows pipe or file is cp1252, which has no box-drawing characters)."""
    with _polars().Config(
        tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_FULL_CONDENSED"
    ):
        print(table)


def _plot(
    name: str,
    problem: Problem,
    measure: Measure | None,
    study_dir: Path,
    *,
    runs: Sequence[int] | None,
) -> None:
    """Saves the figures of the study `name` in `study_dir` to its `figures/` folder:
    `<name>.png` for each of `_FIGURES` (`convergence.png`, and `regret.png` when the
    study has an oracle optimum) and
    `<method>/run-000/iteration-00.png` for every iteration of each method's runs
    with these indices (None: every run), with the oracle values from `measure` if
    given; hybrid BO's are drawn without the GP panels when it has more than one
    unknown state, or `h_inputs` other than u. A figure that can't be drawn is
    skipped with a message. Saves `summary.csv` too if the study has none (a study
    saved before the CLI wrote summaries).

    Raises `ValueError` if the study was run from another script.
    """
    plt = _pyplot()
    study = load(study_dir)
    if study.name != name:
        msg = f"{study_dir} holds a study of {study.name}, not {name}"
        raise ValueError(msg)
    if not (study_dir / _SUMMARY).exists():
        _print_summary(_save_summary(study, study_dir))
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # The study figures, each skipped if the study can't give it (regret without
    # an oracle optimum)
    for figure_name, figure in _FIGURES.items():
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
        # Hybrid BO's GP panels draw one unknown state over u, so with more, or with
        # other h_inputs (as the S2MPJ benchmarks have), it goes without
        system = problem.system
        hybrid_without_gp_panels = isinstance(run.settings, HybridBOSettings) and (
            system.n_y > 1 or list(system.h_inputs) != list(system.u)
        )
        for i in range(n_iterations(run.history)):
            try:
                figure = plots.iteration(
                    run,
                    problem,
                    i,
                    measure=measure,
                    gp=not hybrid_without_gp_panels,
                )
            except (ValueError, HybridBOError) as error:
                print(f"{run.method} run {run.index}: no iteration figures: {error}")
                break
            run_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(run_dir / f"iteration-{i:02d}.png")
            plt.close(figure)
    print(f"Saved the figures to {figures_dir}")
