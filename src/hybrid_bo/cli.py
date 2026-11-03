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
the same. What the study folder holds, and when a task is done, is
`hybrid_bo.study_folder`'s; this module parses the command line, launches the tasks
and logs.

Running a study needs Polars (`hybrid-bo[analysis]`), for its summary; plotting needs
matplotlib too (`hybrid-bo[plots]`). Both are imported on first use, so tasks start
without them.
"""

import argparse
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
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from hybrid_bo import plots
from hybrid_bo.errors import HybridBOError, StudyFolderError
from hybrid_bo.evaluation import Measure
from hybrid_bo.history import n_iterations
from hybrid_bo.problem import Problem
from hybrid_bo.run import code_version, is_clean
from hybrid_bo.saving import load
from hybrid_bo.settings import HybridBOSettings
from hybrid_bo.studies import Configuration, Study, StudySettings, run_task
from hybrid_bo.study_folder import (
    Experiment,
    StudyFolder,
    assemble_study,
    finished_study,
    is_done,
    is_finished,
    open_study,
    per_run_log,
    prepare,
    run_tag,
    save_result,
    study_log,
    study_settings,
    task,
    tasks,
    tasks_left,
)
from hybrid_bo.views import _polars, summary

if TYPE_CHECKING:
    import polars as pl

_RESULTS_ROOT = Path("results")
_SUMMARY = "summary.csv"
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


def main(
    problem: Problem,
    measure: Measure,
    configurations: Mapping[str, Configuration],
    study_settings: StudySettings,
    *,
    optimum: float | None = None,
    true_curves: bool = False,
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

    A study that doesn't finish exits with status 130 (Ctrl-C) or 1, after saying how
    to resume it. Raises `ImportError`, naming the extra, when running a study without
    Polars or plotting without matplotlib.
    """
    # The command: the first argument if it names one, else a run
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = (
        (args[0], args[1:]) if args[:1] and args[0] in _COMMANDS else ("run", args)
    )
    script = _script()
    parser = _parsers(script, configurations)[command]
    parsed = parser.parse_args(rest)
    experiment = Experiment(
        script=script,
        problem=problem,
        measure=measure,
        configurations=configurations,
        optimum=optimum,
    )
    experiment_dir = _from_init_cwd(parsed.results_root) / experiment.name

    try:
        # Plot: the study given, else the newest finished one
        if command == "plot":
            study_dir = finished_study(experiment_dir, parsed.study)

        # One task, in this process, as a launcher started it
        elif command == "run-task":
            folder = open_study(
                experiment_dir / parsed.study, experiment, _code_version()
            )
            sys.exit(_run_one_task(experiment, folder, parsed.task_index))

        # Prepare, run or resume: the study folder, then its tasks left, logged to
        # its study log
        else:
            version = _code_version()
            if not is_clean(version) and parsed.require_clean:
                msg = f"the code version {version} isn't clean"
                raise _UsageError(msg)
            _polars()
            folder = (
                _resumable(experiment_dir / parsed.study, experiment, version)
                if command == "resume"
                else _prepare(
                    experiment_dir, experiment, study_settings, parsed, version
                )
            )
            with _logging_to(
                study_log(folder), mode="a", console_format=_CONSOLE_FORMAT
            ):
                if not is_clean(version):
                    _logger.warning(
                        "The code version %s isn't clean (uncommitted or untracked "
                        "files), so this study won't be reproducible",
                        version,
                    )
                if command == "prepare":
                    _print_tasks(experiment.script, folder, parsed.results_root)
                else:
                    _finish(
                        experiment, folder, parsed, log_configurations=command == "run"
                    )
            study_dir = folder.path
    except (_UsageError, StudyFolderError) as error:
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


def _parsers(
    script: Path, configurations: Mapping[str, Configuration]
) -> dict[str, argparse.ArgumentParser]:
    """The parsers of the commands of the script, by name."""
    program = _program(script)
    parsers = {
        "run": argparse.ArgumentParser(
            prog=program,
            description=f"Run the configurations of {script.stem} as a study and "
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
            description=f"Plot a saved study of {script.stem} into its figures/ "
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


def _program(script: Path) -> str:
    """The command that runs the script from the folder the command was started from:
    `python <script>`, in pixi's environment if this runs in one, with the script's
    path relative to that folder when it is inside it."""
    start = _from_init_cwd(Path())
    shown = script.relative_to(start) if script.is_relative_to(start) else script
    environment = os.environ.get("PIXI_ENVIRONMENT_NAME")
    python = "python" if environment is None else f"pixi run -e {environment} python"
    return f"{python} {shown.as_posix()}"


def _command(script: Path, arguments: str, results_root: Path) -> str:
    """The command line of the script with `arguments`, and the results root as the
    user gave it, if it isn't the default."""
    line = f"{_program(script)} {arguments}"
    if results_root != _RESULTS_ROOT:
        line += f" --results-root {results_root}"
    return line


# Studies


def _code_version() -> str:
    """The version of the library's code (`code_version`)."""
    return code_version(Path(__file__).parent)


def _prepare(
    experiment_dir: Path,
    experiment: Experiment,
    study_settings: StudySettings,
    parsed: argparse.Namespace,
    version: str,
) -> StudyFolder:
    """Prepares a study of the experiment as the command line `parsed` says, its
    flags overriding `study_settings`, at the code version `version`.

    Raises `StudyFolderError` for a taken `--name`.
    """
    n_runs = study_settings.n_runs if parsed.n_runs is None else parsed.n_runs
    # A fresh seed when none is given, as `study` draws one (128 random bits)
    seed = study_settings.seed if parsed.seed is None else parsed.seed
    seed = secrets.randbits(128) if seed is None else seed
    labels = list(dict.fromkeys(parsed.methods or experiment.configurations))
    default_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return prepare(
        experiment_dir,
        experiment,
        labels,
        n_runs=n_runs,
        seed=seed,
        version=version,
        study_id=parsed.name or default_id,
        suffix_if_taken=not parsed.name,
    )


def _resumable(study_dir: Path, experiment: Experiment, version: str) -> StudyFolder:
    """The unfinished study to resume.

    Raises `_UsageError` if the study is finished, and `StudyFolderError` if it has
    no task list or its task list differs from the experiment.
    """
    if is_finished(study_dir):
        msg = f"the study {study_dir} is finished, so there is nothing to resume"
        raise _UsageError(msg)
    return open_study(study_dir, experiment, version)


def _finish(
    experiment: Experiment,
    folder: StudyFolder,
    parsed: argparse.Namespace,
    *,
    log_configurations: bool,
) -> None:
    """Runs the study's tasks left and assembles it, first logging each
    configuration's runs and budget if `log_configurations`.

    Exits with status 130 (Ctrl-C) or 1 if the study doesn't finish, after saying how
    to finish it.
    """
    if log_configurations:
        _log_configurations(folder)
    resume = _command(
        experiment.script, f"resume {folder.path.name}", parsed.results_root
    )
    try:
        _run_tasks(
            experiment,
            folder,
            n_workers=parsed.n_workers,
            results_root=_from_init_cwd(parsed.results_root),
        )
        _assemble(experiment, folder)

    # A study that doesn't finish says how to finish it. Ctrl-C ends it, so another
    # one is ignored (pixi passes each one on, so it comes twice)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _logger.error("The study didn't finish. To finish it: %s", resume)
        sys.exit(130)
    except _CommandFailedError as error:
        by_tag = {run_tag(each): each for each in tasks(folder, experiment)}
        log = per_run_log(folder, by_tag[error.name])
        _logger.error("%s; its per-run log is %s", error, log)
        _logger.error("The study didn't finish. To finish it: %s", resume)
        sys.exit(1)
    except Exception:
        _logger.exception("The study didn't finish. To finish it: %s", resume)
        sys.exit(1)


def _log_configurations(folder: StudyFolder) -> None:
    """Logs at info level each configuration's runs and budget."""
    task_list = folder.task_list
    for label, budget in task_list.configurations.items():
        _logger.info(
            "Configuration %s: %d runs of %d evaluations, %d of them initial points",
            label,
            task_list.n_runs,
            budget["n_evaluations"],
            budget["n_initial"],
        )


def _print_tasks(script: Path, folder: StudyFolder, results_root: Path) -> None:
    """Prints the prepared study's number of tasks and how to run task K."""
    n_tasks = len(folder.task_list.tasks)
    line = _command(script, f"run-task {folder.path.name} --task-index K", results_root)
    print(
        f"Prepared {folder.path} with {n_tasks} tasks. Run task K (0 to {n_tasks - 1}):"
    )
    print(f"  {line}")


# Tasks


def _run_one_task(experiment: Experiment, folder: StudyFolder, task_index: int) -> int:
    """Runs the study's task `task_index` (`run_task`) unless it is done, and saves
    its result (`save_result`). Logs to the run's per-run log (DEBUG and above from
    the library) and to the console (INFO and above, tagged with the run), warnings
    included.

    Returns the exit status: 0 once the task is done, 1 if the run raised an
    exception (logged with its traceback), 130 on Ctrl-C. Raises `StudyFolderError`
    if there is no such task.
    """
    # The task, unless it is done
    this_task = task(folder, experiment, task_index)
    if is_done(folder, this_task):
        return 0

    # The run, logged to its per-run log, saved as it finishes
    log = per_run_log(folder, this_task)
    log.parent.mkdir(parents=True, exist_ok=True)
    tag = run_tag(this_task)
    with _logging_to(log, mode="w", console_format=f"{tag} {_CONSOLE_FORMAT}"):
        try:
            run = run_task(
                this_task,
                experiment.problem,
                experiment.measure,
                study_settings(folder),
            )
            save_result(folder, this_task, run)
        except KeyboardInterrupt:
            _logger.exception("Task %d (%s) was interrupted", task_index, tag)
            return 130
        except Exception:
            _logger.exception("Task %d (%s) failed", task_index, tag)
            return 1
    return 0


def _run_tasks(
    experiment: Experiment,
    folder: StudyFolder,
    *,
    n_workers: int,
    results_root: Path,
) -> None:
    """Runs the study's tasks that aren't done, each as `python <script> run-task`
    from this process's folder, `n_workers` at a time (`_run_commands`), named by
    their tags, with this process's warning options.

    Raises `_CommandFailedError`, naming the task, if one fails, and
    `KeyboardInterrupt` on Ctrl-C, after stopping the tasks running.
    """
    commands = {
        run_tag(each): [
            sys.executable,
            *(f"-W{option}" for option in sys.warnoptions),
            str(experiment.script),
            "run-task",
            folder.path.name,
            "--task-index",
            str(index),
            "--results-root",
            str(results_root),
        ]
        for index, each in tasks_left(folder, experiment)
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


def _assemble(experiment: Experiment, folder: StudyFolder) -> Study:
    """The study, assembled from its run files (`assemble_study`), which finishes it;
    then saves its summary, prints it and logs how many runs of each configuration
    were saved."""
    # The study log is replaced, so the handlers logging to it are closed first;
    # they reopen it on their next record (in append mode). Windows can't replace an
    # open file, and elsewhere they would go on writing to the replaced one
    for handler in logging.getLogger().handlers:
        is_on_study_log = (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == study_log(folder).resolve()
        )
        if is_on_study_log:
            handler.close()
    study = assemble_study(folder, experiment)

    # The summary, and what is printed, after the artifact, so neither can stop the
    # study from finishing (`plot` saves a summary that is missing)
    _print_summary(_save_summary(study, folder.path))
    task_list = folder.task_list
    for label in task_list.configurations:
        n_saved = sum(run.method == label for run in study.runs)
        _logger.info(
            "%s: saved %d of %d runs to %s",
            label,
            n_saved,
            task_list.n_runs,
            folder.path,
        )
    return study


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
    unknown state. A figure that can't be drawn is skipped with a message. Saves
    `summary.csv` too if the study has none (a study saved before the CLI wrote
    summaries).

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
        # Hybrid BO's GP panels draw one unknown state, so with more it goes without
        hybrid_with_several_unknown_states = (
            isinstance(run.settings, HybridBOSettings) and problem.system.n_y > 1
        )
        for i in range(n_iterations(run.history)):
            try:
                figure = plots.iteration(
                    run,
                    problem,
                    i,
                    measure=measure,
                    gp=not hybrid_with_several_unknown_states,
                )
            except (ValueError, HybridBOError) as error:
                print(f"{run.method} run {run.index}: no iteration figures: {error}")
                break
            run_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(run_dir / f"iteration-{i:02d}.png")
            plt.close(figure)
    print(f"Saved the figures to {figures_dir}")
