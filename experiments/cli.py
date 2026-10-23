"""The command line of an experiment: run its methods as a study, run one task of a
study, resume a study, or plot a saved study.

An experiment ends with a call that passes everything it defines:

    if __name__ == "__main__":
        cli.main("illustrative_example", problem, measure, configurations,
                 study_settings, optimum=optimum, true_curves=True)

and the pixi tasks run the script as a module from this folder:

    pixi run experiment illustrative_example [--method hybrid_bo ...] [--n-runs 3]
        [--seed 0] [--n-workers 4] [--name lcb-vs-ei] [--plot] [--plot-runs 0 3]
        [--require-clean]
    pixi run experiment illustrative_example prepare [--method ...] [--n-runs 3] ...
    pixi run experiment illustrative_example run-task STUDY --task-index K
    pixi run experiment illustrative_example resume STUDY [--n-workers 4]
    pixi run plot illustrative_example [STUDY]
    pixi run experiment illustrative_example plot [STUDY] [--plot-runs all]

A study's runs are its tasks (`hb.study_tasks`), each run by `run-task` in a process
of its own, which saves the run's file and per-run log as soon as the run finishes.
Running (`run`, the default command) is `prepare`, then the tasks, then assembling:

- `prepare` reserves the study id by creating the study folder,
  `<results root>/<experiment>/<study id>/` (the results root defaults to `results/`
  at the repo root; the study id is `--name` or the UTC time at the start), and
  writes its task list, `tasks.json`: the seed (a fresh one if None), the number of
  runs, each chosen configuration's budget, the oracle optimum, the code version, the
  problem and measurement, and the tasks in order. A taken `--name` is refused, and a
  taken UTC time gets `-2`, `-3`, ... appended.
- The launcher keeps `--n-workers` tasks running (default 1), each as
  `python -m <experiment> run-task`, in task order; with more than one, the tasks run
  BLAS and OpenMP on one thread each unless their variables are set. A task saves
  `runs/<label>/run-003.json`, or the failure marker `runs/<label>/run-003.failed`
  for a run that failed before evaluating anything, and its per-run log
  `logs/<label>/run-003.log` (DEBUG and above from the library); the console shows
  its INFO lines tagged `<label>/run-003`.
- Assembling loads the run files, saves the study's `artifact.json` beside them (a
  study is finished exactly when it exists), appends each per-run log to the study log
  `study.log`, every line tagged with its run, and prints `hb.summary` of the study and
  saves it as `summary.csv`.

Ctrl-C, or a task that fails, stops the study at once and keeps its finished runs;
`resume STUDY` runs the tasks left and assembles the study. It refuses a finished
study, and an experiment, flags or code version that differ from the task list. The
study log records the launcher's progress and the warnings (the console shows INFO and
above). Running from a tree that isn't clean (uncommitted or untracked files) warns
that the study won't be reproducible; `--require-clean` refuses to run instead. `plot`
loads a study (default: the newest finished) with `hb.load` and saves its figures to
its `figures/` folder: the study figures the experiment passes as `figures` (default:
convergence and regret), and the iteration figures of the runs `--plot-runs` names
(indices, or `all`; default run 0 of each method). On `run` and `resume`,
`--plot-runs` implies `--plot`.
"""

import argparse
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import time
import warnings
from collections import deque
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import hybrid_bo as hb
from hybrid_bo.run import code_version, is_clean, problem_artifact
from hybrid_bo.saving import ARTIFACT_FILE, OLD_ARTIFACT_FILE, save_artifact

RESULTS_ROOT = Path(__file__).parent.parent / "results"
ARTIFACTS = (ARTIFACT_FILE, OLD_ARTIFACT_FILE)
TASK_LIST = "tasks.json"
STUDY_LOG = "study.log"
PER_RUN_LOGS = "logs"
RUN_FILES = "runs"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
COMMANDS = ("run", "prepare", "run-task", "resume", "plot")
NOT_FINISHED = "The study didn't finish; its finished runs are kept. To finish it: %s"

# The environment variables that set the threads of BLAS and OpenMP, read when they
# load, so a task gets them from the environment it starts in
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# How long stopped tasks get to exit before they are killed, and how often the
# launcher checks on its tasks, in seconds
STOP_TIMEOUT = 5.0
POLL_INTERVAL = 0.05

# The study figures `plot` draws unless the experiment passes its own, by file name
FIGURES: Mapping[str, Callable[[hb.Study], object]] = {
    "convergence": hb.plots.convergence,
    "regret": hb.plots.regret,
}

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    """What an experiment defines that its studies depend on: its name, problem,
    measurement, configurations and stated oracle optimum."""

    name: str
    problem: hb.Problem
    measure: hb.Measure
    configurations: Mapping[str, hb.Configuration]
    optimum: float | None


class TaskFailedError(Exception):
    """A task of a study exited without finishing its run, which stops the study."""


def main(
    name: str,
    problem: hb.Problem,
    measure: hb.Measure,
    configurations: Mapping[str, hb.Configuration],
    study_settings: hb.StudySettings,
    *,
    optimum: float | None = None,
    true_curves: bool = False,
    figures: Mapping[str, Callable[[hb.Study], object]] = FIGURES,
    argv: Sequence[str] | None = None,
) -> None:
    """Runs the experiment's configurations as a study and saves it, prepares a
    study, runs one of its tasks, resumes it, or plots a saved study, as the command
    line (`argv`, default `sys.argv`) says.

    `name` names the study and its folder. `--method` chooses configurations by their
    labels, each run with its own budget. `study_settings` are the defaults the
    flags override. `optimum` is the stated oracle optimum of a benchmark experiment,
    which regret is measured from. `true_curves` draws the measurement's values over the
    grid in the iteration figures, which only a cheap measurement (an oracle) can
    afford. `figures` are the study figures `plot` draws, by file name: functions
    of the study that draw on a new figure (default `FIGURES`, convergence and
    regret), e.g. a sweep's `hb.plots.budget_heatmap`.

    Tasks run as `python -m <module> run-task`, the module that calls `main`, so
    `run` and `resume` need it called from the experiment's module run as
    `__main__`. A study that doesn't finish exits with status 1 (a task failed) or
    130 (Ctrl-C), after saying how to resume it.
    """
    # The module that called, which tasks are run as
    frame = inspect.currentframe()
    caller = frame.f_back.f_globals if frame and frame.f_back else {}
    module = _experiment_module(caller)

    # The command: the first argument if it names one, else a run
    args = list(sys.argv[1:] if argv is None else argv)
    command, rest = (
        (args[0], args[1:]) if args[:1] and args[0] in COMMANDS else ("run", args)
    )
    parser = _parsers(name, configurations)[command]
    parsed = parser.parse_args(rest)
    results_root = _from_init_cwd(parsed.results_root)
    experiment_dir = results_root / name
    experiment = Experiment(name, problem, measure, configurations, optimum)

    # Plot: the study given, else the newest finished one
    if command == "plot":
        study_dir = _study_to_plot(parser, experiment_dir, parsed.study)

    # One task, in this process, as the launcher (or a scheduler) started it
    elif command == "run-task":
        study_dir = _existing_study(parser, experiment_dir, parsed.study)
        sys.exit(run_one_task(parser, experiment, study_dir, parsed.task_index))

    # Prepare, run or resume: the code version, then the study's task list
    else:
        version = code_version(Path(hb.__file__).parent)
        if not is_clean(version) and parsed.require_clean:
            parser.error(f"the code version {version} isn't clean")
        launcher = None if command == "prepare" else _launcher(parser, name, module)
        if command == "resume":
            study_dir = _existing_study(parser, experiment_dir, parsed.study)
            task_list = _resumable_task_list(
                parser, experiment, study_dir, parsed, version
            )
        else:
            study_dir, task_list = prepare(
                parser, experiment, experiment_dir, study_settings, parsed, version
            )

        # Run the tasks left and assemble the study, logging to its study log
        with logging_to(study_dir / STUDY_LOG, mode="a", console_format=CONSOLE_FORMAT):
            if not is_clean(version):
                _logger.warning(
                    "The code version %s isn't clean (uncommitted or untracked "
                    "files), so this study won't be reproducible",
                    version,
                )
            if launcher is None:
                _log_tasks(name, study_dir, task_list, parsed.results_root)
                return
            if command == "run":
                _log_configurations(task_list)
            resume = _command(name, "resume", study_dir.name, parsed.results_root)
            try:
                launch(
                    launcher,
                    study_dir,
                    task_list,
                    n_workers=parsed.n_workers,
                    results_root=results_root,
                )
                assemble_study(experiment, study_dir, task_list)
            except KeyboardInterrupt:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                _logger.error(NOT_FINISHED, resume)
                sys.exit(130)
            except TaskFailedError as error:
                _logger.error("%s", error)
                _logger.error(NOT_FINISHED, resume)
                sys.exit(1)

    # Plot, also straight after a run with --plot or --plot-runs
    if command == "plot" or parsed.plot or parsed.plot_runs is not None:
        # Imported here, as in `plot`: tasks never plot, so they start without it
        import matplotlib.pyplot as plt

        runs = _plot_runs(parser, parsed.plot_runs)
        with plt.style.context(hb.plots.STYLE):
            plot(
                name,
                problem,
                measure if true_curves else None,
                study_dir,
                runs=runs,
                figures=figures,
            )


def _launcher(
    parser: argparse.ArgumentParser, name: str, module: tuple[Path, str] | None
) -> tuple[Path, str]:
    """The module that tasks are run as, `module`; None, when `main` wasn't called
    from the experiment run as a module, is a usage error."""
    if module is None:
        parser.error(
            "running tasks needs the experiment run as a module: "
            f"python -m {name} (pixi run experiment {name})"
        )
    return module


def _experiment_module(caller: Mapping[str, Any]) -> tuple[Path, str] | None:
    """The folder and name of the module with globals `caller`, as `python -m` runs
    it from that folder, if it runs as `__main__` from a file; else None."""
    file = caller.get("__file__")
    if caller.get("__name__") != "__main__" or file is None:
        return None
    spec = caller.get("__spec__")
    return Path(file).parent, spec.name if spec is not None else Path(file).stem


def _parsers(
    name: str, configurations: Mapping[str, hb.Configuration]
) -> dict[str, argparse.ArgumentParser]:
    """The parsers of the commands, by name: `run` (the default), `prepare`,
    `run-task`, `resume` and `plot`."""
    # Running, `pixi run experiment <name>`, and preparing, which takes its flags
    # but runs nothing
    run_parser = argparse.ArgumentParser(
        prog=f"pixi run experiment {name}",
        description=f"Run the configurations of {name} as a study and save it.",
        epilog=(
            f"Other commands: pixi run experiment {name} prepare | run-task | "
            f"resume | plot (each with --help). To plot a saved study: "
            f"pixi run plot {name} [STUDY]"
        ),
    )
    prepare_parser = argparse.ArgumentParser(
        prog=f"pixi run experiment {name} prepare",
        description=(
            f"Reserve a study of {name} and write its task list, running nothing: "
            "run its tasks with run-task (e.g. a SLURM array), then resume it."
        ),
    )
    for parser in (run_parser, prepare_parser):
        _add_choices(parser, configurations)
        parser.add_argument("--name", help="the study id; default: the UTC time")
        _add_require_clean(parser)
        parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    _add_running(run_parser)

    # One task of a prepared study
    task_parser = argparse.ArgumentParser(
        prog=f"pixi run experiment {name} run-task",
        description=(
            f"Run one task of a prepared study of {name} and save its run; exits 0 "
            "once the task is done, also if it was already."
        ),
    )
    task_parser.add_argument("study", help="the study id")
    task_parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="the task, by its place in the study's task list, from 0",
    )
    task_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)

    # Resuming a study that didn't finish; the flags given must be the study's
    resume_parser = argparse.ArgumentParser(
        prog=f"pixi run experiment {name} resume",
        description=(
            f"Run the tasks of a study of {name} that have no saved run, then "
            "assemble the study. --method, --n-runs and --seed, if given, must be "
            "the study's."
        ),
    )
    resume_parser.add_argument("study", help="the study id")
    _add_choices(resume_parser, configurations)
    _add_require_clean(resume_parser)
    resume_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    _add_running(resume_parser)

    # Plotting, `pixi run plot <name>`
    plot_parser = argparse.ArgumentParser(
        prog=f"pixi run plot {name}",
        description=f"Plot a saved study of {name} into its figures/ folder.",
    )
    plot_parser.add_argument("study", nargs="?", help="the study id; default: newest")
    _add_plot_runs(plot_parser)
    plot_parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return {
        "run": run_parser,
        "prepare": prepare_parser,
        "run-task": task_parser,
        "resume": resume_parser,
        "plot": plot_parser,
    }


def _add_choices(
    parser: argparse.ArgumentParser, configurations: Mapping[str, hb.Configuration]
) -> None:
    """Adds the flags that choose a study's runs: `--method`, `--n-runs`, `--seed`."""
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=list(configurations),
        metavar="METHOD",
        help=(
            f"a configuration to run, by its label (repeatable): "
            f"{', '.join(configurations)}; default every one"
        ),
    )
    parser.add_argument("--n-runs", type=int, help="default: the experiment's")
    parser.add_argument("--seed", type=int, help="default: the experiment's")


def _add_require_clean(parser: argparse.ArgumentParser) -> None:
    """Adds `--require-clean`."""
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to run with uncommitted or untracked files",
    )


def _add_running(parser: argparse.ArgumentParser) -> None:
    """Adds the flags of a command that runs tasks: `--n-workers`, `--plot` and
    `--plot-runs`."""
    parser.add_argument(
        "--n-workers",
        type=_positive,
        default=1,
        help="the tasks to run at once, each in its own process (default 1)",
    )
    parser.add_argument("--plot", action="store_true", help="plot after running")
    _add_plot_runs(parser, " (implies --plot)")


def _positive(text: str) -> int:
    """A positive integer.

    Raises `argparse.ArgumentTypeError` otherwise.
    """
    if not text.isdigit() or int(text) < 1:
        msg = f"not a positive integer: {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(text)


def _add_plot_runs(parser: argparse.ArgumentParser, note: str = "") -> None:
    """Adds `--plot-runs` to the parser: run indices or `all`, None when not given."""
    parser.add_argument(
        "--plot-runs",
        nargs="+",
        type=_plot_run,
        metavar="RUN",
        help=f"the runs with iteration figures: indices or all; default 0{note}",
    )


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


def _command(name: str, command: str, arguments: str, results_root: Path) -> str:
    """The command line of `command` with `arguments`, with the results root if it
    isn't the default."""
    line = f"pixi run experiment {name} {command} {arguments}"
    if results_root != RESULTS_ROOT:
        line += f" --results-root {results_root}"
    return line


# The task list


def prepare(
    parser: argparse.ArgumentParser,
    experiment: Experiment,
    experiment_dir: Path,
    study_settings: hb.StudySettings,
    parsed: argparse.Namespace,
    version: str,
) -> tuple[Path, dict[str, Any]]:
    """Reserves the study and writes its task list, as the flags choose, and returns
    the study's folder and task list. A taken name is a usage error
    (`parser.error`)."""
    # The experiment's settings, overridden by the flags given, with the seed resolved
    given = {"n_runs": parsed.n_runs, "seed": parsed.seed}
    settings = replace(
        study_settings,
        **{key: value for key, value in given.items() if value is not None},
    )
    settings = replace(settings, seed=np.random.SeedSequence(settings.seed).entropy)
    labels = list(dict.fromkeys(parsed.methods or experiment.configurations))
    task_list = task_list_of(experiment, labels, settings, version)

    # Reserve the study id: a taken name is refused, a taken UTC time suffixed
    default_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    try:
        study_dir = reserve(
            experiment_dir, parsed.name or default_id, suffix_if_taken=not parsed.name
        )
    except FileExistsError as error:
        parser.error(str(error))
    text = json.dumps(task_list, indent=1) + "\n"
    _write_atomically(study_dir / TASK_LIST, lambda path: path.write_text(text))
    return study_dir, task_list


def task_list_of(
    experiment: Experiment,
    labels: Sequence[str],
    settings: hb.StudySettings,
    version: str,
) -> dict[str, Any]:
    """The task list of a study of the experiment's configurations `labels` with
    `settings` (its seed resolved), at the code version `version`, as JSON data."""
    chosen = {label: experiment.configurations[label] for label in labels}
    optimum = experiment.optimum
    return {
        "name": experiment.name,
        "seed": settings.seed,
        "n_runs": settings.n_runs,
        "configurations": {
            label: {
                "n_evaluations": configuration.n_evaluations,
                "n_initial": configuration.n_initial,
            }
            for label, configuration in chosen.items()
        },
        "optimum": None if optimum is None else float(optimum),
        "code_version": version,
        "problem": problem_artifact(experiment.problem, experiment.measure),
        "tasks": [
            [task.label, task.index] for task in hb.study_tasks(chosen, settings)
        ],
    }


def differences(
    experiment: Experiment,
    task_list: Mapping[str, Any],
    version: str,
    *,
    methods: Sequence[str] | None = None,
    n_runs: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """What differs between the study's task list `task_list` and the experiment at
    the code version `version`, with the flags given (None takes the study's), by
    the task list's keys; empty if nothing does."""
    # The study's labels, in its order, unless the flags choose others
    labels = list(task_list["configurations"])
    if methods and set(methods) != set(labels):
        labels = list(dict.fromkeys(methods))
    if any(label not in experiment.configurations for label in labels):
        return ["configurations"]
    settings = hb.StudySettings(
        n_runs=task_list["n_runs"] if n_runs is None else n_runs,
        seed=task_list["seed"] if seed is None else seed,
    )
    expected = task_list_of(experiment, labels, settings, version)
    return [key for key in expected if task_list.get(key) != expected[key]]


def _read_task_list(parser: argparse.ArgumentParser, study_dir: Path) -> dict[str, Any]:
    """The study's task list; a study without one is a usage error."""
    path = study_dir / TASK_LIST
    if not path.exists():
        parser.error(f"{study_dir} has no task list ({TASK_LIST}), so it can't be run")
    return json.loads(path.read_text())


def _resumable_task_list(
    parser: argparse.ArgumentParser,
    experiment: Experiment,
    study_dir: Path,
    parsed: argparse.Namespace,
    version: str,
) -> dict[str, Any]:
    """The task list of the study to resume. A finished study, or one whose task
    list differs from the experiment and the flags given, is a usage error."""
    if _is_finished(study_dir):
        parser.error(
            f"the study {study_dir} is finished, so there is nothing to resume"
        )
    task_list = _read_task_list(parser, study_dir)
    different = differences(
        experiment,
        task_list,
        version,
        methods=parsed.methods,
        n_runs=parsed.n_runs,
        seed=parsed.seed,
    )
    if different:
        parser.error(
            f"the study {study_dir.name} can't be resumed: its task list differs from "
            f"the experiment in {', '.join(different)}"
        )
    return task_list


def _settings(task_list: Mapping[str, Any]) -> hb.StudySettings:
    """The study's settings, as its task list holds them."""
    return hb.StudySettings(n_runs=task_list["n_runs"], seed=task_list["seed"])


def _tasks(experiment: Experiment, task_list: Mapping[str, Any]) -> tuple[hb.Task, ...]:
    """The study's tasks, in the order of its task list."""
    chosen = {
        label: experiment.configurations[label] for label in task_list["configurations"]
    }
    return hb.study_tasks(chosen, _settings(task_list))


def _log_configurations(task_list: Mapping[str, Any]) -> None:
    """Logs at info level each configuration's runs and budget, and the seed."""
    for label, budget in task_list["configurations"].items():
        _logger.info(
            "Configuration %s: %d runs of %d evaluations, %d of them initial points",
            label,
            task_list["n_runs"],
            budget["n_evaluations"],
            budget["n_initial"],
        )
    _logger.info("Seed %d, %d tasks", task_list["seed"], len(task_list["tasks"]))


def _log_tasks(
    name: str, study_dir: Path, task_list: Mapping[str, Any], results_root: Path
) -> None:
    """Logs the prepared study's configurations and tasks, and prints how to run
    each task and then assemble the study."""
    _log_configurations(task_list)
    n_tasks = len(task_list["tasks"])
    run_task_line = _command(
        name, "run-task", f"{study_dir.name} --task-index K", results_root
    )
    print(
        f"Prepared {study_dir} with {n_tasks} tasks. Run task K (0 to {n_tasks - 1}) "
        f"with\n  {run_task_line}\nthen assemble the study with\n  "
        f"{_command(name, 'resume', study_dir.name, results_root)}"
    )


# The study folder


def reserve(experiment_dir: Path, study_id: str, *, suffix_if_taken: bool) -> Path:
    """Reserves a study id in `experiment_dir` by creating its folder, and returns
    the folder. Creating it fails if it exists, so studies started at once never
    share an id, and a study that didn't finish keeps its id.

    A taken id gets `-2`, `-3`, ... appended if `suffix_if_taken`; otherwise raises
    `FileExistsError` saying what holds it.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate, n = study_id, 2
    while True:
        folder = experiment_dir / candidate
        try:
            folder.mkdir()
        except FileExistsError:
            finished = _is_finished(folder)
            holder = "the saved study" if finished else "the unfinished study"
        else:
            return folder
        if not suffix_if_taken:
            msg = f"the study id {candidate} is taken by {holder} {folder}"
            raise FileExistsError(msg)
        candidate, n = f"{study_id}-{n}", n + 1


def _is_finished(study_dir: Path) -> bool:
    """Whether the study in `study_dir` is finished: its artifact is saved."""
    return any((study_dir / name).exists() for name in ARTIFACTS)


def _existing_study(
    parser: argparse.ArgumentParser, experiment_dir: Path, study_id: str
) -> Path:
    """The folder of the study `study_id`; one that doesn't exist is a usage error."""
    study_dir = experiment_dir / study_id
    if not study_dir.is_dir():
        parser.error(f"{experiment_dir} holds no study {study_id}")
    return study_dir


def _study_to_plot(
    parser: argparse.ArgumentParser, experiment_dir: Path, study_id: str | None
) -> Path:
    """The study `study_id`, else the finished study saved last. No finished study, or
    an unfinished one named, is a usage error."""
    if study_id is None:
        saved = [
            path for name in ARTIFACTS for path in experiment_dir.glob(f"*/{name}")
        ]
        if not saved:
            parser.error(f"{experiment_dir} holds no study")
        return max(saved, key=lambda path: path.stat().st_mtime).parent
    study_dir = experiment_dir / study_id
    if not _is_finished(study_dir) and (study_dir / TASK_LIST).exists():
        parser.error(f"the study {study_id} didn't finish: resume it first")
    if not _is_finished(study_dir):
        parser.error(f"{study_dir} holds no study")
    return study_dir


def run_tag(label: str, index: int) -> str:
    """The tag of run `index` of the configuration `label`, `label/run-003`, as its
    lines in the study log start and its files are named."""
    return f"{label}/run-{index:03d}"


def _run_file(study_dir: Path, label: str, index: int) -> Path:
    """The run file of run `index` of `label`, where `hb.save` puts it."""
    return study_dir / RUN_FILES / f"{run_tag(label, index)}.json"


def _failure_marker(study_dir: Path, label: str, index: int) -> Path:
    """The file that says run `index` of `label` failed before evaluating anything."""
    return _run_file(study_dir, label, index).with_suffix(".failed")


def _run_log(study_dir: Path, label: str, index: int) -> Path:
    """The per-run log of run `index` of `label`."""
    return study_dir / PER_RUN_LOGS / f"{run_tag(label, index)}.log"


def _is_done(study_dir: Path, label: str, index: int) -> bool:
    """Whether the task of run `index` of `label` is done: its run is saved, or it
    failed before evaluating anything."""
    run_file = _run_file(study_dir, label, index)
    return run_file.exists() or _failure_marker(study_dir, label, index).exists()


def _write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Writes the file `path` with `write`, which writes to the path it is given, so
    the file appears only when complete: to a temporary file beside it, then
    renamed."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    write(temporary)
    temporary.replace(path)


# Running the tasks


def run_one_task(
    parser: argparse.ArgumentParser,
    experiment: Experiment,
    study_dir: Path,
    task_index: int,
) -> int:
    """Runs the study's task `task_index` (`hb.run_task`) unless it is done, and
    saves its run file, or its failure marker, so it appears only when complete.
    Logs to the run's per-run log (DEBUG and above from the library) and to the
    console (INFO and above), tagged with the run, warnings included.

    Returns the exit status: 0 once the task is done, 1 if the run raised an
    exception, logged with its traceback. An experiment that differs from the task
    list, or an index outside it, is a usage error (`parser.error`, status 2).
    """
    # The task, from the task list the experiment matches
    task_list = _read_task_list(parser, study_dir)
    different = differences(
        experiment, task_list, code_version(Path(hb.__file__).parent)
    )
    if different:
        parser.error(
            f"the experiment differs from the task list of {study_dir.name} in "
            f"{', '.join(different)}"
        )
    tasks = _tasks(experiment, task_list)
    if not 0 <= task_index < len(tasks):
        parser.error(f"the study has tasks 0 to {len(tasks) - 1}, not {task_index}")
    task = tasks[task_index]
    tag = run_tag(task.label, task.index)
    if _is_done(study_dir, task.label, task.index):
        print(f"Task {task_index} ({tag}) is done already", file=sys.stderr)
        return 0

    # The run, logged to its per-run log, saved as it finishes
    run_file = _run_file(study_dir, task.label, task.index)
    log_path = _run_log(study_dir, task.label, task.index)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_file.parent.mkdir(parents=True, exist_ok=True)
    with logging_to(log_path, mode="w", console_format=f"{tag} {CONSOLE_FORMAT}"):
        try:
            run = hb.run_task(
                task, experiment.problem, experiment.measure, _settings(task_list)
            )
            if run is None:
                marker = _failure_marker(study_dir, task.label, task.index)
                _write_atomically(
                    marker,
                    lambda path: path.write_text("failed before evaluating anything\n"),
                )
            else:
                labeled = replace(run, method=task.label)
                _write_atomically(run_file, lambda path: hb.save_run(labeled, path))
        except Exception:
            _logger.exception("Task %d (%s) failed", task_index, tag)
            return 1
    return 0


def launch(
    module: tuple[Path, str],
    study_dir: Path,
    task_list: Mapping[str, Any],
    *,
    n_workers: int,
    results_root: Path,
) -> None:
    """Runs the study's tasks that aren't done, each as `python -m <module> run-task`
    from its folder (`module`), keeping `n_workers` running, in task order, and logs
    each task as it starts and finishes.

    Each task runs in its own process group, so a Ctrl-C in the terminal reaches only
    this process, and with the warning options of this one. With more than one
    worker, the tasks run BLAS and OpenMP on one thread each, unless their variables
    are set here.

    Raises `TaskFailedError` when a task exits with a status other than 0, and
    `KeyboardInterrupt` on Ctrl-C, after stopping the tasks running; after Ctrl-C this
    process ignores Ctrl-C.
    """
    # The tasks left, in order, and their environment
    tasks = task_list["tasks"]
    left = deque(
        index
        for index, (label, run_index) in enumerate(tasks)
        if not _is_done(study_dir, label, run_index)
    )
    n_done = len(tasks) - len(left)
    if n_done:
        _logger.info("%d of %d tasks are done already", n_done, len(tasks))
    environment = dict(os.environ)
    if n_workers > 1:
        for variable in THREAD_VARIABLES:
            environment.setdefault(variable, "1")

    # Ctrl-C only marks the study interrupted, checked once per poll, so it never
    # lands between starting a task and keeping track of it
    interrupts: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _: interrupts.append(signum))

    # Keep n_workers running; a task that fails stops the study
    running: dict[int, subprocess.Popen[bytes]] = {}
    try:
        while left or running:
            if interrupts:
                raise KeyboardInterrupt
            while left and len(running) < n_workers:
                index = left.popleft()
                running[index] = _start(
                    module, study_dir.name, index, results_root, environment
                )
                _logger.info("Started task %d (%s)", index, run_tag(*tasks[index]))
            for index, process in list(running.items()):
                status = process.poll()
                if status is None:
                    continue
                del running[index]
                tag = run_tag(*tasks[index])
                if status != 0:
                    msg = (
                        f"Task {index} ({tag}) failed with exit status {status}; its "
                        f"log is {_run_log(study_dir, *tasks[index])}"
                    )
                    raise TaskFailedError(msg)
                n_done += 1
                _logger.info(
                    "Finished task %d (%s): %d of %d done",
                    index,
                    tag,
                    n_done,
                    len(tasks),
                )
            time.sleep(POLL_INTERVAL)

    # On a failure or Ctrl-C, the tasks still running are stopped at once, as their
    # runs may never see the interrupt (CasADi catches it inside IPOPT). Ctrl-C ends
    # the study, so another one is ignored from then on: pixi passes a Ctrl-C on to
    # the process it runs, which so gets it twice
    finally:
        _stop(list(running.values()))
        signal.signal(signal.SIGINT, signal.SIG_IGN if interrupts else previous)


def _start(
    module: tuple[Path, str],
    study_id: str,
    index: int,
    results_root: Path,
    environment: Mapping[str, str],
) -> "subprocess.Popen[bytes]":
    """Starts task `index` of the study as `python -m <module> run-task`, in a new
    process group."""
    folder, name = module
    command = [
        sys.executable,
        *(f"-W{option}" for option in sys.warnoptions),
        "-m",
        name,
        "run-task",
        study_id,
        "--task-index",
        str(index),
        "--results-root",
        str(results_root),
    ]
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            cwd=folder,
            env=environment,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        command, cwd=folder, env=environment, start_new_session=True
    )


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
    experiment: Experiment, study_dir: Path, task_list: Mapping[str, Any]
) -> hb.Study:
    """The study, from its run files in the order of its task list (`hb.assemble`).
    Appends the per-run logs to the study log and saves the summary, then the
    artifact, which finishes the study, so a study that is finished has both; logs
    how many runs of each configuration were saved."""
    # Each task's run, None for a run that failed before evaluating anything
    tasks = _tasks(experiment, task_list)
    runs = [
        None
        if _failure_marker(study_dir, task.label, task.index).exists()
        else hb.load_run(_run_file(study_dir, task.label, task.index))
        for task in tasks
    ]
    study = hb.assemble(
        tasks,
        runs,
        experiment.problem,
        experiment.measure,
        _settings(task_list),
        name=task_list["name"],
        optimum=task_list["optimum"],
    )

    # The logs and the summary, then the artifact
    _append_run_logs(study_dir, task_list)
    save_summary(study, study_dir)
    save_artifact(study, study_dir)
    for label in task_list["configurations"]:
        n_saved = sum(run.method == label for run in study.runs)
        _logger.info(
            "%s: saved %d of %d runs to %s",
            label,
            n_saved,
            task_list["n_runs"],
            study_dir,
        )
    return study


def _append_run_logs(study_dir: Path, task_list: Mapping[str, Any]) -> None:
    """Appends each per-run log to the study log, in the order of the task list,
    every line starting with the run's tag."""
    with (study_dir / STUDY_LOG).open("a") as study_log:
        for label, index in task_list["tasks"]:
            run_log = _run_log(study_dir, label, index)
            if not run_log.exists():
                continue
            tag = run_tag(label, index)
            for line in run_log.read_text().splitlines():
                study_log.write(f"{tag} {line}\n")


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


def save_summary(study: hb.Study, study_dir: Path) -> None:
    """Prints `hb.summary(study)` and saves it to `summary.csv` in `study_dir`."""
    # Imported here, so tasks, which never summarize, start without it
    import polars as pl

    table = hb.summary(study)
    table.write_csv(study_dir / "summary.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(table)


def plot(
    name: str,
    problem: hb.Problem,
    measure: hb.Measure | None,
    study_dir: Path,
    *,
    runs: Sequence[int] | None,
    figures: Mapping[str, Callable[[hb.Study], object]] = FIGURES,
) -> None:
    """Saves the figures of the study in `study_dir` to its `figures/` folder:
    `<name>.png` for each of `figures` (by default `convergence.png` and
    `regret.png`, when the study has an oracle optimum) and
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
    if not (study_dir / "summary.csv").exists():
        save_summary(study, study_dir)
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # The study figures, each skipped if the study can't give it (regret without
    # an oracle optimum)
    for figure_name, figure in figures.items():
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
