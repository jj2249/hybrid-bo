"""The study folder: what a study run as tasks keeps on disk, and what is left to do.

A study folder, `<experiment dir>/<study id>/`, holds the task list (`tasks.json`),
written when the study is prepared; each task's run file (`runs/<label>/run-003.json`)
or failure marker (`.failed`), and its per-run log (`logs/<label>/run-003.log`); the
study log (`study.log`); and, once the study is assembled, its artifact, which
finishes it. Creating the folder reserves the study id. The files are the checkpoint:
a task is done exactly when its run file or failure marker exists (ADR 0010).

`prepare` reserves a folder and writes its task list, and `open_study` reads one back,
checked against the experiment, so a `StudyFolder` always holds a task list that
matches its experiment. `hb.cli` is the launcher on top: it parses the command line,
runs the tasks as processes and logs to the paths given here.
"""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hybrid_bo.errors import StudyFolderError
from hybrid_bo.evaluation import Measure
from hybrid_bo.problem import Problem
from hybrid_bo.run import Run, problem_artifact
from hybrid_bo.saving import (
    ARTIFACT_FILE,
    OLD_ARTIFACT_FILE,
    load_run,
    save_artifact,
    save_run,
)
from hybrid_bo.studies import (
    Configuration,
    Study,
    StudySettings,
    Task,
    assemble,
    study_tasks,
)

_ARTIFACTS = (ARTIFACT_FILE, OLD_ARTIFACT_FILE)
_TASK_LIST = "tasks.json"
_STUDY_LOG = "study.log"
_PER_RUN_LOGS = "logs"
_RUN_FILES = "runs"


@dataclass(frozen=True, kw_only=True)
class Experiment:
    """What a study script defines that its studies depend on: the script, whose file
    name (without `.py`) is the name of its studies, and the problem, measurement,
    configurations and stated oracle optimum it passes to `hb.cli.main`."""

    script: Path
    problem: Problem
    measure: Measure
    configurations: Mapping[str, Configuration]
    optimum: float | None

    @property
    def name(self) -> str:
        """The name of the script's studies: its file name without `.py`."""
        return self.script.stem


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


@dataclass(frozen=True)
class StudyFolder:
    """A study's folder and its task list, as `prepare` wrote it or `open_study` read
    and checked it."""

    path: Path
    task_list: TaskList


def prepare(
    experiment_dir: Path,
    experiment: Experiment,
    labels: Sequence[str],
    *,
    n_runs: int,
    seed: int,
    version: str,
    study_id: str,
    suffix_if_taken: bool,
) -> StudyFolder:
    """Reserves the study `study_id` in `experiment_dir` and writes its task list: the
    experiment's configurations `labels`, `n_runs` runs each from `seed`, at the code
    version `version`. The folder is created first, so studies prepared at once never
    share an id, and an unfinished study keeps its id.

    A taken id gets `-2`, `-3`, ... appended if `suffix_if_taken`; otherwise raises
    `StudyFolderError`.
    """
    task_list = _task_list_of(
        experiment, labels, n_runs=n_runs, seed=seed, version=version
    )
    path = _reserve(experiment_dir, study_id, suffix_if_taken=suffix_if_taken)
    text = json.dumps(dataclasses.asdict(task_list), indent=1) + "\n"
    _write_atomically(path / _TASK_LIST, lambda temporary: temporary.write_text(text))
    return StudyFolder(path, task_list)


def open_study(study_dir: Path, experiment: Experiment, version: str) -> StudyFolder:
    """The study folder `study_dir`, finished or not, with its task list.

    Raises `StudyFolderError` if it has no task list, or its task list differs from
    the experiment at the code version `version`, naming what differs.
    """
    # The task list
    path = study_dir / _TASK_LIST
    if not path.exists():
        msg = f"{study_dir} holds no unfinished study (no {_TASK_LIST})"
        raise StudyFolderError(msg)
    data = json.loads(path.read_text())
    tasks = tuple((label, index) for label, index in data.pop("tasks"))
    task_list = TaskList(**data, tasks=tasks)

    # Checked against the experiment now
    different = _differences(task_list, experiment, version)
    if different:
        msg = f"the study's task list differs from the experiment in {', '.join(different)}"
        raise StudyFolderError(msg)
    return StudyFolder(study_dir, task_list)


def is_finished(study_dir: Path) -> bool:
    """Whether the study in `study_dir` is finished: its artifact is saved."""
    return any((study_dir / name).exists() for name in _ARTIFACTS)


def finished_study(experiment_dir: Path, study_id: str | None) -> Path:
    """The folder of the finished study `study_id`, else of the finished study saved
    last.

    Raises `StudyFolderError` if there is none, or the study named is unfinished.
    """
    # The newest, by when its artifact was written
    if study_id is None:
        saved = [
            path for name in _ARTIFACTS for path in experiment_dir.glob(f"*/{name}")
        ]
        if not saved:
            msg = f"{experiment_dir} holds no study"
            raise StudyFolderError(msg)
        return max(saved, key=lambda path: path.stat().st_mtime).parent

    # The one named, if it is finished
    study_dir = experiment_dir / study_id
    if not is_finished(study_dir):
        unfinished = (study_dir / _TASK_LIST).exists()
        msg = (
            f"the study {study_id} didn't finish: resume it first"
            if unfinished
            else f"{study_dir} holds no study"
        )
        raise StudyFolderError(msg)
    return study_dir


def study_settings(folder: StudyFolder) -> StudySettings:
    """The study's settings, as its task list holds them."""
    return StudySettings(n_runs=folder.task_list.n_runs, seed=folder.task_list.seed)


def tasks(folder: StudyFolder, experiment: Experiment) -> tuple[Task, ...]:
    """The study's tasks, in the order of its task list."""
    chosen = {
        label: experiment.configurations[label]
        for label in folder.task_list.configurations
    }
    return study_tasks(chosen, study_settings(folder))


def task(folder: StudyFolder, experiment: Experiment, index: int) -> Task:
    """The study's task `index`, from 0 in the order of its task list.

    Raises `StudyFolderError` if there is no such task.
    """
    all_tasks = tasks(folder, experiment)
    if not 0 <= index < len(all_tasks):
        msg = f"the study has tasks 0 to {len(all_tasks) - 1}, not {index}"
        raise StudyFolderError(msg)
    return all_tasks[index]


def tasks_left(folder: StudyFolder, experiment: Experiment) -> list[tuple[int, Task]]:
    """The study's tasks that aren't done, with their indices, in order."""
    return [
        (index, each)
        for index, each in enumerate(tasks(folder, experiment))
        if not is_done(folder, each)
    ]


def is_done(folder: StudyFolder, task: Task) -> bool:
    """Whether the task is done: its run is saved, or it failed before evaluating
    anything."""
    return _run_file(folder, task).exists() or _failure_marker(folder, task).exists()


def run_tag(task: Task) -> str:
    """The task's tag, `label/run-003`, as its lines in the study log start and its
    files are named."""
    return f"{task.label}/run-{task.index:03d}"


def per_run_log(folder: StudyFolder, task: Task) -> Path:
    """The per-run log of the task's run."""
    return folder.path / _PER_RUN_LOGS / f"{run_tag(task)}.log"


def study_log(folder: StudyFolder) -> Path:
    """The study log."""
    return folder.path / _STUDY_LOG


def save_result(folder: StudyFolder, task: Task, run: Run | None) -> None:
    """Saves the task's run, labeled with the task's label, as its run file, or a
    failure marker if `run` is None (it failed before evaluating anything); either
    appears only when complete, and makes the task done."""
    run_file = _run_file(folder, task)
    run_file.parent.mkdir(parents=True, exist_ok=True)
    if run is None:
        _write_atomically(
            _failure_marker(folder, task),
            lambda path: path.write_text("failed before evaluating anything\n"),
        )
    else:
        labeled = replace(run, method=task.label)
        _write_atomically(run_file, lambda path: save_run(labeled, path))


def assemble_study(folder: StudyFolder, experiment: Experiment) -> Study:
    """The study, from its run files in the order of its task list (`assemble`).
    Rewrites the study log as its own lines, then each per-run log, and then saves
    the artifact, which finishes the study.

    The study log is replaced, so close any file handler logging to it first (Windows
    can't replace an open file). Raises `ValueError` if a task isn't done.
    """
    # Each task's run, None for a run that failed before evaluating anything
    all_tasks = tasks(folder, experiment)
    runs = [
        None if _failure_marker(folder, each).exists() else _load(folder, each)
        for each in all_tasks
    ]
    study = assemble(
        all_tasks,
        runs,
        experiment.problem,
        experiment.measure,
        study_settings(folder),
        name=folder.task_list.name,
        optimum=folder.task_list.optimum,
    )

    # The logs, then the artifact
    _rebuild_study_log(folder, all_tasks)
    save_artifact(study, folder.path)
    return study


def _load(folder: StudyFolder, task: Task) -> Run:
    """The task's saved run.

    Raises `ValueError` if the task isn't done.
    """
    run_file = _run_file(folder, task)
    if not run_file.exists():
        msg = f"task {run_tag(task)} isn't done: {run_file} doesn't exist"
        raise ValueError(msg)
    return load_run(run_file)


def _task_list_of(
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
    settings = StudySettings(n_runs=n_runs, seed=seed)
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
        tasks=tuple((each.label, each.index) for each in study_tasks(chosen, settings)),
    )


def _differences(
    task_list: TaskList, experiment: Experiment, version: str
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


def _reserve(experiment_dir: Path, study_id: str, *, suffix_if_taken: bool) -> Path:
    """Reserves a study id in `experiment_dir` by creating its folder, and returns the
    folder, as `prepare` describes.

    Raises `StudyFolderError` for a taken id unless `suffix_if_taken`.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate, n = study_id, 2
    while True:
        try:
            (experiment_dir / candidate).mkdir()
        except FileExistsError:
            if not suffix_if_taken:
                msg = f"the study id {candidate} is taken: {experiment_dir / candidate}"
                raise StudyFolderError(msg) from None
            candidate, n = f"{study_id}-{n}", n + 1
        else:
            return experiment_dir / candidate


def _run_file(folder: StudyFolder, task: Task) -> Path:
    """The task's run file."""
    return folder.path / _RUN_FILES / f"{run_tag(task)}.json"


def _failure_marker(folder: StudyFolder, task: Task) -> Path:
    """The file that says the task's run failed before evaluating anything."""
    return _run_file(folder, task).with_suffix(".failed")


def _rebuild_study_log(folder: StudyFolder, all_tasks: Sequence[Task]) -> None:
    """Rewrites the study log as its own lines, then each per-run log in task order,
    every line starting with its run's tag. Lines tagged by an earlier, interrupted
    assembly are dropped first, so they are never there twice."""
    tags = tuple(f"{run_tag(each)} " for each in all_tasks)
    path = study_log(folder)
    lines = (
        [line for line in path.read_text().splitlines() if not line.startswith(tags)]
        if path.exists()
        else []
    )
    for each in all_tasks:
        log = per_run_log(folder, each)
        if log.exists():
            lines += [
                f"{run_tag(each)} {line}" for line in log.read_text().splitlines()
            ]
    text = "".join(f"{line}\n" for line in lines)
    _write_atomically(path, lambda temporary: temporary.write_text(text))


def _write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Writes the file `path` with `write`, which writes to the path it is given, so
    the file appears only when complete: to a temporary file beside it, then
    renamed."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    write(temporary)
    temporary.replace(path)
