"""The study folder (`hybrid_bo.study_folder`): preparing a study, reading its task
list back checked against the experiment, which tasks are done, and assembling the
study from its run files.

The tests run in this process against a temporary folder; `test_cli.py` runs the same
folders through the command line, as task processes.
"""

from pathlib import Path

import illustrative_example as ex
import pytest
from test_studies import assert_same

import hybrid_bo as hb
from hybrid_bo.errors import StudyFolderError
from hybrid_bo.saving import load
from hybrid_bo.studies import Configuration, StudySettings, run_task
from hybrid_bo.study_folder import (
    Experiment,
    assemble_study,
    finished_study,
    is_done,
    open_study,
    per_run_log,
    prepare,
    save_result,
    study_log,
    study_settings,
    task,
    tasks,
    tasks_left,
)


def experiment_of(n_evaluations=3):
    """The illustrative example with two quick configurations, uniform and Latin
    hypercube sampling, of `n_evaluations` evaluations, 2 of them initial points."""
    return Experiment(
        script=Path("small_example.py"),
        problem=ex.problem,
        measure=ex.measure,
        configurations={
            "uniform": Configuration(
                method=hb.uniform_sampling, n_evaluations=n_evaluations, n_initial=2
            ),
            "lhs": Configuration(
                method=hb.latin_hypercube_sampling,
                n_evaluations=n_evaluations,
                n_initial=2,
            ),
        },
        optimum=ex.optimum,
    )


def prepared(tmp_path, experiment=None, *, study_id="s", version="v1"):
    """A study of `experiment` (default `experiment_of()`), 2 runs of each
    configuration from seed 0, prepared in `tmp_path` at the code version
    `version`."""
    experiment = experiment_of() if experiment is None else experiment
    return prepare(
        tmp_path,
        experiment,
        list(experiment.configurations),
        n_runs=2,
        seed=0,
        version=version,
        study_id=study_id,
        suffix_if_taken=False,
    )


def run_all(folder, experiment):
    """Runs and saves every task left of the study, in this process."""
    for _, each in tasks_left(folder, experiment):
        run = run_task(
            each, experiment.problem, experiment.measure, study_settings(folder)
        )
        save_result(folder, each, run)


# Preparing and opening


def test_a_prepared_study_opens_with_the_same_task_list(tmp_path):
    """`open_study` reads back the task list `prepare` wrote, with its tasks by label,
    then index."""
    experiment = experiment_of()
    folder = prepared(tmp_path, experiment)

    opened = open_study(folder.path, experiment, "v1")

    assert opened == folder
    assert folder.task_list.tasks == (
        ("uniform", 0),
        ("uniform", 1),
        ("lhs", 0),
        ("lhs", 1),
    )
    assert [each.label for each in tasks(opened, experiment)] == [
        "uniform",
        "uniform",
        "lhs",
        "lhs",
    ]


def test_an_unfinished_study_keeps_its_id_taken(tmp_path):
    """A study's folder reserves its id, so a study that didn't finish keeps it: a new
    study gets a suffix, or is refused when it asked for that id."""
    experiment = experiment_of()
    (tmp_path / "old").mkdir()

    suffixed = prepare(
        tmp_path,
        experiment,
        ["uniform"],
        n_runs=1,
        seed=0,
        version="v1",
        study_id="old",
        suffix_if_taken=True,
    )

    assert suffixed.path.name == "old-2"
    # `pytest.raises` checks that the block raises this exception, with a message
    # matching `match`
    with pytest.raises(StudyFolderError, match="the study id old is taken"):
        prepared(tmp_path, experiment, study_id="old")


def test_opening_names_what_differs_from_the_experiment(tmp_path):
    """`open_study` refuses a task list that differs from the experiment, naming the
    field, with both values when it is one value, so a study never mixes runs of two
    experiments or two code versions."""
    folder = prepared(tmp_path)

    with pytest.raises(
        StudyFolderError, match="differs from the experiment in configurations"
    ):
        open_study(folder.path, experiment_of(n_evaluations=4), "v1")
    with pytest.raises(
        StudyFolderError, match=r"code_version \('v1' in the study, 'v2' now\)"
    ):
        open_study(folder.path, experiment_of(), "v2")


def test_a_folder_without_a_task_list_holds_no_study(tmp_path):
    """`open_study` refuses a folder with no task list."""
    (tmp_path / "empty").mkdir()

    with pytest.raises(StudyFolderError, match="holds no unfinished study"):
        open_study(tmp_path / "empty", experiment_of(), "v1")


# Tasks


def test_a_task_is_done_once_its_run_or_failure_marker_is_saved(tmp_path):
    """A task is done once its run is saved, or a failure marker for a run that failed
    before evaluating anything; `tasks_left` skips done tasks and keeps the indices of
    the rest."""
    experiment = experiment_of()
    folder = prepared(tmp_path, experiment)
    first, second = task(folder, experiment, 0), task(folder, experiment, 1)
    run = run_task(
        first, experiment.problem, experiment.measure, study_settings(folder)
    )

    save_result(folder, first, run)
    save_result(folder, second, None)

    assert is_done(folder, first)
    assert is_done(folder, second)
    assert [index for index, _ in tasks_left(folder, experiment)] == [2, 3]
    with pytest.raises(StudyFolderError, match="tasks 0 to 3, not 4"):
        task(folder, experiment, 4)


# Assembling


def test_the_assembled_study_is_the_serial_study(tmp_path):
    """A study assembled from its run files, saved task by task, is the study `study`
    runs in one go, and it is finished."""
    experiment = experiment_of()
    folder = prepared(tmp_path, experiment)
    run_all(folder, experiment)

    assembled = assemble_study(folder, experiment)

    serial = hb.study(
        dict(experiment.configurations),
        ex.problem,
        ex.measure,
        StudySettings(n_runs=2, seed=0),
        name="small_example",
        optimum=ex.optimum,
    )
    assert_same(assembled, serial)
    assert_same(load(folder.path), serial)
    assert finished_study(tmp_path, None) == folder.path


def test_assembling_again_writes_each_run_log_once(tmp_path):
    """The study log is the study's own lines, then each per-run log tagged with its
    run, written once even when an earlier assembly was stopped after writing it."""
    # A study whose runs each logged one line, and one line of its own
    experiment = experiment_of()
    folder = prepared(tmp_path, experiment)
    run_all(folder, experiment)
    for each in tasks(folder, experiment):
        per_run_log(folder, each).parent.mkdir(parents=True, exist_ok=True)
        per_run_log(folder, each).write_text("a line\n")
    study_log(folder).write_text("the launcher's line\n")

    # Assembled, then its artifact removed as if stopped before it, and assembled again
    assemble_study(folder, experiment)
    (folder.path / "artifact.json").unlink()
    assemble_study(folder, experiment)

    assert study_log(folder).read_text().splitlines() == [
        "the launcher's line",
        "uniform/run-000 a line",
        "uniform/run-001 a line",
        "lhs/run-000 a line",
        "lhs/run-001 a line",
    ]


def test_an_unfinished_study_is_not_plotted(tmp_path):
    """`finished_study` refuses a study named that didn't finish, and an experiment
    folder with no finished study."""
    folder = prepared(tmp_path)

    with pytest.raises(StudyFolderError, match="didn't finish: resume it first"):
        finished_study(tmp_path, folder.path.name)
    with pytest.raises(StudyFolderError, match="holds no study"):
        finished_study(tmp_path, None)
