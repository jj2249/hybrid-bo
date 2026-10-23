"""The command line of the experiments (experiments/cli.py, behind `pixi run experiment`
and `pixi run plot`): running studies as tasks, resuming them and plotting them.

The tests write small experiment modules and run them as `python -m` runs an
experiment, in this process (`runpy`), so the task processes the CLI starts rebuild
the same experiment from the module.
"""

import importlib
import json
import logging
import os
import runpy
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cli
import illustrative_example as ex
import numpy as np
import polars as pl
import pytest
from test_studies import assert_same

import hybrid_bo as hb
from hybrid_bo import (
    Configuration,
    StudySettings,
    initial_points,
    load,
    summary,
)

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


def summary_of(study):
    """`hb.summary(study)` as it reads back from CSV."""
    return pl.read_csv(summary(study).write_csv().encode())


def small_configurations(n_iterations=1):
    """Every method of the illustrative example, with its 2 initial points but only
    `n_iterations` iterations."""
    return {
        label: Configuration(method=method, n_evaluations=2 + n_iterations, n_initial=2)
        for label, method in ex.methods.items()
    }


def small(n_iterations=1, true_curves=False):
    """The module of a small test experiment: every method of the illustrative
    example, as `small_configurations` gives them."""
    return f"""
import cli
import illustrative_example as ex
import hybrid_bo as hb

configurations = {{
    label: hb.Configuration(method=method, n_evaluations={2 + n_iterations}, n_initial=2)
    for label, method in ex.methods.items()
}}

if __name__ == "__main__":
    cli.main(
        "illustrative_example",
        ex.problem,
        ex.measure,
        configurations,
        ex.study_settings,
        optimum=ex.optimum,
        true_curves={true_curves},
    )
"""


def experiment_of(configurations, *, name="illustrative_example", optimum=None):
    """The module of a test experiment on the illustrative example, with
    `configurations`, the source of a dict of configurations that may use `cli`,
    `ex` (the illustrative example), `hb`, `np`, `os`, `time` and `warnings`."""
    return f"""
import os
import time
import warnings

import numpy as np

import cli
import illustrative_example as ex
import hybrid_bo as hb

{configurations}

if __name__ == "__main__":
    cli.main(
        {name!r},
        ex.problem,
        ex.measure,
        configurations,
        ex.study_settings,
        optimum={optimum!r},
    )
"""


# A fixture is a function marked `@pytest.fixture`: a test with an argument of the
# same name gets its return value. `tmp_path` and `monkeypatch` are pytest's own
# fixtures: a fresh temporary folder, and setting attributes and environment
# variables for this test only
@pytest.fixture
def experiment_folder(tmp_path):
    """The folder of the test's experiment modules, which tasks run from."""
    folder = tmp_path / "experiment"
    folder.mkdir()
    return folder


@pytest.fixture
def experiment(experiment_folder, monkeypatch):
    """A function `experiment(argv, module=small())` that writes the experiment
    module `module` (source text) to `experiment_folder` and runs it with the command
    line `argv`, in this process, as `python -m` would, with the task processes it
    starts able to import it."""
    # The module importable here and in tasks, and no bytecode, which could be older
    # than a module rewritten in the same second
    monkeypatch.syspath_prepend(str(experiment_folder))
    monkeypatch.setenv("PYTHONPATH", str(EXPERIMENTS_DIR))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def run(argv, module=None, name="experiment_module"):
        """Writes the module `name` and runs it with `argv`."""
        (experiment_folder / f"{name}.py").write_text(
            small() if module is None else module
        )
        importlib.invalidate_caches()
        monkeypatch.setattr(sys, "argv", [name, *argv])
        runpy.run_module(name, run_name="__main__", alter_sys=True)

    return run


def results(tmp_path, *argv):
    """`argv` with the results root `tmp_path`."""
    return [*argv, "--results-root", str(tmp_path)]


# Running a study


def test_run_then_plot_the_newest_study(tmp_path, experiment):
    """Running saves a study of the chosen methods, seeded like the experiment; `plot`
    then finds the newest study and saves its figures under `figures/` from the saved
    study alone."""
    # Run two methods, two runs each, into a temporary results root
    experiment(
        results(tmp_path, "--method", "standard_bo", "--method", "uniform_sampling")
        + ["--n-runs", "2", "--seed", "0", "--n-workers", "4"]
    )

    # One study, two runs per method, run i starting from the seed's ith child, with
    # the stated optimum
    (study_dir,) = (tmp_path / "illustrative_example").iterdir()
    study = load(study_dir)
    assert study.name == "illustrative_example"
    assert study.artifact["oracle_optimum"] == repr(ex.optimum)
    runs = [run for run in study.runs if run.method == "standard_bo"]
    assert len(runs) == 2
    assert runs[0].artifact["seed"] == "0"
    n_initial = runs[1].history.n_initial
    rng = np.random.default_rng(0).spawn(2)[1]
    expected = initial_points(ex.system.u, n_initial, rng)
    np.testing.assert_array_equal(runs[1].history.u[:n_initial], expected)
    assert sorted(path.name for path in (study_dir / "runs").iterdir()) == [
        "standard_bo",
        "uniform_sampling",
    ]

    # The summary, one row per method, saved with the study
    summary = pl.read_csv(study_dir / "summary.csv")
    assert summary.equals(summary_of(study))

    # Plot the newest study, without its summary as a study saved before summaries
    (study_dir / "summary.csv").unlink()
    experiment(["plot", *results(tmp_path)], small(true_curves=True))

    # The summary is back; the figures are convergence, regret and every iteration of
    # run 0 of each method (the default of `--plot-runs`)
    assert pl.read_csv(study_dir / "summary.csv").equals(summary)
    figures = {
        path.relative_to(study_dir / "figures").as_posix()
        for path in (study_dir / "figures").rglob("*.png")
    }
    assert figures == {
        "convergence.png",
        "regret.png",
        "standard_bo/run-000/iteration-00.png",
        "uniform_sampling/run-000/iteration-00.png",
    }


def test_a_study_run_as_tasks_is_the_serial_study(tmp_path, experiment):
    """The study the CLI saves, its runs run as tasks in separate processes, four at a
    time, equals `hb.study` run in this process with the same seed, field by field
    and in the same order, whatever order the tasks finish in."""
    # Every method, two runs of one iteration
    experiment(
        results(tmp_path, "--n-runs", "2", "--seed", "0", "--n-workers", "4")
        + ["--name", "tasks"]
    )

    serial = hb.study(
        small_configurations(),
        ex.problem,
        ex.measure,
        StudySettings(n_runs=2, seed=0),
        name="illustrative_example",
        optimum=ex.optimum,
    )
    assert_same(load(tmp_path / "illustrative_example" / "tasks"), serial)


def iteration_figures(study_dir):
    """The runs with iteration figures in the study's `figures/` folder, as
    `<method>/run-<index>` paths."""
    return {
        path.parent.relative_to(study_dir / "figures").as_posix()
        for path in (study_dir / "figures").rglob("iteration-*.png")
    }


def test_plot_runs_chooses_the_runs_with_iteration_figures(tmp_path, experiment):
    """`--plot-runs` takes run indices or `all`. On the run command it implies
    `--plot`; on `plot` it draws the runs given. Convergence is always drawn."""
    # Run three runs, plotting runs 1 and 2 without `--plot`
    experiment(
        results(tmp_path, "--method", "uniform_sampling", "--n-runs", "3")
        + ["--name", "demo", "--plot-runs", "1", "2", "--n-workers", "3"]
    )
    study_dir = tmp_path / "illustrative_example" / "demo"
    assert (study_dir / "figures" / "convergence.png").exists()
    assert iteration_figures(study_dir) == {
        "uniform_sampling/run-001",
        "uniform_sampling/run-002",
    }

    # Plot every run
    experiment(["plot", "demo", "--plot-runs", "all", *results(tmp_path)])
    assert iteration_figures(study_dir) == {
        f"uniform_sampling/run-{index:03d}" for index in range(3)
    }


# `parametrize` runs the test once per value in the list, each as its own test case
@pytest.mark.parametrize("plot_runs", [["all", "1"], ["-1"], ["one"]])
def test_plot_runs_takes_indices_or_all(tmp_path, experiment, plot_runs):
    """`--plot-runs` takes run indices (from 0) or `all` alone; anything else is a
    usage error."""
    # A usage error makes argparse exit, which `pytest.raises` catches; `exit_info`
    # holds the exit code
    with pytest.raises(SystemExit) as exit_info:
        experiment(["plot", "--plot-runs", *plot_runs, *results(tmp_path)])
    assert exit_info.value.code != 0


def test_plot_takes_the_finished_study_saved_last(tmp_path, experiment):
    """Without a study id, `plot` draws the finished study saved last, even when an
    older one has a name that sorts after it, and skips an unfinished one."""
    # Two studies, "zeta" first, then "alpha", then an unfinished one
    command = results(tmp_path, "--method", "uniform_sampling", "--n-runs", "1")
    experiment([*command, "--name", "zeta"])
    experiment([*command, "--name", "alpha"])
    experiment(["prepare", *command, "--name", "unfinished"])

    experiment(["plot", *results(tmp_path)])

    experiment_dir = tmp_path / "illustrative_example"
    assert (experiment_dir / "alpha" / "figures").exists()
    assert not (experiment_dir / "zeta" / "figures").exists()
    assert not (experiment_dir / "unfinished" / "figures").exists()


def test_plot_refuses_an_unfinished_study(tmp_path, experiment, capsys):
    """Plotting a study that didn't finish is a usage error that says to resume it,
    since a partial study can't be loaded.

    `capsys` is a pytest fixture that captures what the test prints.
    """
    experiment(["prepare", *results(tmp_path, "--n-runs", "1", "--name", "unfinished")])
    capsys.readouterr()

    with pytest.raises(SystemExit) as exit_info:
        experiment(["plot", "unfinished", *results(tmp_path)])
    assert exit_info.value.code != 0
    assert "resume it" in capsys.readouterr().err


def test_a_relative_results_root_is_relative_to_where_pixi_started(
    tmp_path, experiment, monkeypatch
):
    """Pixi runs the experiment from `experiments/` and keeps the folder it was started
    from in `INIT_CWD`; a relative `--results-root` is relative to that folder, for
    the tasks too."""
    # Pixi started in the temporary folder
    monkeypatch.setenv("INIT_CWD", str(tmp_path))

    experiment(
        ["--method", "uniform_sampling", "--n-runs", "1"]
        + ["--name", "here", "--results-root", "out"]
    )

    # Saved under the temporary folder, not experiments/
    assert (
        tmp_path / "out" / "illustrative_example" / "here" / "artifact.json"
    ).exists()


def test_help_names_the_configurations_and_the_other_commands(experiment, capsys):
    """`--help` names every configuration of the experiment, the other commands and
    how to plot."""
    # argparse exits after printing the help, which `pytest.raises` catches
    with pytest.raises(SystemExit):
        experiment(["--help"])

    printed = " ".join(capsys.readouterr().out.split())
    assert ", ".join(ex.configurations) in printed
    assert "prepare | run-task | resume | plot" in printed
    assert "pixi run plot illustrative_example" in printed


def test_a_tree_that_is_not_clean_warns_but_still_prepares(
    tmp_path, experiment, monkeypatch, capsys
):
    """Preparing from a tree with uncommitted or untracked files warns that the
    study won't be reproducible, then prepares it anyway."""
    # The code version git would give with an untracked file
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234-untracked")

    experiment(["prepare", *results(tmp_path, "--n-runs", "1", "--name", "dirty")])

    assert "abc1234-untracked" in capsys.readouterr().err
    task_list = json.loads(
        (tmp_path / "illustrative_example" / "dirty" / "tasks.json").read_text()
    )
    assert task_list["code_version"] == "abc1234-untracked"


def test_require_clean_refuses_a_tree_that_is_not_clean(
    tmp_path, experiment, monkeypatch
):
    """`--require-clean` refuses to run from a tree that isn't clean, saving nothing,
    and prepares from a clean one."""
    command = results(tmp_path, "--n-runs", "1", "--require-clean")

    # A changed file: a usage error, before anything runs
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234-dirty")
    with pytest.raises(SystemExit) as exit_info:
        experiment(command)
    assert exit_info.value.code != 0
    assert not (tmp_path / "illustrative_example").exists()

    # A clean tree: the study is prepared
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234")
    experiment(["prepare", *command])
    assert (tmp_path / "illustrative_example").exists()


def test_run_rejects_a_method_the_experiment_lacks(tmp_path, experiment):
    """An unknown method name is a usage error, before anything runs."""
    with pytest.raises(SystemExit):
        experiment(results(tmp_path, "--method", "nope"))
    assert not (tmp_path / "illustrative_example").exists()


def test_run_has_no_n_iterations_flag(tmp_path, experiment):
    """Each configuration has its own budget, so there is no study-wide number of
    iterations to override: `--n-iterations` is a usage error, before anything
    runs."""
    with pytest.raises(SystemExit):
        experiment(results(tmp_path, "--n-iterations", "1"))
    assert not (tmp_path / "illustrative_example").exists()


def test_plot_without_a_study_is_a_usage_error(tmp_path, experiment):
    """Plotting an experiment that has no saved study is a usage error."""
    with pytest.raises(SystemExit):
        experiment(["plot", *results(tmp_path)])


def test_running_needs_the_experiment_run_as_a_module(tmp_path):
    """Tasks are run as `python -m <experiment>`, so running from a call of `main`
    that isn't the experiment run as `__main__` is a usage error, before anything
    runs."""
    with pytest.raises(SystemExit):
        cli.main(
            "illustrative_example",
            ex.problem,
            ex.measure,
            small_configurations(),
            ex.study_settings,
            argv=results(tmp_path, "--n-runs", "1"),
        )
    assert not (tmp_path / "illustrative_example").exists()


def test_the_experiment_runs_as_a_module_from_the_experiments_folder():
    """`pixi run experiment illustrative_example --help` is `python -m
    illustrative_example --help` in `experiments/`: the script's own `cli.main` call
    answers it."""
    result = subprocess.run(
        [sys.executable, "-m", "illustrative_example", "--help"],
        cwd=EXPERIMENTS_DIR,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "pixi run experiment illustrative_example" in result.stdout


# Logs


def test_the_study_log_records_each_run_and_iteration(tmp_path, experiment):
    """A study saves its study log, `study.log`, in its folder: each configuration,
    each task as it starts and finishes with a count, and every line of each run,
    tagged with the run: each iteration with its time and the incumbent at INFO, and
    the time of each GP fit and multistart at DEBUG."""
    # One method, two runs of two iterations
    experiment(
        results(
            tmp_path, "--method", "standard_bo", "--n-runs", "2", "--name", "logged"
        ),
        small(n_iterations=2),
    )
    log = (tmp_path / "illustrative_example" / "logged" / "study.log").read_text()

    # The launcher's lines: the configuration and each task
    assert "Configuration standard_bo: 2 runs of 4 evaluations" in log
    assert "Started task 0 (standard_bo/run-000)" in log
    assert "Finished task 1 (standard_bo/run-001): 2 of 2 done" in log

    # INFO: each run, the initial points and each iteration
    assert "Run 1 of 2 finished in" in log
    assert "Run 2 of 2 finished in" in log
    assert "standard_bo: evaluated 2 initial points in" in log
    assert log.count("standard_bo iteration 1 of 2:") == 2
    assert log.count("standard_bo iteration 2 of 2:") == 2
    assert "incumbent" in log

    # DEBUG: the GP fits and the multistarts, named by their NLP
    assert "DEBUG hybrid_bo.gp: Fitted the GP in" in log
    assert "DEBUG hybrid_bo.solvers: negative_log_marginal_likelihood:" in log


@pytest.mark.parametrize("n_workers", ["1", "2"])
def test_each_run_logs_to_its_own_log_and_tags_the_study_log(
    tmp_path, experiment, n_workers
):
    """Every line a run logs goes to the run's per-run log,
    `logs/<label>/run-<index>.log`, in the study's folder, and to the study log, in
    task order, starting with its label and index (`standard_bo/run-001`). The
    launcher's lines are in the study log only, untagged."""
    # One method, two runs of two iterations, one or two at a time
    experiment(
        results(
            tmp_path, "--method", "standard_bo", "--n-runs", "2", "--name", "tagged"
        )
        + ["--n-workers", n_workers],
        small(n_iterations=2),
    )

    # The study log: each run's lines tagged, run 0's before run 1's; the launcher's
    # not
    study_dir = tmp_path / "illustrative_example" / "tagged"
    lines = (study_dir / "study.log").read_text().splitlines()
    for index in (0, 1):
        tagged = [
            line for line in lines if line.startswith(f"standard_bo/run-00{index} ")
        ]
        assert any(f"Run {index + 1} of 2 finished in" in line for line in tagged)
        assert sum("standard_bo iteration 2 of 2:" in line for line in tagged) == 1
        assert any("DEBUG hybrid_bo.gp: Fitted the GP in" in line for line in tagged)
    first_of_run_1 = next(
        i for i, line in enumerate(lines) if line.startswith("standard_bo/run-001 ")
    )
    assert not any(
        line.startswith("standard_bo/run-000 ") for line in lines[first_of_run_1:]
    )
    start = [line for line in lines if "Configuration standard_bo:" in line]
    assert len(start) == 1
    assert not start[0].startswith("standard_bo/")

    # Each run's per-run log holds its lines and no other run's
    for index in (0, 1):
        run_log = (
            study_dir / "logs" / "standard_bo" / f"run-00{index}.log"
        ).read_text()
        assert f"Run {index + 1} of 2 finished in" in run_log
        assert f"Run {2 - index} of 2" not in run_log
        assert run_log.count("standard_bo iteration 2 of 2:") == 1
        assert "Configuration standard_bo" not in run_log


WARNING_SAMPLING = """
def warning_sampling(*args, **kwargs):
    warnings.warn("a warning from the method", UserWarning, stacklevel=2)
    return hb.uniform_sampling(*args, **kwargs)

configurations = {
    "warning_sampling": hb.Configuration(
        method=warning_sampling, n_evaluations=3, n_initial=2
    )
}
"""


@pytest.mark.parametrize("n_workers", ["1", "2"])
def test_warnings_reach_the_study_log(tmp_path, experiment, n_workers):
    """A warning raised during a run (`warnings.warn`) is in the run's log and in the
    study log, tagged with the run, one task at a time or two."""
    # A study of a method that warns
    experiment(
        results(
            tmp_path, "--n-runs", "2", "--n-workers", n_workers, "--name", "warned"
        ),
        experiment_of(WARNING_SAMPLING),
    )

    # The warning, logged through `py.warnings`
    study_dir = tmp_path / "illustrative_example" / "warned"
    log = (study_dir / "study.log").read_text()
    assert log.count("a warning from the method") == 2
    assert any(
        line.startswith("warning_sampling/run-001 ") and "WARNING py.warnings:" in line
        for line in log.splitlines()
    )
    run_log = (study_dir / "logs" / "warning_sampling" / "run-000.log").read_text()
    assert "a warning from the method" in run_log


def test_running_twice_in_one_process_logs_each_record_once(tmp_path, experiment):
    """`main` puts logging back as it was when it returns, so a second study in the
    same process logs each record once, to its own log only."""
    # The root logger's handlers before, to compare after; one quick method
    root = logging.getLogger()
    handlers = list(root.handlers)
    command = results(tmp_path, "--method", "uniform_sampling", "--n-runs", "1")

    # Two studies in a row
    experiment([*command, "--name", "first"])
    experiment([*command, "--name", "second"])

    # Each record once in each log, and the handlers as they were
    experiment_dir = tmp_path / "illustrative_example"
    first = (experiment_dir / "first" / "study.log").read_text()
    second = (experiment_dir / "second" / "study.log").read_text()
    assert first.count("Configuration uniform_sampling") == 1
    assert second.count("Configuration uniform_sampling") == 1
    assert "second" not in first
    assert root.handlers == handlers


# Study ids


def test_an_unfinished_study_keeps_its_id_taken(tmp_path):
    """A study's folder reserves its id, so a study that didn't finish keeps it: a
    new study doesn't reuse it, and the message says it is unfinished."""
    (tmp_path / "old").mkdir()

    assert cli.reserve(tmp_path, "old", suffix_if_taken=True).name == "old-2"
    with pytest.raises(FileExistsError, match="unfinished study"):
        cli.reserve(tmp_path, "old", suffix_if_taken=False)


def test_a_taken_name_is_refused_before_running(tmp_path, experiment, capsys):
    """`--name` names the study's folder. A name taken by a saved study, or by a study
    that didn't finish, is a usage error before anything runs, and the message says
    which holds it."""
    # One quick method into a temporary results root
    command = results(tmp_path, "--method", "uniform_sampling", "--n-runs", "1")
    experiment_dir = tmp_path / "illustrative_example"

    # Taken by a saved study
    experiment([*command, "--name", "demo"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:
        experiment([*command, "--name", "demo"])
    assert exit_info.value.code != 0
    assert "taken by the saved study" in capsys.readouterr().err

    # Taken by a study that didn't finish
    experiment(["prepare", *command, "--name", "crashed"])
    task_list = (experiment_dir / "crashed" / "tasks.json").read_text()
    with pytest.raises(SystemExit):
        experiment([*command, "--name", "crashed"])
    assert "taken by the unfinished study" in capsys.readouterr().err

    # Nothing else was saved, and the unfinished study is untouched
    assert sorted(path.name for path in experiment_dir.iterdir()) == ["crashed", "demo"]
    assert (experiment_dir / "crashed" / "tasks.json").read_text() == task_list


def test_a_taken_default_id_gets_a_suffix(tmp_path, experiment, monkeypatch):
    """Without `--name` the study id is the UTC time at the start; two studies
    started in the same second get `<time>` and `<time>-2`."""

    # A clock stopped at one second
    class StoppedClock:
        """Stands in for `datetime`: `now` is always the same time."""

        @staticmethod
        def now(tz):
            """The stopped time."""
            return datetime(2026, 10, 20, 13, 24, 7, tzinfo=tz)

    monkeypatch.setattr(cli, "datetime", StoppedClock)

    # Two studies without a name, in the same second
    command = results(tmp_path, "--method", "uniform_sampling", "--n-runs", "1")
    experiment(["prepare", *command])
    experiment(["prepare", *command])

    # The second got the suffix
    experiment_dir = tmp_path / "illustrative_example"
    assert sorted(path.name for path in experiment_dir.iterdir()) == [
        "20261020-132407",
        "20261020-132407-2",
    ]


# Plots


def test_plot_flag_plots_straight_after_the_run(tmp_path, experiment):
    """`--plot` saves the figures straight after the run."""
    experiment(
        results(tmp_path, "--method", "uniform_sampling", "--n-runs", "1")
        + ["--name", "demo", "--plot"]
    )

    study_dir = tmp_path / "illustrative_example" / "demo"
    assert (study_dir / "figures" / "convergence.png").exists()


SWEEP = """
import functools

configurations = {
    f"uniform_sampling_{n_initial}": hb.Configuration(
        method=hb.uniform_sampling, n_evaluations=4, n_initial=n_initial
    )
    for n_initial in (1, 2)
}

if __name__ == "__main__":
    cli.main(
        "illustrative_example",
        ex.problem,
        ex.measure,
        configurations,
        ex.study_settings,
        optimum=ex.optimum,
        figures={
            "budget": hb.plots.budget_heatmap,
            "at_budget_3": functools.partial(hb.plots.regret_at_budget, budget=3),
            "at_budget_9": functools.partial(hb.plots.regret_at_budget, budget=9),
        },
    )
"""


def test_an_experiment_chooses_the_study_figures(tmp_path, experiment, capsys):
    """An experiment that passes `figures` gets those study figures, saved under
    their names, instead of convergence and regret; one that can't be drawn is
    skipped with a message."""
    # A sweep of uniform sampling over 1 and 2 initial points, 4 evaluations, two runs,
    # plotted with the experiment's figures
    experiment(
        results(tmp_path, "--n-runs", "2", "--name", "sweep", "--plot")
        + ["--n-workers", "4"],
        "import cli\nimport illustrative_example as ex\nimport hybrid_bo as hb\n"
        + SWEEP,
    )

    # No convergence or regret, and no figure at a budget the runs don't reach
    figures_dir = tmp_path / "illustrative_example" / "sweep" / "figures"
    assert {path.name for path in figures_dir.glob("*.png")} == {
        "budget.png",
        "at_budget_3.png",
    }
    assert "no at_budget_9 figure" in capsys.readouterr().out


TWO_STATES = """
import functools

import casadi as cas

import cli
import illustrative_example as ex
import hybrid_bo as hb

# Two decision variables, no known equations, two measured unknown states y1 = u1
# and y2 = u2, and f = y1^2 + y2^2
u1, u2 = hb.Var("u1", lower=-1.0, upper=1.0), hb.Var("u2", lower=-1.0, upper=1.0)
y1, y2 = hb.Var("y1", start=(-1.0, 1.0)), hb.Var("y2", start=(-1.0, 1.0))
system = hb.System(u=[u1, u2], y=[y1, y2], h_inputs=[u1, u2], measured=[y1, y2])
problem = hb.Problem(system, f=lambda v: v.y1**2 + v.y2**2)
measure = hb.oracle_measure(system, lambda v: [v.u1, v.u2], noise_std=0.0)
settings = hb.HybridBOSettings(
    gp=ex.gp,
    acquisition=hb.EI(),
    approximation=hb.SAA(n_scenarios=5, unknown_states_as_variables=False),
)
configurations = {
    "hybrid_bo": hb.Configuration(
        method=functools.partial(hb.hybrid_bo, settings=settings),
        n_evaluations=3,
        n_initial=2,
    )
}

if __name__ == "__main__":
    cli.main(
        "two_states",
        problem,
        measure,
        configurations,
        hb.StudySettings(n_runs=1, seed=0),
        optimum=0.0,
        true_curves=True,
    )
"""


def test_hybrid_bo_with_two_unknown_states_gets_iteration_figures_without_gps(
    tmp_path, experiment, capsys
):
    """The iteration figure draws hybrid BO's GP panels for one unknown state only, so
    with two the CLI draws its figures without them rather than skipping the run.
    The oracle is a lambda, which tasks rebuild from the module."""
    # Run one iteration and plot it with the oracle
    experiment(results(tmp_path, "--name", "demo", "--plot"), TWO_STATES)

    # The run has its iteration figure, and nothing was skipped
    study_dir = tmp_path / "two_states" / "demo"
    assert iteration_figures(study_dir) == {"hybrid_bo/run-000"}
    assert "no iteration figures" not in capsys.readouterr().out


# Tasks, failures and resuming


def test_a_prepared_study_resumes_to_the_study_run_in_one_go(
    tmp_path, experiment, experiment_folder, capsys
):
    """`prepare` writes the task list and runs nothing; `run-task` runs one task and
    saves its run, and does nothing for a task that is done; `resume` runs the tasks
    left and assembles the study, which equals the study run in one go; naming its
    methods in another order changes nothing."""
    # Prepare: the task list, the tasks in order, and how to run them
    command = results(
        tmp_path, "--method", "standard_bo", "--method", "uniform_sampling"
    )
    command += ["--n-runs", "2", "--seed", "0"]
    experiment(["prepare", *command, "--name", "prepared"])
    study_dir = tmp_path / "illustrative_example" / "prepared"
    task_list = json.loads((study_dir / "tasks.json").read_text())
    assert task_list["tasks"] == [
        ["standard_bo", 0],
        ["standard_bo", 1],
        ["uniform_sampling", 0],
        ["uniform_sampling", 1],
    ]
    assert "run-task prepared --task-index K" in capsys.readouterr().out
    assert not (study_dir / "runs").exists()

    # Two tasks, as a scheduler runs them; the second time, task 1 is done
    task = [sys.executable, "-m", "experiment_module", "run-task", "prepared"]
    for index in ("1", "2", "1"):
        subprocess.run(
            [*task, "--task-index", index, *results(tmp_path)],
            cwd=experiment_folder,
            check=True,
            capture_output=True,
        )
    run_file = study_dir / "runs" / "standard_bo" / "run-001.json"
    saved = run_file.read_text()
    assert sorted(path.name for path in (study_dir / "runs").rglob("*.json")) == [
        "run-000.json",
        "run-001.json",
    ]

    # Act: resume the rest, its methods named in another order, then the same study
    # in one go
    reordered = ["--method", "uniform_sampling", "--method", "standard_bo"]
    experiment(
        ["resume", "prepared", *reordered, "--n-workers", "2", *results(tmp_path)]
    )
    experiment([*command, "--name", "in-one-go", "--n-workers", "4"])

    assert run_file.read_text() == saved
    assert_same(load(study_dir), load(tmp_path / "illustrative_example" / "in-one-go"))


def test_resume_refuses_a_finished_study(tmp_path, experiment, capsys):
    """A finished study has nothing to resume: `resume` is a usage error and leaves
    it as it was."""
    experiment(
        results(
            tmp_path, "--method", "uniform_sampling", "--n-runs", "1", "--name", "done"
        )
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as exit_info:
        experiment(["resume", "done", *results(tmp_path)])
    assert exit_info.value.code != 0
    assert "is finished" in capsys.readouterr().err


def test_resume_refuses_a_study_of_another_experiment(tmp_path, experiment, capsys):
    """`resume` refuses, naming what differs, when the experiment's configurations
    or the flags given differ from the study's task list, so a resumed study never
    mixes runs of two experiments; nothing runs then."""
    # A study of one iteration, prepared
    experiment(
        ["prepare", *results(tmp_path, "--n-runs", "1", "--seed", "0", "--name", "s")]
    )
    study_dir = tmp_path / "illustrative_example" / "s"
    capsys.readouterr()

    # The experiment changed: two iterations
    with pytest.raises(SystemExit) as exit_info:
        experiment(["resume", "s", *results(tmp_path)], small(n_iterations=2))
    assert exit_info.value.code != 0
    assert "differs from the experiment in configurations" in capsys.readouterr().err

    # Another seed given
    with pytest.raises(SystemExit):
        experiment(["resume", "s", "--seed", "1", *results(tmp_path)])
    assert "differs from the experiment in seed" in capsys.readouterr().err
    assert not (study_dir / "runs").exists()


def test_run_task_refuses_another_code_version(
    tmp_path, experiment, experiment_folder, monkeypatch
):
    """A task refuses to run (exit status 2) unless the code version is the one the
    study was prepared with, so every run of a study comes from the same code."""
    # Prepared at another code version
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234")
    experiment(["prepare", *results(tmp_path, "--n-runs", "1", "--name", "old")])

    task = subprocess.run(
        [sys.executable, "-m", "experiment_module", "run-task", "old"]
        + ["--task-index", "0", *results(tmp_path)],
        cwd=experiment_folder,
        capture_output=True,
        text=True,
        check=False,
    )

    assert task.returncode == 2
    assert "code_version" in task.stderr


def failing(*args, **kwargs):
    """Fails before evaluating anything, as the method below does."""
    raise hb.HybridBOError("failed before evaluating anything")


FAILING_BEFORE_EVALUATING = """
def failing(*args, **kwargs):
    raise hb.HybridBOError("failed before evaluating anything")

configurations = {
    "failing": hb.Configuration(method=failing, n_evaluations=3, n_initial=2),
    "uniform": hb.Configuration(method=hb.uniform_sampling, n_evaluations=3, n_initial=2),
}
"""


def test_a_run_that_failed_before_evaluating_is_done(tmp_path, experiment):
    """A run that raises a `HybridBOError` with no partial run leaves a failure
    marker, so its task is done, and the study is the serial study, which drops the
    run."""
    experiment(
        results(tmp_path, "--n-runs", "2", "--seed", "0", "--name", "failed")
        + ["--n-workers", "4"],
        experiment_of(FAILING_BEFORE_EVALUATING),
    )

    # The failure markers, and the study without the failed runs
    study_dir = tmp_path / "illustrative_example" / "failed"
    assert (study_dir / "runs" / "failing" / "run-001.failed").exists()
    serial = hb.study(
        {
            "failing": Configuration(method=failing, n_evaluations=3, n_initial=2),
            "uniform": Configuration(
                method=hb.uniform_sampling, n_evaluations=3, n_initial=2
            ),
        },
        ex.problem,
        ex.measure,
        StudySettings(n_runs=2, seed=0),
        name="illustrative_example",
    )
    assert_same(load(study_dir), serial)
    assert "failing: no runs" in serial.artifact["methods"]


FAILING_TASK = """
def failing(*args, **kwargs):
    if os.environ.get("FIXED") is None:
        raise RuntimeError("boom")
    return hb.uniform_sampling(*args, **kwargs)

configurations = {
    "quick": hb.Configuration(method=hb.uniform_sampling, n_evaluations=2, n_initial=1),
    "failing": hb.Configuration(method=failing, n_evaluations=2, n_initial=1),
}
"""


def test_a_task_that_fails_stops_the_study_and_leaves_it_resumable(
    tmp_path, experiment, monkeypatch, capsys
):
    """A run that raises an unexpected exception stops the study with exit status 1,
    naming the task and its per-run log, which holds the traceback. The finished runs
    are kept, and once the bug is fixed `resume` finishes the study."""
    # Act: a study whose last task fails
    with pytest.raises(SystemExit) as exit_info:
        experiment(
            results(tmp_path, "--n-runs", "1", "--name", "broken"),
            experiment_of(FAILING_TASK),
        )

    # Stopped, the failing task named, its traceback in its log, the other run kept
    assert exit_info.value.code == 1
    study_dir = tmp_path / "illustrative_example" / "broken"
    run_log = study_dir / "logs" / "failing" / "run-000.log"
    printed = capsys.readouterr().err
    assert (
        f"Task 1 (failing/run-000) failed with exit status 1; its log is {run_log}"
        in printed
    )
    assert "resume broken" in printed
    assert "RuntimeError: boom" in run_log.read_text()
    assert (study_dir / "runs" / "quick" / "run-000.json").exists()
    assert not (study_dir / "artifact.json").exists()

    # The bug fixed, resume finishes the study
    monkeypatch.setenv("FIXED", "1")
    experiment(["resume", "broken", *results(tmp_path)], experiment_of(FAILING_TASK))
    assert [run.method for run in load(study_dir).runs] == ["quick", "failing"]


NESTED_MEASUREMENT = """
def measurement():
    def measure(u, rng):
        \"\"\"y = sin(u), from a nested function.\"\"\"
        return np.sin(u)

    return measure

configurations = {
    "uniform": hb.Configuration(method=lambda *args, **kwargs: hb.uniform_sampling(*args, **kwargs), n_evaluations=2, n_initial=1)
}

if __name__ == "__main__":
    cli.main(
        "nested", ex.problem, measurement(), configurations, ex.study_settings
    )
"""


def test_workers_take_nested_functions_and_lambdas(tmp_path, experiment):
    """Each task rebuilds the experiment from its module, so nothing is pickled: a
    measurement that is a nested function and a method that is a lambda run with
    two workers."""
    experiment(
        results(tmp_path, "--n-runs", "2", "--n-workers", "2", "--name", "nested"),
        "import numpy as np\nimport cli\nimport illustrative_example as ex\n"
        "import hybrid_bo as hb\n" + NESTED_MEASUREMENT,
    )

    study = load(tmp_path / "nested" / "nested")
    assert len(study.runs) == 2
    assert study.artifact["measure"] == "y = sin(u), from a nested function."


THREAD_REPORTING = """
def thread_reporting(*args, **kwargs):
    threads = {name: os.environ.get(name) for name in cli.THREAD_VARIABLES}
    warnings.warn(f"threads: {threads}", UserWarning, stacklevel=2)
    return hb.uniform_sampling(*args, **kwargs)

configurations = {
    "threads": hb.Configuration(method=thread_reporting, n_evaluations=2, n_initial=1)
}
"""


def test_workers_run_single_threaded_blas_unless_set(tmp_path, experiment, monkeypatch):
    """With several workers, tasks run BLAS and OpenMP on one thread each, so n
    workers use n cores, except for a variable the user set, which is kept; this
    process's environment stays as it was."""
    # The user set OpenMP's threads
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    for variable in cli.THREAD_VARIABLES[1:]:
        monkeypatch.delenv(variable, raising=False)
    before = dict(os.environ)

    # Act: the tasks report their threads as warnings, in their per-run logs
    experiment(
        results(tmp_path, "--n-runs", "2", "--n-workers", "2", "--name", "threads"),
        experiment_of(THREAD_REPORTING),
    )

    # Both runs had one thread, but OpenMP's three
    expected = {name: "1" for name in cli.THREAD_VARIABLES} | {"OMP_NUM_THREADS": "3"}
    logs = tmp_path / "illustrative_example" / "threads" / "logs" / "threads"
    for index in (0, 1):
        assert f"threads: {expected}" in (logs / f"run-00{index}.log").read_text()
    assert dict(os.environ) == before


# A study whose second configuration ignores Ctrl-C, as a run inside IPOPT does
# (CasADi catches the KeyboardInterrupt and counts it as a failed start); its runs
# write a file named by their process id once running
STUBBORN = """
import os
import pathlib
import time

import cli
import illustrative_example as ex
import hybrid_bo as hb


def stubborn(problem, measure, initial_points, n, rng):
    pathlib.Path(os.environ["STARTED"], str(os.getpid())).touch()
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            pass


configurations = {
    "quick": hb.Configuration(method=hb.uniform_sampling, n_evaluations=2, n_initial=1),
    "stubborn": hb.Configuration(method=stubborn, n_evaluations=2, n_initial=1),
}

if __name__ == "__main__":
    cli.main(
        "stubborn", ex.problem, ex.measure, configurations, hb.StudySettings(n_runs=2)
    )
"""


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to a process group")
@pytest.mark.parametrize("n_interrupts", [1, 2])
def test_ctrl_c_stops_a_study_at_once(tmp_path, n_interrupts):
    """Ctrl-C (SIGINT to the terminal's process group) stops a study within seconds,
    even while its runs ignore the interrupt, with exit status 130 and the command
    that resumes it. It leaves no task process behind, and the finished runs saved.
    Under pixi, which passes a Ctrl-C on to the process it runs, the study gets it
    twice.

    `skipif` skips the test where the condition holds.
    """
    # Setup: the study in its own process group, as a terminal runs a command, with
    # two tasks at once
    (tmp_path / "stubborn.py").write_text(STUBBORN)
    started = tmp_path / "started"
    started.mkdir()
    env = os.environ | {"STARTED": str(started), "PYTHONPATH": str(EXPERIMENTS_DIR)}
    study = subprocess.Popen(
        [sys.executable, "-m", "stubborn", "--n-workers", "2", "--name", "s"]
        + results(tmp_path),
        cwd=tmp_path,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    # Act: Ctrl-C once both stubborn runs are running
    deadline = time.monotonic() + 60
    while len(list(started.iterdir())) < 2 and time.monotonic() < deadline:
        time.sleep(0.2)
    assert len(list(started.iterdir())) == 2, "the stubborn runs never started"
    for _ in range(n_interrupts):
        os.killpg(study.pid, signal.SIGINT)
        time.sleep(0.01)
    try:
        _, stderr = study.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(study.pid, signal.SIGKILL)
        study.communicate()
        pytest.fail("the study didn't stop within 20s of Ctrl-C")

    # Check: stopped by the interrupt, with how to resume it, the quick runs kept and
    # no task left running
    assert study.returncode == 130
    assert "resume s" in stderr
    runs = tmp_path / "stubborn" / "s" / "runs"
    assert sorted(path.name for path in (runs / "quick").iterdir()) == [
        "run-000.json",
        "run-001.json",
    ]
    for task in started.iterdir():
        with pytest.raises(ProcessLookupError):
            os.kill(int(task.name), 0)
