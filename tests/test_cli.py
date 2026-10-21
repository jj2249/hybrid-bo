"""The command line of the experiments (experiments/cli.py, behind `pixi run experiment`
and `pixi run plot`): running studies and plotting them."""

import logging
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cli
import illustrative_example as ex
import numpy as np
import polars as pl
import pytest

from hybrid_bo import initial_points, load, summary, uniform_sampling


def summary_of(study):
    """`hb.summary(study)` as it reads back from CSV."""
    return pl.read_csv(summary(study).write_csv().encode())


EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


def run_cli(argv, true_curves=False):
    """`cli.main` as the illustrative example calls it, with the command line `argv`."""
    cli.main(
        "illustrative_example",
        ex.problem,
        ex.measure,
        ex.methods,
        ex.study_settings,
        optimum=ex.optimum,
        true_curves=true_curves,
        argv=argv,
    )


def test_run_then_plot_the_newest_study(tmp_path):
    """Running saves a study of the chosen methods, seeded like the experiment; `plot`
    then finds the newest study and saves its figures under `figures/` from the saved
    study alone.

    `tmp_path` is a pytest fixture: a fresh temporary folder for this test.
    """
    # Run two methods, two runs each, into a temporary results root
    run_cli(
        ["--method", "standard_bo", "--method", "uniform_sampling"]
        + ["--n-runs", "2", "--seed", "0", "--n-iterations", "1"]
        + ["--results-root", str(tmp_path)]
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
    n_initial = study.settings.n_initial
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
    run_cli(["plot", "--results-root", str(tmp_path)], true_curves=True)

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


def iteration_figures(study_dir):
    """The runs with iteration figures in the study's `figures/` folder, as
    `<method>/run-<index>` paths."""
    return {
        path.parent.relative_to(study_dir / "figures").as_posix()
        for path in (study_dir / "figures").rglob("iteration-*.png")
    }


def test_plot_runs_chooses_the_runs_with_iteration_figures(tmp_path):
    """`--plot-runs` takes run indices or `all`. On the run command it implies
    `--plot`; on `plot` it draws the runs given. Convergence is always drawn."""
    # Run three runs, plotting runs 1 and 2 without `--plot`
    run_cli(
        ["--method", "uniform_sampling", "--n-runs", "3", "--n-iterations", "1"]
        + ["--name", "demo", "--plot-runs", "1", "2"]
        + ["--results-root", str(tmp_path)]
    )
    study_dir = tmp_path / "illustrative_example" / "demo"
    assert (study_dir / "figures" / "convergence.png").exists()
    assert iteration_figures(study_dir) == {
        "uniform_sampling/run-001",
        "uniform_sampling/run-002",
    }

    # Plot every run
    run_cli(["plot", "demo", "--plot-runs", "all", "--results-root", str(tmp_path)])
    assert iteration_figures(study_dir) == {
        f"uniform_sampling/run-{index:03d}" for index in range(3)
    }


@pytest.mark.parametrize("plot_runs", [["all", "1"], ["-1"], ["one"]])
def test_plot_runs_takes_indices_or_all(tmp_path, plot_runs):
    """`--plot-runs` takes run indices (from 0) or `all` alone; anything else is a
    usage error."""
    with pytest.raises(SystemExit) as exit_info:
        run_cli(["plot", "--plot-runs", *plot_runs, "--results-root", str(tmp_path)])
    assert exit_info.value.code != 0


def test_plot_takes_the_study_saved_last(tmp_path):
    """Without a study id, `plot` draws the study saved last, even when an older one
    has a name that sorts after it."""
    # Two studies: "zeta" first, then "alpha"
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--results-root", str(tmp_path)]
    run_cli([*command, "--name", "zeta"])
    run_cli([*command, "--name", "alpha"])

    run_cli(["plot", "--results-root", str(tmp_path)])

    experiment_dir = tmp_path / "illustrative_example"
    assert (experiment_dir / "alpha" / "figures").exists()
    assert not (experiment_dir / "zeta" / "figures").exists()


def test_a_relative_results_root_is_relative_to_where_pixi_started(
    tmp_path, monkeypatch
):
    """Pixi runs the experiment from `experiments/` and keeps the folder it was started
    from in `INIT_CWD`; a relative `--results-root` is relative to that folder.

    `monkeypatch` is a pytest fixture that sets environment variables for this test
    only.
    """
    # Pixi started in the temporary folder
    monkeypatch.setenv("INIT_CWD", str(tmp_path))

    run_cli(
        ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
        + ["--name", "here", "--results-root", "out"]
    )

    # Saved under the temporary folder, not experiments/
    assert (
        tmp_path / "out" / "illustrative_example" / "here" / "artifact.json"
    ).exists()


def test_help_names_the_methods_and_the_plot_command(capsys):
    """`--help` names every method of the experiment and how to plot.

    `capsys` is a pytest fixture that captures what the test prints.
    """
    # argparse exits after printing the help, which `pytest.raises` catches
    with pytest.raises(SystemExit):
        run_cli(["--help"])

    printed = " ".join(capsys.readouterr().out.split())
    assert ", ".join(ex.methods) in printed
    assert "pixi run plot illustrative_example" in printed


def test_a_tree_that_is_not_clean_warns_but_still_runs(tmp_path, monkeypatch, capsys):
    """Running from a tree with uncommitted or untracked files warns that the study
    won't be reproducible, then runs and saves it anyway."""
    # The code version git would give with an untracked file
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234-untracked")

    run_cli(
        ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
        + ["--results-root", str(tmp_path)]
    )

    assert "abc1234-untracked" in capsys.readouterr().err
    assert len(list((tmp_path / "illustrative_example").iterdir())) == 1


def test_require_clean_refuses_a_tree_that_is_not_clean(tmp_path, monkeypatch):
    """`--require-clean` refuses to run from a tree that isn't clean, saving nothing,
    and runs from a clean one."""
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--require-clean", "--results-root", str(tmp_path)]

    # A changed file: a usage error, before anything runs
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234-dirty")
    with pytest.raises(SystemExit) as exit_info:
        run_cli(command)
    assert exit_info.value.code != 0
    assert not (tmp_path / "illustrative_example").exists()

    # A clean tree: the study runs and is saved
    monkeypatch.setattr(cli, "code_version", lambda path: "abc1234")
    run_cli(command)
    assert (tmp_path / "illustrative_example").exists()


def test_run_rejects_a_method_the_experiment_lacks(tmp_path):
    """An unknown method name is a usage error, before anything runs."""
    with pytest.raises(SystemExit):
        run_cli(["--method", "nope", "--results-root", str(tmp_path)])
    assert not (tmp_path / "illustrative_example").exists()


def test_plot_without_a_study_is_a_usage_error(tmp_path):
    """Plotting an experiment that has no saved study is a usage error."""
    with pytest.raises(SystemExit):
        run_cli(["plot", "--results-root", str(tmp_path)])


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


def test_the_study_log_records_each_run_and_iteration(tmp_path, capsys):
    """A study saves its study log, `study.log`, beside its artifact: each method,
    run and iteration with its time and the incumbent at INFO, and the time of each
    GP fit and multistart at DEBUG. The console shows the INFO records only."""
    # One method, two runs of two iterations
    run_cli(
        ["--method", "standard_bo", "--n-runs", "2", "--n-iterations", "2"]
        + ["--name", "logged", "--results-root", str(tmp_path)]
    )

    # The log was moved into the study's folder
    experiment_dir = tmp_path / "illustrative_example"
    assert not (experiment_dir / "logged.log").exists()
    log = (experiment_dir / "logged" / "study.log").read_text()

    # INFO: the method, each run, the initial points and each iteration
    assert "INFO hybrid_bo.studies: Method standard_bo: 2 runs" in log
    assert "Run 1 of 2 finished in" in log
    assert "Run 2 of 2 finished in" in log
    assert "standard_bo: evaluated 2 initial points in" in log
    assert log.count("standard_bo iteration 1 of 2:") == 2
    assert log.count("standard_bo iteration 2 of 2:") == 2
    assert "incumbent" in log

    # DEBUG: the GP fits and the multistarts, named by their NLP
    assert "DEBUG hybrid_bo.gp: Fitted the GP in" in log
    assert "DEBUG hybrid_bo.solvers: negative_log_marginal_likelihood:" in log

    # The console shows the iterations but not the DEBUG records
    console = capsys.readouterr().err
    assert "standard_bo iteration 2 of 2:" in console
    assert "Fitted the GP" not in console


def test_a_taken_name_is_refused_before_running(tmp_path, capsys):
    """`--name` names the study's folder. A name taken by a saved study, or by the
    log of a study that didn't finish, is a usage error before anything runs, and
    the message says which holds it."""
    # One quick method into a temporary results root
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--results-root", str(tmp_path)]
    experiment_dir = tmp_path / "illustrative_example"

    # Taken by a saved study
    run_cli([*command, "--name", "demo"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:
        run_cli([*command, "--name", "demo"])
    assert exit_info.value.code != 0
    assert "taken by the saved study" in capsys.readouterr().err

    # Taken by the log a crashed study left
    (experiment_dir / "crashed.log").write_text("partial\n")
    with pytest.raises(SystemExit):
        run_cli([*command, "--name", "crashed"])
    assert "a study that didn't finish" in capsys.readouterr().err

    # Nothing else was saved, and the crashed study's log is untouched
    assert sorted(path.name for path in experiment_dir.iterdir()) == [
        "crashed.log",
        "demo",
    ]
    assert (experiment_dir / "crashed.log").read_text() == "partial\n"


def test_a_taken_default_id_gets_a_suffix(tmp_path, monkeypatch):
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
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--results-root", str(tmp_path)]
    run_cli(command)
    run_cli(command)

    # The second got the suffix
    experiment_dir = tmp_path / "illustrative_example"
    assert sorted(path.name for path in experiment_dir.iterdir()) == [
        "20261020-132407",
        "20261020-132407-2",
    ]


def test_plot_flag_plots_straight_after_the_run(tmp_path):
    """`--plot` saves the figures straight after the run."""
    run_cli(
        ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
        + ["--name", "demo", "--plot", "--results-root", str(tmp_path)]
    )

    study_dir = tmp_path / "illustrative_example" / "demo"
    assert (study_dir / "figures" / "convergence.png").exists()


def test_warnings_reach_the_study_log(tmp_path):
    """A warning raised during the study (`warnings.warn`) is in its study log."""

    # Uniform sampling that warns first
    def warning_sampling(*args, **kwargs):
        """`hb.uniform_sampling`, after a warning."""
        warnings.warn("a warning from the method", UserWarning, stacklevel=2)
        return uniform_sampling(*args, **kwargs)

    # A study of that method alone
    cli.main(
        "illustrative_example",
        ex.problem,
        ex.measure,
        {"warning_sampling": warning_sampling},
        ex.study_settings,
        argv=["--n-runs", "1", "--n-iterations", "1", "--name", "warned"]
        + ["--results-root", str(tmp_path)],
    )

    # The warning, logged through `py.warnings`
    log = (tmp_path / "illustrative_example" / "warned" / "study.log").read_text()
    assert "WARNING py.warnings:" in log
    assert "a warning from the method" in log


def test_a_study_that_does_not_finish_leaves_its_log(tmp_path, capsys):
    """A study whose method raises an error the study doesn't catch saves nothing,
    leaves its log beside the study folders with what it recorded, and prints the
    log's path; the error propagates."""

    # A method that fails outright
    def failing(*args, **kwargs):
        """Raises an error that isn't a `HybridBOError`."""
        raise RuntimeError("boom")

    # The error comes out of `main`
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(
            "illustrative_example",
            ex.problem,
            ex.measure,
            {"failing": failing},
            ex.study_settings,
            argv=["--n-runs", "1", "--name", "broken", "--results-root", str(tmp_path)],
        )

    # Only the log, holding the study's start, with its path printed
    experiment_dir = tmp_path / "illustrative_example"
    assert [path.name for path in experiment_dir.iterdir()] == ["broken.log"]
    assert "Method failing: 1 runs" in (experiment_dir / "broken.log").read_text()
    assert str(experiment_dir / "broken.log") in capsys.readouterr().err


def test_running_twice_in_one_process_logs_each_record_once(tmp_path):
    """`main` puts logging back as it was when it returns, so a second study in the
    same process logs each record once, to its own log only."""
    # The root logger's handlers before, to compare after; one quick method
    root = logging.getLogger()
    handlers = list(root.handlers)
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--results-root", str(tmp_path)]

    # Two studies in a row
    run_cli([*command, "--name", "first"])
    run_cli([*command, "--name", "second"])

    # Each record once in each log, and the handlers as they were
    experiment_dir = tmp_path / "illustrative_example"
    first = (experiment_dir / "first" / "study.log").read_text()
    second = (experiment_dir / "second" / "study.log").read_text()
    assert first.count("Method uniform_sampling") == 1
    assert second.count("Method uniform_sampling") == 1
    assert "second" not in first
    assert root.handlers == handlers
