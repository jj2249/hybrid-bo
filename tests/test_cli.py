"""The command line of the experiments (experiments/cli.py, behind `pixi run experiment`
and `pixi run plot`): running studies and plotting them."""

import subprocess
import sys
from pathlib import Path

import cli
import illustrative_example as ex
import numpy as np
import polars as pl
import pytest

from hybrid_bo import initial_points, load, summary


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
    assert runs[0].record["seed"] == "0"
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
    # every run
    assert pl.read_csv(study_dir / "summary.csv").equals(summary)

    figures = {
        path.relative_to(study_dir / "figures").as_posix()
        for path in (study_dir / "figures").rglob("*.png")
    }
    assert figures == {
        "convergence.png",
        "regret.png",
        "standard_bo/run-000/iteration-00.png",
        "standard_bo/run-001/iteration-00.png",
        "uniform_sampling/run-000/iteration-00.png",
        "uniform_sampling/run-001/iteration-00.png",
    }


def test_a_named_study_gets_a_suffix_when_taken_and_plot_flag_plots(tmp_path):
    """`--name` names the study's folder; a name already taken gets `-2`, and
    `--plot` saves the figures straight after the run."""
    # Two runs under the same name, the second plotted
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--name", "demo", "--results-root", str(tmp_path)]
    run_cli(command)
    run_cli([*command, "--plot"])

    # Two studies, only the second with figures
    experiment_dir = tmp_path / "illustrative_example"
    assert sorted(path.name for path in experiment_dir.iterdir()) == ["demo", "demo-2"]
    assert not (experiment_dir / "demo" / "figures").exists()
    assert (experiment_dir / "demo-2" / "figures" / "convergence.png").exists()


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
