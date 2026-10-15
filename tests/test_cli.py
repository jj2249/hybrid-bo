"""The command line of the examples (examples/cli.py, behind `pixi run example` and
`pixi run plot`): running studies and plotting them."""

import subprocess
import sys
from pathlib import Path

import cli
import illustrative_example as ex
import numpy as np
import pytest

from hybrid_bo import initial_points, load

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def run_cli(argv, true_curves=False):
    """`cli.main` as the illustrative example calls it, with the command line `argv`."""
    cli.main(
        "illustrative_example",
        ex.problem,
        ex.experiment,
        ex.methods,
        ex.study_settings,
        optimum=ex.optimum,
        true_curves=true_curves,
        argv=argv,
    )


def test_run_then_plot_the_newest_study(tmp_path):
    """Running saves a study of the chosen methods, seeded like the example; `plot`
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
    assert study.record["oracle_optimum"] == repr(ex.optimum)
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

    # Plot the newest study: convergence, regret and every iteration of every run
    run_cli(["plot", "--results-root", str(tmp_path)], true_curves=True)

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
    example_dir = tmp_path / "illustrative_example"
    assert sorted(path.name for path in example_dir.iterdir()) == ["demo", "demo-2"]
    assert not (example_dir / "demo" / "figures").exists()
    assert (example_dir / "demo-2" / "figures" / "convergence.png").exists()


def test_plot_takes_the_study_saved_last(tmp_path):
    """Without a study id, `plot` draws the study saved last, even when an older one
    has a name that sorts after it."""
    # Two studies: "zeta" first, then "alpha"
    command = ["--method", "uniform_sampling", "--n-runs", "1", "--n-iterations", "1"]
    command += ["--results-root", str(tmp_path)]
    run_cli([*command, "--name", "zeta"])
    run_cli([*command, "--name", "alpha"])

    run_cli(["plot", "--results-root", str(tmp_path)])

    example_dir = tmp_path / "illustrative_example"
    assert (example_dir / "alpha" / "figures").exists()
    assert not (example_dir / "zeta" / "figures").exists()


def test_a_relative_results_root_is_relative_to_where_pixi_started(
    tmp_path, monkeypatch
):
    """Pixi runs the example from `examples/` and keeps the folder it was started
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

    # Saved under the temporary folder, not examples/
    assert (tmp_path / "out" / "illustrative_example" / "here" / "study.json").exists()


def test_help_names_the_methods_and_the_plot_command(capsys):
    """`--help` names every method of the example and how to plot.

    `capsys` is a pytest fixture that captures what the test prints.
    """
    # argparse exits after printing the help, which `pytest.raises` catches
    with pytest.raises(SystemExit):
        run_cli(["--help"])

    printed = " ".join(capsys.readouterr().out.split())
    assert ", ".join(ex.methods) in printed
    assert "pixi run plot illustrative_example" in printed


def test_run_rejects_a_method_the_example_lacks(tmp_path):
    """An unknown method name is a usage error, before anything runs."""
    with pytest.raises(SystemExit):
        run_cli(["--method", "nope", "--results-root", str(tmp_path)])
    assert not (tmp_path / "illustrative_example").exists()


def test_plot_without_a_study_is_a_usage_error(tmp_path):
    """Plotting an example that has no saved study is a usage error."""
    with pytest.raises(SystemExit):
        run_cli(["plot", "--results-root", str(tmp_path)])


def test_the_example_runs_as_a_module_from_the_examples_folder():
    """`pixi run example illustrative_example --help` is `python -m
    illustrative_example --help` in `examples/`: the script's own `cli.main` call
    answers it."""
    result = subprocess.run(
        [sys.executable, "-m", "illustrative_example", "--help"],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "pixi run example illustrative_example" in result.stdout
