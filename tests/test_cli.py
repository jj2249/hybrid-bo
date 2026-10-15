"""The CLI for the examples (examples/main.py, behind `pixi run example` and
`pixi run plot`): running studies and plotting them."""

import illustrative_example as ex
import main
import numpy as np
import pytest

from hybrid_bo import initial_points, load


def test_run_then_plot_the_newest_study(tmp_path):
    """`run` saves a study of the chosen methods, seeded like the example; `plot`
    then finds the newest study by the example's name and saves its figures under
    `figures/` from the saved study alone.

    `tmp_path` is a pytest fixture: a fresh temporary folder for this test.
    """
    # Run two methods, two runs each, into a temporary results root
    main.main(
        ["run", "illustrative_example", "--method", "standard_bo"]
        + ["--method", "uniform_sampling", "--n-runs", "2", "--seed", "0"]
        + ["--n-iterations", "1", "--results-root", str(tmp_path)]
    )

    # One study, two runs per method, run i starting from the seed's ith child
    (study_dir,) = (tmp_path / "illustrative_example").iterdir()
    study = load(study_dir)
    assert study.name == "illustrative_example"
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
    main.main(["plot", "illustrative_example", "--results-root", str(tmp_path)])

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
    command = ["run", "illustrative_example", "--method", "uniform_sampling"]
    command += ["--n-runs", "1", "--n-iterations", "1", "--name", "demo"]
    command += ["--results-root", str(tmp_path)]
    main.main(command)
    main.main([*command, "--plot"])

    # Two studies, only the second with figures
    example_dir = tmp_path / "illustrative_example"
    assert sorted(path.name for path in example_dir.iterdir()) == ["demo", "demo-2"]
    assert not (example_dir / "demo" / "figures").exists()
    assert (example_dir / "demo-2" / "figures" / "convergence.png").exists()


def test_help_lists_the_examples_and_their_methods(capsys):
    """`pixi run example --help` names every example and its methods.

    `capsys` is a pytest fixture that captures what the test prints.
    """
    # argparse exits after printing the help, which `pytest.raises` catches
    with pytest.raises(SystemExit):
        main.main(["run", "--help"])

    printed = capsys.readouterr().out
    assert "illustrative_example: " + ", ".join(ex.methods) in printed


def test_run_rejects_a_method_the_example_lacks(tmp_path):
    """An unknown method name is a usage error, before anything runs."""
    with pytest.raises(SystemExit):
        main.main(
            ["run", "illustrative_example", "--method", "nope"]
            + ["--results-root", str(tmp_path)]
        )
    assert not (tmp_path / "illustrative_example").exists()


def test_plot_without_a_study_is_a_usage_error(tmp_path):
    """Plotting an example that has no saved study is a usage error."""
    with pytest.raises(SystemExit):
        main.main(["plot", "illustrative_example", "--results-root", str(tmp_path)])
