"""The CLI for the examples (examples/main.py): running methods and plotting runs."""

import pickle

import illustrative_example as ex
import main
import numpy as np
import pytest

from hybrid_bo import initial_points


def test_run_then_plot_the_illustrative_example(tmp_path):
    """`run` pickles the chosen methods' runs, seeded like the example; `plot` then
    saves the convergence figure and one figure per method from the pickles alone.

    `tmp_path` is a pytest fixture: a fresh temporary folder for this test.
    """
    main.main(
        [
            "run",
            "illustrative_example",
            "--method",
            "standard_bo",
            "--method",
            "uniform_sampling",
            "--n-runs",
            "2",
            "--seed",
            "0",
            "--n-iterations",
            "1",
            "--results-root",
            str(tmp_path),
        ]
    )

    # One pickle per method, two runs each, run i starting from the seed's ith child
    (results_dir,) = (tmp_path / "illustrative_example").iterdir()
    runs = pickle.loads((results_dir / "standard_bo.pkl").read_bytes())
    assert len(runs) == 2
    assert runs[0].record["seed"] == "0"
    rng = np.random.default_rng(0).spawn(2)[1]
    expected = initial_points(ex.system.u, ex.n_initial, rng)
    np.testing.assert_array_equal(runs[1].history.u[: ex.n_initial], expected)
    assert sorted(path.name for path in results_dir.iterdir()) == [
        "standard_bo.pkl",
        "uniform_sampling.pkl",
    ]

    main.main(["plot", str(results_dir)])

    assert {"convergence.png", "standard_bo.png", "uniform_sampling.png"} <= {
        path.name for path in results_dir.iterdir()
    }


def test_run_rejects_a_method_the_example_lacks(tmp_path):
    """An unknown method name is a usage error, before anything runs."""
    # argparse reports usage errors by exiting, which `pytest.raises` catches
    with pytest.raises(SystemExit):
        main.main(
            ["run", "illustrative_example", "--method", "nope"]
            + ["--results-root", str(tmp_path)]
        )
    assert not (tmp_path / "illustrative_example").exists()
