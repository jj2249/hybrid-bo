"""Tests of the Benchopt benchmark in experiments/: the hash of the code that keeps its
cache honest, its plots, and one real `benchopt run`."""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from benchmark_utils import FOLDER, MODULES, fingerprint
from plots.regret import Plot as Regret
from plots.summary import Plot as Summary

import hybrid_bo as hb


def results(rows: list[tuple[str, int, int, float, float]]) -> pd.DataFrame:
    """A results table from (method, run, evaluation, regret, complete) rows."""
    columns = [
        "solver_name",
        "idx_rep",
        "objective_evaluation",
        "objective_regret",
        "objective_complete",
    ]
    table = pd.DataFrame(rows, columns=columns)
    return table.assign(dataset_name="d", objective_name="o")


def test_modules_are_the_experiments():
    """Every experiment is found, and Benchopt's own file is not taken for one."""
    assert MODULES == [
        "illustrative_example",
        "illustrative_example_2d",
        "langermann",
        "rosenbrock",
    ]


def test_fingerprint_changes_with_the_experiment(tmp_path: Path):
    """An edited experiment gets a new hash, so Benchopt's cache can't return its old
    results; the same file keeps its hash, so finished runs stay cached."""
    # tmp_path is a pytest fixture: a new, empty folder for this test
    path = tmp_path / "experiment.py"
    path.write_text("budget = 7\n")
    before = fingerprint(tmp_path)

    # Check: the same text gives the same hash, and a change gives another
    assert fingerprint(tmp_path) == before
    path.write_text("budget = 4\n")
    assert fingerprint(tmp_path) != before


def test_regret_is_the_median_over_runs_with_its_quartiles():
    """Each method's curve is its median regret at each evaluation, in a band from the
    first to the third quartile."""
    df = results(
        [
            ("a", 0, 1, 1.0, 1.0),
            ("a", 1, 1, 3.0, 1.0),
            ("a", 2, 1, 8.0, 1.0),
            ("a", 0, 2, 0.5, 1.0),
            ("a", 1, 2, 0.1, 1.0),
            ("a", 2, 2, 0.3, 1.0),
        ]
    )

    (curve,) = Regret().plot(df, "d", "o")

    assert curve["x"] == [1, 2]
    assert curve["y"] == [3.0, 0.3]
    # `pytest.approx` compares with a tolerance
    assert curve["y_low"] == pytest.approx([2.0, 0.2])
    assert curve["y_high"] == pytest.approx([5.5, 0.4])


def test_summary_counts_complete_runs():
    """A run that stopped early counts as a run but not as complete, and the quartiles
    are of each run's last evaluation."""
    df = results(
        [
            ("a", 0, 1, 9.0, 1.0),
            ("a", 0, 2, 2.0, 1.0),
            ("a", 1, 1, 4.0, 0.0),
        ]
    )

    (row,) = Summary().plot(df, "d", "o")

    assert row[:3] == ["a", 2, 1]
    assert row[4] == 3.0  # the median of 2 and 4


def test_a_benchopt_run_saves_rows_and_the_full_run(tmp_path: Path):
    """`benchopt run` gives one row per evaluation of each run, and saves each run in
    full, as a file `hb.load_run` reads."""
    # Run one fast method twice, through Benchopt's command line
    output = tmp_path / "results.parquet"
    started = time.time()
    subprocess.run(
        [
            *(sys.executable, "-m", "benchopt", "run", "."),
            *("-d", "experiment[module=illustrative_example]"),
            *("-s", "method[label=uniform_sampling]"),
            *("-r", "2", "--no-cache", "--no-html", "--no-plot"),
            *("--output", str(output)),
        ],
        cwd=FOLDER,
        check=True,
        capture_output=True,
    )

    # Check the rows: the budget is 7 evaluations
    table = pd.read_parquet(output)
    assert len(table) == 2 * 7
    assert table["objective_complete"].eq(1.0).all()

    # Check the saved runs: the rows' code names their folder, which may also hold
    # runs of earlier studies
    code = table["p_dataset_code"].iloc[0]
    folder = FOLDER / "outputs" / "runs" / "illustrative_example" / code
    saved = (folder / "uniform_sampling").glob("seed-*.json")
    runs = [path for path in saved if path.stat().st_mtime >= started - 1]
    assert len(runs) == 2
    assert hb.load_run(runs[0]).history.u.shape == (7, 1)
