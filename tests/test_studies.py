"""Studies: repeating methods, the oracle optimum, and saving and loading as JSON.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

import dataclasses
import functools
import json
import warnings

import illustrative_example as ex
import numpy as np
import pytest

import hybrid_bo as hb

SETTINGS = hb.StudySettings(n_runs=2, n_initial=2, n_iterations=1, seed=0)


def assert_same(actual, expected, where="study"):
    """`actual` equals `expected` field by field, arrays bit for bit with the same
    shape and dtype; `where` names the field in a failure."""
    if dataclasses.is_dataclass(expected):
        assert type(actual) is type(expected), where
        for field in dataclasses.fields(expected):
            assert_same(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{where}.{field.name}",
            )
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), where
        assert actual.dtype == expected.dtype, where
        np.testing.assert_array_equal(actual, expected, err_msg=where, strict=True)
    elif isinstance(expected, list | tuple):
        assert type(actual) is type(expected), where
        assert len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{where}[{i}]")
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key in expected:
            assert_same(actual[key], expected[key], f"{where}[{key!r}]")
    else:
        assert actual == expected, where


def history_of(n: int) -> hb.History:
    """A history of n zero points, two mechanistic states and no oracle values."""
    return hb.History(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 2)),
        y=np.zeros((n, 1)),
        f=np.zeros(n),
        x_oracle=None,
        y_oracle=None,
        f_oracle=None,
        n_initial=n,
    )


# `scope="module"` makes pytest compute the fixture once for the whole file, since
# running a study takes a few seconds
@pytest.fixture(scope="module")
def illustrative_study() -> hb.Study:
    """Every method of the illustrative example, 2 runs of 1 iteration from seed 0."""
    return hb.study(
        ex.methods,
        ex.problem,
        ex.experiment,
        SETTINGS,
        name="illustrative_example",
        optimum=ex.optimum,
    )


# Running a study


def test_study_gives_the_runs_of_each_method_from_spawned_generators(
    illustrative_study,
):
    """Run i of every method is the run the method gives with the ith generator
    spawned from the seed, initial points drawn from it, as the examples' CLI ran
    them before studies."""
    for name, method in ex.methods.items():
        runs = [run for run in illustrative_study.runs if run.method == name]
        assert [run.index for run in runs] == [0, 1]
        for run, rng in zip(runs, np.random.default_rng(0).spawn(2), strict=True):
            u = hb.initial_points(ex.system.u, SETTINGS.n_initial, rng)
            expected = method(ex.problem, ex.experiment, u, 1, rng=rng)
            assert_same(run.history, expected.history, f"{name} run {run.index}")


def test_run_i_of_every_method_starts_from_the_same_initial_points(
    illustrative_study,
):
    """The methods share generators, so run i of each has the same initial points."""
    for index in range(SETTINGS.n_runs):
        initial = [
            run.history.u[: SETTINGS.n_initial]
            for run in illustrative_study.runs
            if run.index == index
        ]
        assert len(initial) == len(ex.methods)
        for u in initial[1:]:
            np.testing.assert_array_equal(u, initial[0])
    run_0, run_1 = illustrative_study.runs[:2]
    assert not np.array_equal(run_0.history.u, run_1.history.u)


def test_study_records_the_problem_methods_and_stated_oracle_optimum(
    illustrative_study,
):
    """The study's record holds the problem, each method with its settings and the
    stated oracle optimum."""
    record = illustrative_study.record
    run_record = illustrative_study.runs[0].record
    for key in ("u", "x", "y", "h_inputs", "measured", "g", "f", "experiment"):
        assert record[key] == run_record[key]
    assert "hybrid_bo: hybrid_bo(" in record["methods"]
    assert "uniform_sampling: uniform_sampling(n_points=1)" in record["methods"]
    assert record["oracle_optimum"] == repr(ex.optimum)
    assert illustrative_study.name == "illustrative_example"
    assert illustrative_study.settings == SETTINGS


def test_without_a_stated_optimum_the_study_has_none(illustrative_problem):
    """The study never computes the oracle optimum, even when the experiment has an
    oracle."""
    methods = {"uniform": hb.uniform_sampling}
    settings = hb.StudySettings(n_runs=1, n_initial=1, n_iterations=1)

    study = hb.study(methods, illustrative_problem, ex.experiment, settings)

    assert "oracle_optimum" not in study.record


def test_an_optimum_beaten_by_an_evaluated_point_warns(illustrative_problem):
    """A stated optimum above an evaluated point's oracle objective can't be the
    optimum, so `study` warns; the true optimum gives no warning."""
    # Two uniform points; the true optimum is about -50.46, and f is at most about 75
    methods = {"uniform": hb.uniform_sampling}
    settings = hb.StudySettings(n_runs=1, n_initial=1, n_iterations=1, seed=0)

    # `pytest.warns` fails unless the block warns with the category, matching `match`
    with pytest.warns(hb.OracleOptimumWarning, match="below the stated"):
        hb.study(methods, illustrative_problem, ex.experiment, settings, optimum=100.0)

    # The warning turned into an error, so the test fails if it is raised
    with warnings.catch_warnings():
        warnings.simplefilter("error", hb.OracleOptimumWarning)
        hb.study(
            methods, illustrative_problem, ex.experiment, settings, optimum=ex.optimum
        )


def test_two_configurations_of_one_method_keep_their_labels(illustrative_problem):
    """Each run carries the label it was run under, so one method can appear twice."""
    methods = {
        "ei": functools.partial(hb.standard_bo, settings=hb.StandardBOSettings()),
        "lcb": functools.partial(
            hb.standard_bo, settings=hb.StandardBOSettings(acquisition=hb.LCB())
        ),
    }
    settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=1)

    study = hb.study(
        methods,
        illustrative_problem,
        ex.experiment,
        settings,
    )

    assert [run.method for run in study.runs] == ["ei", "lcb"]
    assert [run.record["method"] for run in study.runs] == ["standard_bo"] * 2


def test_repeat_keeps_a_failed_run_as_far_as_it_got(illustrative_problem, caplog):
    """A run that raises a `HybridBOError` is kept as its partial run, and the error
    is logged; the other runs go on.

    `caplog` is a pytest fixture that captures what the code logs.
    """
    # An experiment that measures NaN, so the root-finding problem fails, for the
    # second evaluation of each run
    calls = []

    def experiment(u, rng):
        """y = sin(u), but NaN for the second evaluation of every run."""
        calls.append(u)
        return np.array([np.nan]) if len(calls) % 2 == 0 else np.sin(u)

    settings = hb.StudySettings(n_runs=3, n_initial=2, n_iterations=1)

    runs = hb.repeat(
        hb.uniform_sampling,
        illustrative_problem,
        experiment,
        settings,
    )

    # Every run failed at its second initial point, after one evaluated point
    assert [run.index for run in runs] == [0, 1, 2]
    assert [len(run.history.u) for run in runs] == [1, 1, 1]
    assert all(run.method == "uniform_sampling" for run in runs)
    assert caplog.text.count("RootFindingError") == 3


def test_the_same_seed_gives_the_same_study(illustrative_problem):
    """A study is reproducible from its settings: seed None draws a seed, which the
    study's settings keep, and running again with them gives the same runs."""
    methods = {"uniform": hb.uniform_sampling}
    settings = hb.StudySettings(n_runs=2, n_initial=1, n_iterations=1)

    first = hb.study(methods, illustrative_problem, ex.experiment, settings)
    again = hb.study(methods, illustrative_problem, ex.experiment, first.settings)

    assert first.settings.seed is not None
    assert again.settings == first.settings
    assert_same(
        [run.history for run in again.runs], [run.history for run in first.runs]
    )


# The oracle optimum


def test_oracle_optimum_solves_the_problem_with_the_oracle_in_place_of_h():
    """With y = u known, min (x - 1)^2 + y subject to x = y over u in [-2, 2] is at
    u = 0.5, where f = 0.75."""
    u = hb.Var("u", -2.0, 2.0)
    x = hb.Var("x", start=(-3.0, 3.0))
    y = hb.Var("y", start=(-3.0, 3.0))
    system = hb.System(
        u=[u], x=[x], y=[y], g=lambda v: [v.x - v.y], h_inputs=[u], measured=[y]
    )
    problem = hb.Problem(system, f=lambda v: (v.x - 1) ** 2 + v.y)

    optimum = hb.oracle_optimum(problem, lambda v: [v.u], np.random.default_rng(0))

    assert optimum == pytest.approx(0.75, abs=1e-8)


def test_oracle_optimum_rejects_an_oracle_of_the_wrong_size(illustrative_problem):
    """The oracle must give one value per unknown state, as for the experiment."""
    with pytest.raises(ValueError, match="one value per unknown state"):
        hb.oracle_optimum(
            illustrative_problem,
            lambda v: [v.u, v.u],
            np.random.default_rng(0),
        )


# Saving and loading


def test_load_after_save_gives_the_study_back(illustrative_study, tmp_path):
    """Every field of every run comes back, arrays bit for bit, from one JSON file
    for the study and one per run.

    `tmp_path` is a pytest fixture: a fresh temporary folder for this test.
    """
    path = hb.save(illustrative_study, tmp_path / "study")

    loaded = hb.load(path)

    assert_same(loaded, illustrative_study)
    assert (path / "study.json").is_file()
    assert (path / "runs" / "hybrid_bo" / "run-001.json").is_file()
    run_json = json.loads((path / "runs" / "standard_bo" / "run-000.json").read_text())
    assert run_json["method"] == "standard_bo"


def test_save_and_load_keep_partial_runs_and_empty_arrays(tmp_path):
    """A run that failed at its first point (no evaluated points) round-trips too,
    with its arrays' shapes."""
    run = hb.Run(history_of(0), record={"method": "m"}, method="m")
    study = hb.Study((run,), SETTINGS, None, {})

    loaded = hb.load(hb.save(study, tmp_path / "study"))

    assert_same(loaded, study)


def test_save_refuses_an_existing_folder(illustrative_study, tmp_path):
    """Saving never overwrites: an existing folder raises `FileExistsError`."""
    with pytest.raises(FileExistsError):
        hb.save(illustrative_study, tmp_path)


def test_save_refuses_a_label_that_is_not_a_folder_name(tmp_path):
    """A method label is a folder name, so it can't contain a path separator."""
    run = hb.Run(history_of(1), record={}, method="a/b")
    study = hb.Study((run,), SETTINGS, None, {})

    with pytest.raises(ValueError, match="a/b"):
        hb.save(study, tmp_path / "study")
    assert not (tmp_path / "study").exists()
