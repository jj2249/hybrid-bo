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
        ex.measure,
        SETTINGS,
        name="illustrative_example",
        optimum=ex.optimum,
    )


# Running a study


def test_study_gives_the_runs_of_each_method_from_spawned_generators(
    illustrative_study,
):
    """Run i of every method is the run the method gives with the ith generator
    spawned from the seed, initial points drawn from it, as the experiments' CLI ran
    them before studies."""
    for name, method in ex.methods.items():
        runs = [run for run in illustrative_study.runs if run.method == name]
        assert [run.index for run in runs] == [0, 1]
        for run, rng in zip(runs, np.random.default_rng(0).spawn(2), strict=True):
            u = hb.initial_points(ex.system.u, SETTINGS.n_initial, rng)
            expected = method(ex.problem, ex.measure, u, 1, rng=rng)
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


def test_artifact_holds_the_problem_methods_and_stated_oracle_optimum(
    illustrative_study,
):
    """The study's artifact holds the problem, each method with its settings and the
    stated oracle optimum."""
    artifact = illustrative_study.artifact
    run_artifact = illustrative_study.runs[0].artifact
    for key in ("u", "x", "y", "h_inputs", "measured", "g", "f", "measure"):
        assert artifact[key] == run_artifact[key]
    assert "hybrid_bo: hybrid_bo(" in artifact["methods"]
    assert "uniform_sampling: uniform_sampling(n_points=1)" in artifact["methods"]
    assert artifact["oracle_optimum"] == repr(ex.optimum)
    assert illustrative_study.name == "illustrative_example"
    assert illustrative_study.settings == SETTINGS


def test_artifact_tells_apart_methods_that_differ_only_in_their_prior_gp(
    illustrative_problem,
):
    """The study's artifact gives each BO method's prior GP on the line after its
    settings, since the study can't be run again without it, so two configurations
    that differ only in their prior GP get different entries."""
    # Standard BO twice, the second with a known noise
    noisy_gp = dataclasses.replace(ex.gp, noise_std=0.1)
    methods = {
        "noise_free": functools.partial(
            hb.standard_bo, settings=hb.StandardBOSettings(gp=ex.gp)
        ),
        "noisy": functools.partial(
            hb.standard_bo, settings=hb.StandardBOSettings(gp=noisy_gp)
        ),
    }
    settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=1, seed=0)

    study = hb.study(methods, illustrative_problem, ex.measure, settings)

    # Each method's entry is its settings line, then its prior GP
    noise_free, noisy = study.artifact["methods"].split("\nnoisy: ")
    assert noise_free.endswith(f")\n  gp: {ex.gp!r}")
    assert noisy.endswith(f")\n  gp: {noisy_gp!r}")
    assert "noise_std=0.1" in noisy


def test_without_a_stated_optimum_the_study_has_none(illustrative_problem):
    """The study never computes the oracle optimum, even when the measurement has an
    oracle."""
    methods = {"uniform": hb.uniform_sampling}
    settings = hb.StudySettings(n_runs=1, n_initial=1, n_iterations=1)

    study = hb.study(methods, illustrative_problem, ex.measure, settings)

    assert "oracle_optimum" not in study.artifact


def test_an_optimum_beaten_by_an_evaluated_point_warns(illustrative_problem):
    """A stated optimum above an evaluated point's oracle objective can't be the
    optimum, so `study` warns; the true optimum gives no warning."""
    # Two uniform points; the true optimum is about -50.46, and f is at most about 75
    methods = {"uniform": hb.uniform_sampling}
    settings = hb.StudySettings(n_runs=1, n_initial=1, n_iterations=1, seed=0)

    # `pytest.warns` fails unless the block warns with the category, matching `match`
    with pytest.warns(hb.OracleOptimumWarning, match="below the stated"):
        hb.study(methods, illustrative_problem, ex.measure, settings, optimum=100.0)

    # The warning turned into an error, so the test fails if it is raised
    with warnings.catch_warnings():
        warnings.simplefilter("error", hb.OracleOptimumWarning)
        hb.study(
            methods, illustrative_problem, ex.measure, settings, optimum=ex.optimum
        )


def test_two_configurations_of_one_method_keep_their_labels(illustrative_problem):
    """Each run carries the label it was run under, so one method can appear twice."""
    methods = {
        "ei": functools.partial(
            hb.standard_bo, settings=hb.StandardBOSettings(gp=ex.gp)
        ),
        "lcb": functools.partial(
            hb.standard_bo,
            settings=hb.StandardBOSettings(gp=ex.gp, acquisition=hb.LCB()),
        ),
    }
    settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=1)

    study = hb.study(
        methods,
        illustrative_problem,
        ex.measure,
        settings,
    )

    assert [run.method for run in study.runs] == ["ei", "lcb"]
    assert [run.artifact["method"] for run in study.runs] == ["standard_bo"] * 2


def test_repeat_keeps_a_failed_run_as_far_as_it_got(illustrative_problem, caplog):
    """A run that raises a `HybridBOError` is kept as its partial run, and the error
    is logged; the other runs go on.

    `caplog` is a pytest fixture that captures what the code logs.
    """
    # A `measure` that returns NaN, so the root-finding problem fails, for the
    # second evaluation of each run
    calls = []

    def measure(u, rng):
        """y = sin(u), but NaN for the second evaluation of every run."""
        calls.append(u)
        return np.array([np.nan]) if len(calls) % 2 == 0 else np.sin(u)

    settings = hb.StudySettings(n_runs=3, n_initial=2, n_iterations=1)

    runs = hb.repeat(
        hb.uniform_sampling,
        illustrative_problem,
        measure,
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

    first = hb.study(methods, illustrative_problem, ex.measure, settings)
    again = hb.study(methods, illustrative_problem, ex.measure, first.settings)

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
    """The oracle must give one value per unknown state, as for the measurement."""
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
    artifact = json.loads((path / "artifact.json").read_text())
    assert artifact["artifact"] == illustrative_study.artifact
    assert (path / "runs" / "hybrid_bo" / "run-001.json").is_file()
    run_json = json.loads((path / "runs" / "standard_bo" / "run-000.json").read_text())
    assert run_json["method"] == "standard_bo"


def test_load_reads_a_study_saved_before_artifacts(illustrative_study, tmp_path):
    """A study saved before the artifact was named (`study.json` with a `record` key)
    still loads, with its record as the artifact."""
    # Save, then rewrite the artifact in the old format
    path = hb.save(illustrative_study, tmp_path / "study")
    data = json.loads((path / "artifact.json").read_text())
    data["record"] = data.pop("artifact")
    (path / "study.json").write_text(json.dumps(data))
    (path / "artifact.json").unlink()

    loaded = hb.load(path)

    assert_same(loaded, illustrative_study)


def test_a_saved_run_holds_its_artifact(illustrative_study, tmp_path):
    """A saved run's JSON has the run's artifact under `artifact`, and no `record`,
    and loading gives the artifact back."""
    path = hb.save(illustrative_study, tmp_path / "study")
    run_json = json.loads((path / "runs" / "hybrid_bo" / "run-000.json").read_text())

    assert "record" not in run_json
    assert run_json["artifact"] == illustrative_study.runs[0].artifact
    assert hb.load(path).runs[0].artifact == illustrative_study.runs[0].artifact


def test_load_rejects_a_run_saved_with_a_record(illustrative_study, tmp_path):
    """A run saved before runs had an artifact (its JSON names it `record`) raises
    `StudyFormatError`."""
    # Save, then rename one run's artifact to `record`
    path = hb.save(illustrative_study, tmp_path / "study")
    run_path = path / "runs" / "hybrid_bo" / "run-000.json"
    data = json.loads(run_path.read_text())
    data["record"] = data.pop("artifact")
    run_path.write_text(json.dumps(data))

    with pytest.raises(hb.errors.StudyFormatError, match="record"):
        hb.load(path)


def test_save_and_load_keep_partial_runs_and_empty_arrays(tmp_path):
    """A run that failed at its first point (no evaluated points) round-trips too,
    with its arrays' shapes."""
    run = hb.Run(history_of(0), artifact={"method": "m"}, method="m")
    study = hb.Study((run,), SETTINGS, None, {})

    loaded = hb.load(hb.save(study, tmp_path / "study"))

    assert_same(loaded, study)


def test_a_loaded_run_draws_its_iterations_with_its_own_gp(tmp_path):
    """A run saves its fitted GPs whole, so a loaded run whose GP isn't the
    experiment's (a known noise, a fixed output scale) draws its iteration figures
    with no settings passed, and its GP panel is that GP's prediction."""
    # Setup: one hybrid BO run with its own GP, saved and loaded
    gp = hb.gp.GP(
        hb.gp.Zero(), hb.gp.Scale(hb.gp.Matern52(), outputscale=2.0), noise_std=0.05
    )
    approximation = hb.SAA(unknown_states_as_variables=False)
    settings = hb.HybridBOSettings(gp=gp, approximation=approximation)
    methods = {"noisy": functools.partial(hb.hybrid_bo, settings=settings)}
    study_settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=2, seed=0)
    study = hb.study(methods, ex.problem, ex.measure, study_settings)
    (run,) = hb.load(hb.save(study, tmp_path / "study")).runs

    # Act: every iteration's figure, from the loaded run alone
    figures = [hb.plots.iteration(run, ex.problem, i) for i in range(2)]

    # Check: the stored GP is the prior GP with its Params filled in, and the GP
    # panel draws its prediction
    assert run.surrogates[1][0].gp.noise_std == 0.05
    assert run.surrogates[1][0].gp.kernel == study.runs[0].surrogates[1][0].gp.kernel
    (mean,) = [
        line for line in figures[1].axes[0].get_lines() if line.get_label() == "GP mean"
    ]
    u = np.asarray(mean.get_xdata())[:, np.newaxis]
    expected, _ = hb.gp.predict(study.runs[0].surrogates[1][0], u)
    np.testing.assert_allclose(np.asarray(mean.get_ydata()), expected, rtol=1e-12)
    assert all(len(figure.axes) == 3 for figure in figures)


def test_a_loaded_study_draws_every_bo_method_from_its_stored_settings(tmp_path):
    """The iteration figures replay each run from the settings it stored, so every
    BO method's runs draw with no settings passed, a custom solver's included, and
    the replayed acquisition function passes through the stored value."""
    # Setup: each approximation and acquisition function, and a custom solver
    solver = hb.solvers.Ipopt(solver_options={"print_level": 0, "max_iter": 500})
    hybrid = {
        "saa": hb.HybridBOSettings(gp=ex.gp, approximation=hb.SAA(5, False)),
        "soft": hb.HybridBOSettings(gp=ex.gp, approximation=hb.SoftSAA(0.5, 5, False)),
        "lin": hb.HybridBOSettings(gp=ex.gp, approximation=hb.Linearization(False)),
        "lcb": hb.HybridBOSettings(gp=ex.gp, acquisition=hb.LCB(2.0), solver=solver),
    }
    standard = hb.StandardBOSettings(gp=ex.gp, acquisition=hb.LCB(1.0))
    all_settings = hybrid | {"standard": standard}
    methods = {
        label: functools.partial(hb.hybrid_bo, settings=settings)
        for label, settings in hybrid.items()
    } | {"standard": functools.partial(hb.standard_bo, settings=standard)}
    study_settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=1, seed=0)
    study = hb.study(methods, ex.problem, ex.measure, study_settings)
    loaded = hb.load(hb.save(study, tmp_path / "study"))

    # Act: iteration 0 of every run, with the acquisition panel alone
    figures = [
        hb.plots.iteration(run, ex.problem, 0, gp=False, scenarios=False)
        for run in loaded.runs
    ]

    # Check: each run's settings, and each curve through its stored value (SoftSAA
    # stores -log EI and draws -EI)
    for run, figure in zip(loaded.runs, figures, strict=True):
        assert run.settings == all_settings[run.method]
        (ax,) = figure.axes
        (curve,) = ax.get_lines()
        trial = run.history.u[run.history.n_initial, 0]
        (at_trial,) = np.flatnonzero(np.asarray(curve.get_xdata()) == trial)
        stored = run.acquisitions[0].value
        if run.method == "soft":
            stored = -np.exp(-stored)
        replayed = np.asarray(curve.get_ydata())[at_trial]
        np.testing.assert_allclose(replayed, stored, rtol=1e-8, err_msg=run.method)


def test_save_and_load_keep_every_kind_of_gp_parameter(tmp_path):
    """A fitted GP's parameters round-trip as they were given: a number, a tuple
    (one per input) and, inside a kernel held by another, the right kernel type."""
    gp = hb.gp.GP(
        hb.gp.Zero(),
        hb.gp.Scale(hb.gp.Matern32(lengthscale=0.3), outputscale=1.5),
        noise_std=0.2,
    )
    ard = hb.gp.GP(hb.gp.Zero(), hb.gp.SquaredExponential(lengthscale=(0.1, 0.2)))
    scaler = hb.gp.Scaler(np.ones(1), np.zeros(1))
    surrogates = [
        tuple(
            hb.gp.FittedGP(g, np.zeros((2, 1)), np.zeros(2), scaler, scaler)
            for g in (gp, ard)
        )
    ]
    run = hb.Run(history_of(2), artifact={"type": "an artifact entry"}, method="m")
    study = hb.Study(
        (dataclasses.replace(run, surrogates=surrogates),), SETTINGS, None, {}
    )

    loaded = hb.load(hb.save(study, tmp_path / "study"))

    assert_same(loaded, study)


def test_save_and_load_keep_every_kind_of_method_settings(tmp_path):
    """A run's settings round-trip whole and of the right types: each approximation
    with its own settings (SoftSAA's temperature), LCB with its factor, and a custom
    solver."""
    # Setup: one run per kind of settings, with no points
    solver = hb.solvers.Ipopt(
        plugin_options={"print_time": 0, "expand": True},
        solver_options={"print_level": 0, "max_iter": 50, "tol": 1e-6},
    )
    all_settings = [
        hb.HybridBOSettings(gp=ex.gp, approximation=hb.SAA(5, False), solver=solver),
        hb.HybridBOSettings(gp=ex.gp, approximation=hb.SoftSAA(0.5, 7, False)),
        hb.HybridBOSettings(gp=ex.gp, approximation=hb.Linearization(False)),
        hb.HybridBOSettings(gp=ex.gp, acquisition=hb.LCB(2.5), n_starts=3),
        hb.StandardBOSettings(gp=ex.gp, acquisition=hb.LCB(1.5), n_starts_gp=4),
        hb.StandardBOSettings(gp=ex.gp, solver=solver),
    ]
    runs = tuple(
        hb.Run(history_of(0), {"method": "m"}, "m", index, settings=settings)
        for index, settings in enumerate(all_settings)
    )
    study = hb.Study(runs, SETTINGS, None, {})

    # Act: save and load
    loaded = hb.load(hb.save(study, tmp_path / "study"))

    # Check: equal settings, of the same types throughout
    assert [run.settings for run in loaded.runs] == all_settings
    assert_same(loaded, study)


def test_load_rejects_a_bo_run_saved_without_its_settings(illustrative_study, tmp_path):
    """A BO run saved before runs held their settings (only its record's text) raises
    `StudyFormatError`: the plots replay from the settings."""
    # Save, then rewrite one BO run in the old format: a record and no settings
    path = hb.save(illustrative_study, tmp_path / "study")
    run_path = path / "runs" / "hybrid_bo" / "run-000.json"
    data = json.loads(run_path.read_text())
    data["record"] = data.pop("artifact")
    del data["settings"]
    run_path.write_text(json.dumps(data))

    with pytest.raises(hb.errors.StudyFormatError, match="settings"):
        hb.load(path)


def test_load_reads_a_run_with_one_fitted_gp_per_iteration_as_1_tuples(
    illustrative_study, tmp_path
):
    """A run saved before more than one unknown state, with one fitted GP per
    iteration rather than a tuple of them, loads with each as a 1-tuple."""
    # Save, then rewrite every BO run in the old format: each surrogate a single GP
    path = hb.save(illustrative_study, tmp_path / "study")
    for method in ("hybrid_bo", "standard_bo"):
        run_path = path / "runs" / method / "run-000.json"
        data = json.loads(run_path.read_text())
        data["surrogates"] = [gp for (gp,) in data["surrogates"]]
        run_path.write_text(json.dumps(data))

    loaded = hb.load(path)

    assert_same(loaded, illustrative_study)


def test_load_rejects_a_study_saved_before_gps_were_data(illustrative_study, tmp_path):
    """A study whose fitted GPs were saved as named hyperparameters, before GPs were
    data, raises `StudyFormatError` saying to rerun it."""
    # Rewrite one run's fitted GPs in the old format: hyperparameters, no GP
    path = hb.save(illustrative_study, tmp_path / "study")
    run_path = path / "runs" / "standard_bo" / "run-000.json"
    data = json.loads(run_path.read_text())
    data["surrogates"] = [gp for (gp,) in data["surrogates"]]
    for surrogate in data["surrogates"]:
        del surrogate["gp"]
        surrogate["hyperparameters"] = {"outputscale": [1.0], "lengthscale": [0.5]}
    run_path.write_text(json.dumps(data))

    with pytest.raises(hb.errors.StudyFormatError, match="rerun"):
        hb.load(path)


def test_load_rejects_an_old_bo_run_without_iterations(illustrative_study, tmp_path):
    """An old BO run that stopped before its first iteration has no fitted GP to
    show its format, but its record held the GP's kernel as text: it raises
    `StudyFormatError` too."""
    # Save, then rewrite one run in the old format: no iterations, a record
    path = hb.save(illustrative_study, tmp_path / "study")
    run_path = path / "runs" / "standard_bo" / "run-000.json"
    data = json.loads(run_path.read_text())
    data["surrogates"], data["acquisitions"] = [], []
    data["record"] = data.pop("artifact")
    data["record"]["kernel"] = "outputscale*matern52(u, u')"
    run_path.write_text(json.dumps(data))

    with pytest.raises(hb.errors.StudyFormatError, match="rerun"):
        hb.load(path)


def test_load_rejects_a_gp_part_of_the_wrong_type(illustrative_study, tmp_path):
    """A saved kernel whose tag names no kernel (here, the mean's type) raises
    `StudyFormatError`, not an error from inside the loader."""
    path = hb.save(illustrative_study, tmp_path / "study")
    run_path = path / "runs" / "standard_bo" / "run-000.json"
    data = json.loads(run_path.read_text())
    data["surrogates"][0][0]["gp"]["kernel"]["type"] = "Zero"
    run_path.write_text(json.dumps(data))

    with pytest.raises(hb.errors.StudyFormatError, match="'Zero'"):
        hb.load(path)


def test_save_refuses_an_existing_folder(illustrative_study, tmp_path):
    """Saving never overwrites: an existing folder raises `FileExistsError`."""
    with pytest.raises(FileExistsError):
        hb.save(illustrative_study, tmp_path)


def test_save_refuses_a_label_that_is_not_a_folder_name(tmp_path):
    """A method label is a folder name, so it can't contain a path separator."""
    run = hb.Run(history_of(1), artifact={}, method="a/b")
    study = hb.Study((run,), SETTINGS, None, {})

    with pytest.raises(ValueError, match="a/b"):
        hb.save(study, tmp_path / "study")
    assert not (tmp_path / "study").exists()
