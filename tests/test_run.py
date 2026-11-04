"""Runs: what a method's run holds, and saving and loading one as JSON.

pytest runs every function whose name starts with `test_`.
"""

import dataclasses
import functools
import json
import warnings
from pathlib import Path

import illustrative_example as ex
import numpy as np
import pytest

import hybrid_bo as hb

# The illustrative example's prior GP with a known noise of 0.01
NOISY_GP = dataclasses.replace(ex.gp, noise_std=0.01)


def run_of(method, n_iterations: int = 1, measure=ex.measure) -> hb.Run:
    """A run of `method` on the illustrative example from seed 0: 2 initial points,
    then `n_iterations` iterations."""
    rng = np.random.default_rng(0)
    initial_points = hb.initial_points(ex.system.u, 2, rng)
    return method(ex.problem, measure, initial_points, n_iterations, rng=rng)


def assert_same(actual, expected, where="run"):
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


def round_trip(run: hb.Run, tmp_path: Path) -> hb.Run:
    """The run after saving and loading it."""
    return hb.load_run(hb.save_run(run, tmp_path / "run.json"))


# `parametrize` runs the test once per value, each as its own test case
@pytest.mark.parametrize(
    "method",
    [
        functools.partial(
            hb.hybrid_bo,
            settings=hb.HybridBOSettings(
                gp=NOISY_GP,
                acquisition=hb.EI(),
                approximation=hb.SAA(n_scenarios=5, unknown_states_as_variables=False),
            ),
        ),
        functools.partial(
            hb.standard_bo,
            settings=hb.StandardBOSettings(gp=NOISY_GP, acquisition=hb.EI()),
        ),
    ],
    ids=["hybrid_bo", "standard_bo"],
)
def test_a_larger_budget_passes_through_the_evaluations_of_a_smaller_one(method):
    """The prefix property: a BO method doesn't depend on how many evaluations it has
    left, so with the same seed a run of 3 iterations is the start of a run of 5, its
    points, states, objectives, fitted GPs and acquisition solutions alike. The
    measurement is noisy, so every evaluation draws from the run's generator too."""
    # Setup and act: one run of each budget, from the same seed
    noisy = hb.oracle_measure(ex.system, ex.oracle, noise_std=0.01)
    short = run_of(method, 3, noisy)
    long = run_of(method, 5, noisy)

    # Check: the first 5 evaluations and 3 iterations of the long run are the short run
    for field in ("u", "x", "y", "f", "x_oracle", "y_oracle", "f_oracle"):
        expected = getattr(short.history, field)
        assert_same(getattr(long.history, field)[:5], expected, field)
    assert_same(long.surrogates[:3], short.surrogates, "surrogates")
    assert_same(long.acquisitions[:3], short.acquisitions, "acquisitions")
    assert_same(long.scenarios, short.scenarios, "scenarios")


def test_an_optimum_beaten_by_an_evaluated_point_warns():
    """A stated optimum above an evaluated point's oracle objective can't be the
    optimum, so it warns; the true optimum gives no warning."""
    # Two uniform points; the true optimum is about -50.46, and f is at most about 75
    history = run_of(hb.uniform_sampling, 1).history

    # `pytest.warns` fails unless the block warns with the category, matching `match`
    with pytest.warns(hb.OracleOptimumWarning, match="below the stated"):
        assert hb.check_oracle_optimum(history, 100.0)

    # The warning turned into an error, so the test fails if it is raised
    with warnings.catch_warnings():
        warnings.simplefilter("error", hb.OracleOptimumWarning)
        assert not hb.check_oracle_optimum(history, ex.optimum)


def test_a_history_without_an_oracle_has_no_optimum_to_beat():
    """Without oracle values there is nothing to compare, so it doesn't warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", hb.OracleOptimumWarning)
        assert not hb.check_oracle_optimum(history_of(2), 1e9)


# `tmp_path` is a pytest fixture: a new, empty folder for this test
@pytest.mark.parametrize("label", ex.methods)
def test_load_run_after_save_run_gives_the_run_back(label, tmp_path):
    """Every method's run loads back equal to the one saved, arrays bit for bit."""
    run = run_of(ex.methods[label])

    assert_same(round_trip(run, tmp_path), run)


def test_a_saved_run_holds_its_artifact(tmp_path):
    """A saved run's JSON has the run's artifact under `artifact`, and loading gives
    it back."""
    run = run_of(ex.methods["standard_bo"])

    path = hb.save_run(run, tmp_path / "run.json")

    assert json.loads(path.read_text())["artifact"] == run.artifact
    assert hb.load_run(path).artifact == run.artifact


def test_save_and_load_keep_partial_runs_and_empty_arrays(tmp_path):
    """A run that failed at its first point (no evaluated points) round-trips too,
    with its arrays' shapes."""
    run = hb.Run(history_of(0), artifact={"method": "m"}, method="m")

    assert_same(round_trip(run, tmp_path), run)


def test_a_loaded_run_draws_its_iterations_with_its_own_gp(tmp_path):
    """A run saves its fitted GPs whole, so a loaded run whose GP isn't the
    experiment's (a known noise, a fixed output scale) draws its iteration figures
    with no settings passed, and its GP panel is that GP's prediction."""
    # Setup: one hybrid BO run with its own GP, saved and loaded
    gp = hb.gp.GP(
        mean=hb.gp.Zero(),
        kernel=hb.gp.Scale(hb.gp.Matern52(), outputscale=2.0),
        noise_std=0.05,
    )
    approximation = hb.SAA(n_scenarios=25, unknown_states_as_variables=False)
    settings = hb.HybridBOSettings(
        gp=gp, acquisition=hb.EI(), approximation=approximation
    )
    original = run_of(functools.partial(hb.hybrid_bo, settings=settings), 2)
    run = round_trip(original, tmp_path)

    # Act: every iteration's figure, from the loaded run alone
    figures = [hb.plots.iteration(run, ex.problem, i) for i in range(2)]

    # Check: the stored GP is the prior GP with its Params filled in, and the GP
    # panel draws its prediction
    assert run.surrogates[1][0].gp.noise_std == 0.05
    assert run.surrogates[1][0].gp.kernel == original.surrogates[1][0].gp.kernel
    (mean,) = [
        line for line in figures[1].axes[0].get_lines() if line.get_label() == "GP mean"
    ]
    u = np.asarray(mean.get_xdata())[:, np.newaxis]
    expected, _ = hb.gp.predict(original.surrogates[1][0], u)
    np.testing.assert_allclose(np.asarray(mean.get_ydata()), expected, rtol=1e-12)
    assert all(len(figure.axes) == 3 for figure in figures)


def test_save_and_load_keep_every_kind_of_gp_parameter(tmp_path):
    """A fitted GP's parameters round-trip as they were given: a number, a tuple
    (one per input) and, inside a kernel held by another, the right kernel type."""
    gp = hb.gp.GP(
        mean=hb.gp.Zero(),
        kernel=hb.gp.Scale(hb.gp.Matern32(lengthscale=0.3), outputscale=1.5),
        noise_std=0.2,
    )
    ard = hb.gp.GP(
        mean=hb.gp.Zero(),
        kernel=hb.gp.SquaredExponential(lengthscale=(0.1, 0.2)),
        noise_std=0.0,
    )
    scaler = hb.gp.Scaler(np.ones(1), np.zeros(1))
    surrogates = [
        tuple(
            hb.gp.FittedGP(g, np.zeros((2, 1)), np.zeros(2), scaler, scaler)
            for g in (gp, ard)
        )
    ]
    run = hb.Run(
        history_of(2),
        artifact={"type": "an artifact entry"},
        method="m",
        surrogates=surrogates,
    )

    assert_same(round_trip(run, tmp_path), run)


def test_an_empty_array_in_an_optional_field_loads_with_its_shape(tmp_path):
    """An empty array where None is allowed too (here the oracle states of a system
    without mechanistic states, shape (n, 0)) loads as an array of the saved shape,
    not None."""
    history = dataclasses.replace(
        history_of(3), x=np.zeros((3, 0)), x_oracle=np.zeros((3, 0))
    )
    run = hb.Run(history, artifact={}, method="m")

    assert_same(round_trip(run, tmp_path), run)


SOLVER = hb.solvers.Ipopt(
    plugin_options={"print_time": 0, "expand": True},
    solver_options={"print_level": 0, "max_iter": 50, "tol": 1e-6},
)


@pytest.mark.parametrize(
    "settings",
    [
        hb.HybridBOSettings(
            gp=ex.gp,
            acquisition=hb.EI(),
            approximation=hb.SAA(n_scenarios=5, unknown_states_as_variables=False),
            solver=SOLVER,
        ),
        hb.HybridBOSettings(
            gp=ex.gp,
            acquisition=hb.EI(),
            approximation=hb.SoftSAA(
                temperature=0.5, n_scenarios=7, unknown_states_as_variables=False
            ),
        ),
        hb.HybridBOSettings(
            gp=ex.gp,
            acquisition=hb.EI(),
            approximation=hb.Linearization(unknown_states_as_variables=False),
        ),
        hb.HybridBOSettings(
            gp=ex.gp,
            acquisition=hb.LCB(std_factor=2.5),
            approximation=hb.SAA(n_scenarios=25),
            n_starts=3,
        ),
        hb.StandardBOSettings(
            gp=ex.gp, acquisition=hb.LCB(std_factor=1.5), n_starts_gp=4
        ),
        hb.StandardBOSettings(gp=ex.gp, acquisition=hb.EI(), solver=SOLVER),
    ],
)
def test_save_and_load_keep_every_kind_of_method_settings(settings, tmp_path):
    """A run's settings round-trip whole and of the right types: each approximation
    with its own settings (SoftSAA's temperature), LCB with its factor, and a custom
    solver."""
    run = hb.Run(history_of(0), {"method": "m"}, "m", settings=settings)

    loaded = round_trip(run, tmp_path)

    assert loaded.settings == settings
    assert_same(loaded, run)


def test_load_rejects_a_gp_part_of_the_wrong_type(tmp_path):
    """A saved kernel whose tag names no kernel (here, the mean's type) raises
    `RunFormatError`, not an error from inside the loader."""
    # Save a run, then give its kernel the mean's type
    path = hb.save_run(run_of(ex.methods["standard_bo"]), tmp_path / "run.json")
    data = json.loads(path.read_text())
    data["surrogates"][0][0]["gp"]["kernel"]["type"] = "Zero"
    path.write_text(json.dumps(data))

    # `pytest.raises` passes only if the block raises this exception, with a message
    # matching the regular expression `match`
    with pytest.raises(hb.errors.RunFormatError, match="'Zero'"):
        hb.load_run(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data | {"index": 0},  # a field a run no longer has
        lambda data: {key: data[key] for key in data if key != "history"},
    ],
    ids=["old_field", "missing_field"],
)
def test_load_rejects_a_file_that_is_not_a_run_of_this_version(change, tmp_path):
    """A run saved by an older version, with a field `Run` no longer has or without
    one it needs, raises `RunFormatError`, not an error from inside the loader."""
    # Save a run, then change its fields
    path = hb.save_run(hb.Run(history_of(0), {}, "m"), tmp_path / "run.json")
    path.write_text(json.dumps(change(json.loads(path.read_text()))))

    with pytest.raises(hb.errors.RunFormatError, match="of this version"):
        hb.load_run(path)


def test_save_run_refuses_an_existing_file(tmp_path):
    """`save_run` never overwrites a file: an existing one raises `FileExistsError`."""
    path = tmp_path / "run.json"
    path.write_text("{}")

    with pytest.raises(FileExistsError):
        hb.save_run(hb.Run(history_of(0), {}, "m"), path)
    assert path.read_text() == "{}"
