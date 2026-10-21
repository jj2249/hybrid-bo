"""Hybrid BO with two unknown states (P = 2), end to end: the runs, saving and loading
them, and their iteration figures.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture below (here `study_1d`) is filled in by pytest with that fixture's value;
`tmp_path` is pytest's own fixture, a fresh temporary folder for the test.
"""

import functools
from typing import Any

import casadi as cas
import illustrative_example as ex
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hybrid_bo as hb
from hybrid_bo.gp import FittedGP
from hybrid_bo.saving import _encode

# Each approximation, with few scenarios so the runs are quick
APPROXIMATIONS = {
    "saa": hb.SAA(n_scenarios=5),
    "soft": hb.SoftSAA(0.5, n_scenarios=5),
    "lin": hb.Linearization(),
}


def benchmark_problem(n_u: int) -> tuple[hb.Problem, hb.Measure]:
    """A test system with n_u decision variables, one mechanistic state x1 and two
    unknown states y1 = sin(u1) and y2 = cos(u_last), both measured, with
    x1 + exp(x1) = y1 + y2 and f = x1^2 + (y1 - y2)^2; and its oracle measurement."""
    u = [hb.Var(f"u{i + 1}", -2.0, 2.0) for i in range(n_u)]
    x1 = hb.Var("x1", start=(-2.5, 1.0))
    y1, y2 = hb.Var("y1", start=(-1.0, 1.0)), hb.Var("y2", start=(-1.0, 1.0))
    system = hb.System(
        u=u,
        x=[x1],
        y=[y1, y2],
        g=lambda v: [v.x1 + cas.exp(v.x1) - v.y1 - v.y2],
        h_inputs=u,
        measured=[y1, y2],
    )
    problem = hb.Problem(system, f=lambda v: v.x1**2 + (v.y1 - v.y2) ** 2)
    last = f"u{n_u}"
    measure = hb.oracle_measure(
        system, oracle=lambda v: [cas.sin(v.u1), cas.cos(getattr(v, last))]
    )
    return problem, measure


def two_state_study(n_u: int, approximations: dict[str, Any]) -> hb.Study:
    """One hybrid BO run per approximation on the test system: 2 initial points, 2
    iterations."""
    problem, measure = benchmark_problem(n_u)
    methods = {
        label: functools.partial(
            hb.hybrid_bo,
            settings=hb.HybridBOSettings(gp=ex.gp, approximation=approximation),
        )
        for label, approximation in approximations.items()
    }
    settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=2, seed=0)
    return hb.study(methods, problem, measure, settings)


# A fixture is a value pytest builds once and passes to every test that names it as an
# argument; `scope="module"` shares it between the tests of this file, since runs are
# slow to produce
@pytest.fixture(scope="module")
def study_1d() -> hb.Study:
    """The study with one decision variable and every approximation."""
    return two_state_study(1, APPROXIMATIONS)


def test_each_iteration_fits_one_gp_per_unknown_state(study_1d):
    """Every approximation runs its 2 iterations, and each iteration's surrogate holds
    two fitted GPs, one per unknown state, on the same arguments of h and fitted to
    that unknown state's values."""
    for run in study_1d.runs:
        assert len(run.surrogates) == 2, run.method
        for i, surrogate in enumerate(run.surrogates):
            assert len(surrogate) == 2
            assert all(isinstance(fitted, FittedGP) for fitted in surrogate)

            # Both GPs see the points before the trial point; each its own column
            n = run.history.n_initial + i
            for j, fitted in enumerate(surrogate):
                scaler = fitted.y_scaler
                Y = (fitted.Y - scaler.intercept) / scaler.slope
                np.testing.assert_allclose(Y, run.history.y[:n, j], atol=1e-12)
            np.testing.assert_array_equal(surrogate[0].X, surrogate[1].X)

        # Each acquisition solution has both unknown states per scenario
        n_scenarios = len(run.scenarios.w)
        assert all(a.y.shape == (n_scenarios, 2) for a in run.acquisitions)


def test_the_study_saves_and_loads_equal(study_1d, tmp_path):
    """A study with two GPs per iteration loads back equal to the one saved."""
    loaded = hb.load(hb.save(study_1d, tmp_path / "study"))

    assert _encode(loaded) == _encode(study_1d)
    assert all(len(s) == 2 for run in loaded.runs for s in run.surrogates)


def test_iteration_figures_replay_without_the_gp_panels(study_1d):
    """With two unknown states, the iteration figure draws the scenarios and the
    replayed acquisition function, whose curve passes through the stored value at the
    trial point (SoftSAA's -log EI drawn as -EI)."""
    problem, _ = benchmark_problem(1)
    for run in study_1d.runs:
        figure = hb.plots.iteration(run, problem, 1, gp=False)

        # Two panels: the scenarios and the acquisition function
        scenarios_ax, acquisition_ax = figure.axes
        assert scenarios_ax.get_ylabel() == "f"
        (curve,) = acquisition_ax.get_lines()
        trial = run.history.u[run.history.n_initial + 1, 0]
        (at_trial,) = np.flatnonzero(np.asarray(curve.get_xdata()) == trial)
        stored = run.acquisitions[1].value
        if run.method == "soft":
            stored = -np.exp(-stored)
        replayed = np.asarray(curve.get_ydata())[at_trial]
        np.testing.assert_allclose(replayed, stored, rtol=1e-8, err_msg=run.method)
    plt.close("all")


def test_iteration_figures_refuse_the_gp_panels(study_1d):
    """The GP panels draw one unknown state, so asking for them with two raises
    `ValueError` saying to pass gp=False."""
    problem, _ = benchmark_problem(1)

    # `pytest.raises` passes only if the block raises this exception, with a message
    # matching the regular expression `match`
    with pytest.raises(ValueError, match="gp=False"):
        hb.plots.iteration(study_1d.runs[0], problem, 0)


def test_2d_iteration_figure_replays_without_the_gp_panels():
    """With two decision variables, the figure is one row of f under the scenarios
    and log10 EI, whose note gives the same stored and replayed value."""
    # Setup: one SAA run with two decision variables
    problem, _ = benchmark_problem(2)
    (run,) = two_state_study(2, {"saa": APPROXIMATIONS["saa"]}).runs

    figure = hb.plots.iteration(run, problem, 1, gp=False)

    # Check: the panels of f, and the replay through the stored value
    titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
    assert titles == ["mean f over scenarios", "std f over scenarios", "log10 EI"]
    (note,) = [t.get_text() for t in figure.axes[2].texts]
    stored, replayed = (
        float(part.split()[-1])
        for part in note.removeprefix("trial point: ").split(",")
    )
    assert replayed == pytest.approx(stored, rel=1e-6)
    plt.close("all")
