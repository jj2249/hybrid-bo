"""Hybrid BO's replay of an iteration's acquisition function over a grid of u, which
the iteration figures draw.

The replay solves each scenario's equations with u fixed by Newton's method and
evaluates the acquisition problem's objective there. The reference values in `tests/data/replay/`
were recorded from the replay it replaced (#154), which solved the acquisition problem
with IPOPT at each grid point, so they check the root-finder against an independent
solver on the same runs.
"""

import functools
import pickle
from dataclasses import replace
from pathlib import Path

import casadi as cas
import illustrative_example as ex
import illustrative_example_2d as ex_2d
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy

from hybrid_bo import (
    EI,
    LCB,
    SAA,
    HybridBOSettings,
    Linearization,
    Problem,
    ReplayWarning,
    Run,
    SoftSAA,
    System,
    Var,
    hybrid_bo,
    initial_points,
    plots,
)
from hybrid_bo.gp import predict
from hybrid_bo.plots import _N_GRID, _replay
from hybrid_bo.sampling import _start_bounds

REFERENCES = Path(__file__).parent / "data" / "replay"

# Each approximation the figures' tests cover, with the unknown states substituted
# except for LCB's
APPROXIMATIONS = {
    "saa": (EI(), SAA(n_scenarios=25, unknown_states_as_variables=False)),
    "lcb": (LCB(std_factor=2.0), SAA(n_scenarios=25)),
    "lin": (EI(), Linearization(unknown_states_as_variables=False)),
    "soft": (
        EI(),
        SoftSAA(temperature=0.5, n_scenarios=10, unknown_states_as_variables=False),
    ),
}
EXAMPLES = {"1d": (ex.problem, ex.measure), "2d": (ex_2d.problem, ex_2d.measure)}
# Grid points over each decision variable in 2D: fewer than the figure's 30, which
# keeps the recorded values small; the replay takes any points
N_GRID_2D = 10


# `functools.cache` builds each run once, for every test that needs it
@functools.cache
def hybrid_run(example: str, approximation: str) -> Run:
    """A pickled hybrid BO run of an illustrative example with one of
    `APPROXIMATIONS`: 2 initial points, then 2 iterations."""
    problem, measure = EXAMPLES[example]
    acquisition, approx = APPROXIMATIONS[approximation]
    settings = HybridBOSettings(gp=ex.gp, acquisition=acquisition, approximation=approx)
    rng = np.random.default_rng(0)
    u = initial_points(problem.system.u, 2, rng)
    run = hybrid_bo(problem, measure, u, 2, rng, settings=settings)
    return pickle.loads(pickle.dumps(run))


def replay_grid(problem: Problem, run: Run, i: int) -> np.ndarray:
    """The rows of u to replay at, as the iteration figure orders them: in 1D its 101
    points over the start bounds with the trial point inserted, in 2D the trial
    point, then a 10 x 10 grid over the start bounds."""
    trial = run.history.u[run.history.n_initial + i]
    lower, upper = _start_bounds(problem.system.u)
    if problem.system.n_u == 1:
        grid = np.linspace(lower[0], upper[0], _N_GRID)
        return np.sort(np.append(grid, trial))[:, np.newaxis]
    u1, u2 = (np.linspace(lower[j], upper[j], N_GRID_2D) for j in range(2))
    U1, U2 = np.meshgrid(u1, u2)
    return np.vstack([trial, np.column_stack([U1.ravel(), U2.ravel()])])


CASES = [
    (example, approximation, i)
    for example in EXAMPLES
    for approximation in APPROXIMATIONS
    for i in (0, 1)
]


# `parametrize` runs the test once per case in the list, each as its own test
@pytest.mark.parametrize(("example", "approximation", "i"), CASES)
def test_replay_matches_the_ipopt_replay(example, approximation, i):
    """The replayed acquisition values, f under each scenario and Lin-EI's std match
    the values the IPOPT replay recorded, with NaN at the same grid points (the
    recorded values have none; `test_a_root_outside_the_bounds_leaves_a_gap` checks
    the gaps). The values and the std agree to 1e-5 relative, f to 1e-6 relative or
    1e-5 absolute: IPOPT met the known equations only to its tolerance, which moves f
    by up to about 1e-5 where it is steep (Newton's roots match scipy's brentq on
    x + exp(x) = y to 1e-9)."""
    # Setup: the run and the recorded values
    problem = EXAMPLES[example][0]
    run = hybrid_run(example, approximation)
    reference = np.load(REFERENCES / f"{example}-{approximation}-{i}.npz")
    u = replay_grid(problem, run, i)
    np.testing.assert_array_equal(u, reference["u"])

    # Act
    assert run.settings is not None
    values, f, f_std = _replay(run, problem, i, run.settings, u)

    # Check: the values, and Lin-EI's std only with linearization
    assert f is not None
    np.testing.assert_allclose(values, reference["values"], rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(f, reference["f"], rtol=1e-6, atol=1e-5)
    if approximation == "lin":
        assert f_std is not None
        np.testing.assert_allclose(f_std, reference["f_std"], rtol=1e-5, atol=1e-12)
    else:
        assert f_std is None


@pytest.mark.parametrize("example", list(EXAMPLES))
def test_both_formulations_replay_the_same_values(example):
    """With the unknown states as variables or substituted, the scenarios' equations
    have the same roots, so the replay gives the same values."""
    # Setup: the SAA-EI run, which substitutes the unknown states, and its settings
    # with them as variables
    problem = EXAMPLES[example][0]
    run = hybrid_run(example, "saa")
    assert isinstance(run.settings, HybridBOSettings)
    as_variables = replace(
        run.settings,
        approximation=replace(
            run.settings.approximation, unknown_states_as_variables=True
        ),
    )
    u = replay_grid(problem, run, 1)

    # Act
    values, f, _ = _replay(run, problem, 1, run.settings, u)
    values_as_variables, f_as_variables, _ = _replay(run, problem, 1, as_variables, u)

    # Check: the acquisition values and f under each scenario
    assert f is not None and f_as_variables is not None
    np.testing.assert_allclose(values_as_variables, values, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(f_as_variables, f, rtol=1e-9, atol=1e-12)


def bounded_problem(x_upper: float) -> Problem:
    """The illustrative example with its mechanistic state bounded above (its start
    bounds cut there too)."""
    x = Var("x", start=(-2.5, min(1.0, x_upper)), upper=x_upper)
    system = System(
        u=ex.system.u,
        x=[x],
        y=ex.system.y,
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        h_inputs=ex.system.u,
        measured=ex.system.y,
    )
    return Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))


def scenario_x(run: Run, i: int, u: np.ndarray) -> np.ndarray:
    """x_s at each u (n_points, 1) and scenario, (n_points, S), computed independently:
    y_s from the GP, then x_s from x + exp(x) = y_s with scipy's brentq."""
    assert run.scenarios is not None
    mean, std = predict(run.surrogates[i][0], u)
    y = mean[:, np.newaxis] + std[:, np.newaxis] * run.scenarios.z[:, 0]
    return np.vectorize(
        lambda y_s: scipy.optimize.brentq(lambda x: x + np.exp(x) - y_s, -50, 50)
    )(y)


def test_a_root_outside_the_bounds_leaves_a_gap():
    """A grid point where some scenario's root lies above the mechanistic state's upper
    bound is NaN in the acquisition curve; the other points are drawn."""
    # Setup: x bounded just above its largest root at the trial point, so the trial
    # point is solved and some grid points aren't
    run, i = hybrid_run("1d", "saa"), 1
    trial = run.history.u[run.history.n_initial + i]
    x_upper = float(scenario_x(run, i, trial[np.newaxis]).max()) + 0.05
    problem = bounded_problem(x_upper)

    # Act: the acquisition panel alone
    figure = plots.iteration(run, problem, i, gp=False, scenarios=False)

    # Check: NaN exactly where a scenario's root is above the bound
    (curve,) = figure.axes[0].get_lines()
    u = np.asarray(curve.get_xdata())[:, np.newaxis]
    outside = (scenario_x(run, i, u) > x_upper).any(axis=1)
    assert outside.any() and not outside.all()
    np.testing.assert_array_equal(np.isnan(curve.get_ydata()), outside)
    plt.close("all")


def test_a_stored_solution_off_the_root_warns():
    """If the root at the trial point isn't the stored solution, here because the
    stored states were moved off it, the figure warns with `ReplayWarning` and is
    still drawn."""
    # Setup: iteration 1's stored states moved by 0.5
    run = hybrid_run("1d", "saa")
    acquisitions = list(run.acquisitions)
    stored = acquisitions[1]
    assert stored.x is not None
    acquisitions[1] = replace(stored, x=stored.x + 0.5)
    run = replace(run, acquisitions=acquisitions)

    # Act and check: `pytest.warns` passes only if the block warns with this category
    with pytest.warns(ReplayWarning, match="branch"):
        figure = plots.iteration(run, ex.problem, 1, gp=False, scenarios=False)
    assert len(figure.axes) == 1
    plt.close("all")


def test_a_trial_point_without_a_root_in_the_bounds_warns():
    """If no root at the trial point is within the bounds, the figure warns with
    `ReplayWarning` and is still drawn, with a gap there."""
    # Setup: x bounded below its largest root at the trial point
    run, i = hybrid_run("1d", "saa"), 1
    trial = run.history.u[run.history.n_initial + i]
    problem = bounded_problem(float(scenario_x(run, i, trial[np.newaxis]).max()) - 0.01)

    # Act and check
    with pytest.warns(ReplayWarning, match="no root"):
        figure = plots.iteration(run, problem, i, gp=False, scenarios=False)
    (curve,) = figure.axes[0].get_lines()
    at_trial = np.asarray(curve.get_xdata()) == trial[0]
    assert np.isnan(np.asarray(curve.get_ydata())[at_trial]).all()
    plt.close("all")
