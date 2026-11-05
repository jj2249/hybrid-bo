"""Systems with no known equations (M = 0), end to end: every method runs, runs save
and load, and the iteration figures draw. Hybrid BO is then BO of a composite function
f(y, u) with y = h(u) (Astudillo & Frazier, arXiv 1906.01537).

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture below (here `runs_1d`) is filled in by pytest with that fixture's value;
`tmp_path` is pytest's own fixture, a fresh temporary folder for the test.
"""

import dataclasses
import functools

import casadi as cas
import illustrative_example as ex
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hybrid_bo as hb
from hybrid_bo.acquisition import (
    hybrid_acquisition,
    saa_scenarios,
    standard_acquisition,
)
from hybrid_bo.hybrid_model import _pack
from hybrid_bo.saving import _encode

# Hybrid BO with each approximation, both ways where it applies, and the other methods
HYBRID = {
    f"{label}-{as_variables}": hb.settings.HybridBOSettings(
        gp=ex.gp,
        acquisition=hb.EI(),
        approximation=approximation(as_variables),
    )
    for as_variables in (True, False)
    for label, approximation in {
        "saa": lambda flag: hb.SAA(n_scenarios=5, unknown_states_as_variables=flag),
        "soft": lambda flag: hb.SoftSAA(
            temperature=0.5, n_scenarios=5, unknown_states_as_variables=flag
        ),
        "lin": lambda flag: hb.Linearization(unknown_states_as_variables=flag),
    }.items()
}
METHODS = {
    label: functools.partial(hb.hybrid_bo, **vars(settings))
    for label, settings in HYBRID.items()
} | {
    "standard": functools.partial(hb.standard_bo, gp=ex.gp, acquisition=hb.EI()),
    "lhs": hb.latin_hypercube_sampling,
    "uniform": hb.uniform_sampling,
}


def runs_of(methods: dict, problem: hb.Problem, measure: hb.Measure) -> list[hb.Run]:
    """A run of each method from seed 0, named by its label: 2 initial points and 2
    iterations."""
    runs = []
    for label, method in methods.items():
        rng = np.random.default_rng(0)
        run = method(problem, measure, n_initial=2, n_iterations=2, rng=rng)
        runs.append(dataclasses.replace(run, method=label))
    return runs


def composite_problem(n_u: int) -> tuple[hb.Problem, hb.Measure]:
    """A test system with n_u decision variables in [-2, 2], no mechanistic states and
    one unknown state y = sin(u1) + u_last / 2, measured, with the composite objective
    f = (y - 0.3)^2 + u1^2 / 10; and its oracle measurement."""
    u = [hb.Var(f"u{i + 1}", -2.0, 2.0) for i in range(n_u)]
    y = hb.Var("y", start=(-2.0, 2.0))
    system = hb.System(u=u, y=[y], h_inputs=u, measured=[y])
    problem = hb.Problem(system, f=lambda v: (v.y - 0.3) ** 2 + v.u1**2 / 10)
    last = f"u{n_u}"
    measure = hb.oracle_measure(
        system,
        oracle=lambda v: [cas.sin(v.u1) + getattr(v, last) / 2],
        noise_std=0.0,
    )
    return problem, measure


# A fixture is a value pytest builds once and passes to every test that names it as an
# argument; `scope="module"` shares it between the tests of this file, since runs are
# slow to produce
@pytest.fixture(scope="module")
def runs_1d() -> list[hb.Run]:
    """Every method on the test system with one decision variable."""
    problem, measure = composite_problem(1)
    return runs_of(METHODS, problem, measure)


# Each `parametrize` runs the test once per entry; together, once per combination. The
# third evaluation is the first after the initial points and the fourth is the last,
# after which no GP is fitted that could trip over the NaN
@pytest.mark.parametrize("label", ["saa-True", "standard", "lhs"])
@pytest.mark.parametrize("failing_call", [3, 4])
def test_a_nan_measurement_fails_the_run_and_keeps_the_points_before(
    label: str, failing_call: int
):
    """With no known equations nothing is solved, so no root can fail: a measurement of
    NaN fails the run all the same, and the partial run holds no NaN."""
    # A measurement that returns NaN at one evaluation
    problem, _ = composite_problem(1)
    calls = []

    def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """y = sin(u), and NaN at the failing call."""
        calls.append(u)
        return np.array([np.nan]) if len(calls) == failing_call else np.sin(u)

    # `pytest.raises` fails the test unless the block raises that exception, whose
    # message matches; `info.value` is the exception
    with pytest.raises(hb.RootFindingError, match="isn't finite") as info:
        METHODS[label](
            problem, measure, n_initial=2, n_iterations=2, rng=np.random.default_rng(0)
        )

    # The partial run: the points before the failing one, all of them finite
    run = info.value.run
    assert run is not None
    assert len(run.history.u) == failing_call - 1
    assert np.isfinite(run.history.y).all()
    assert np.isfinite(run.history.f).all()

    # Only a trial point that was evaluated keeps its surrogate
    n_surrogates = 0 if label == "lhs" else failing_call - 3
    assert len(run.surrogates) == n_surrogates


def test_every_method_runs_without_known_equations(runs_1d):
    """Every method finishes its 2 iterations (design points, for the baselines),
    with no mechanistic states in its history and, for hybrid BO, none in its
    scenario states; the artifact says there are no known equations."""
    assert [run.method for run in runs_1d] == list(METHODS)
    for run in runs_1d:
        assert len(run.history.f) == 4, run.method
        assert run.history.x.shape == (4, 0)
        assert run.artifact["g"] == "none"
    for run in (run for run in runs_1d if run.method in HYBRID):
        assert len(run.surrogates) == 2
        assert all(solution.x.shape[1] == 0 for solution in run.acquisitions)


def test_the_oracle_optimum_works_without_known_equations():
    """The oracle optimum of a system without known equations is the minimum of the
    composite objective (here by a dense grid over u)."""
    problem, _ = composite_problem(1)
    optimum = hb.oracle_optimum(
        problem, lambda v: [cas.sin(v.u1) + v.u1 / 2], np.random.default_rng(0)
    )

    # Check: the optimum against the grid
    u = np.linspace(-2.0, 2.0, 400_001)
    f = (np.sin(u) + u / 2 - 0.3) ** 2 + u**2 / 10
    # `pytest.approx` compares with a tolerance
    assert optimum == pytest.approx(f.min(), abs=1e-8)


def test_a_run_saves_and_loads_equal(runs_1d, tmp_path):
    """A run of a system without known equations loads back equal to the one saved,
    its empty arrays (states, oracle states and scenario states of shape (., 0))
    included."""
    run = hb.load_run(hb.save_run(runs_1d[0], tmp_path / "run.json"))

    assert _encode(run) == _encode(runs_1d[0])
    solution = run.acquisitions[0]
    assert run.history.x_oracle is not None and solution.x is not None
    assert run.history.x.shape == run.history.x_oracle.shape == (4, 0)
    assert solution.x.shape == (5, 0)


def test_iteration_figures_draw_without_known_equations(runs_1d):
    """Every BO method's iteration figure draws with one decision variable, its GP
    panel (y over u for hybrid BO), scenarios and acquisition function included."""
    problem, measure = composite_problem(1)
    bo_runs = (run for run in runs_1d if run.method in [*HYBRID, "standard"])
    for run in bo_runs:
        figure = hb.plots.iteration(run, problem, 1, measure=measure)
        assert len(figure.axes) == (3 if run.method in HYBRID else 2), run.method
    plt.close("all")


def test_2d_iteration_figure_draws_without_known_equations():
    """With two decision variables, hybrid BO's figure has its row of y and its row
    of f, and the replayed EI agrees with the stored value at the trial point."""
    # Setup: one SAA run with two decision variables
    problem, measure = composite_problem(2)
    methods = {"saa": METHODS["saa-False"]}
    (run,) = runs_of(methods, problem, measure)

    figure = hb.plots.iteration(run, problem, 1, measure=measure)

    # Check: both rows, and the replay through the stored value
    titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
    assert titles[:3] == ["oracle y", "GP mean of y", "GP std of y"]
    assert titles[-1] == "log10 EI"
    (ei_ax,) = [ax for ax in figure.axes if ax.get_title() == "log10 EI"]
    (note,) = [t.get_text() for t in ei_ax.texts]
    stored, replayed = (
        float(part.split()[-1])
        for part in note.removeprefix("trial point: ").split(",")
    )
    assert replayed == pytest.approx(stored, rel=1e-6)
    plt.close("all")


def test_with_f_y_hybrid_bo_is_standard_bo():
    """Without known equations and with f = y over h(u), hybrid BO models what
    standard BO models: from the same seed their first GPs see the same points and
    the same GP-fit starts (the scenario draws use the generator's stream, which the
    Latin hypercube starts don't), so they predict the same to rounding. SAA-EI at a
    fixed u, over many scenarios, is then within Monte Carlo error of standard BO's
    analytic EI there."""
    # Setup: the composite system with f = y, and one iteration of each method
    u = hb.Var("u", -2.0, 2.0)
    y = hb.Var("y", start=(-2.0, 2.0))
    system = hb.System(u=[u], y=[y], h_inputs=[u], measured=[y])
    problem = hb.Problem(system, f=lambda v: v.y)
    measure = hb.oracle_measure(
        system, oracle=lambda v: [cas.sin(3 * v.u)], noise_std=0.0
    )
    initial = np.array([[-1.5], [-0.2], [0.4], [1.3]])
    runs = {
        name: method(
            problem,
            measure,
            initial_points=initial,
            n_iterations=1,
            rng=np.random.default_rng(0),
        )
        for name, method in {
            "hybrid": functools.partial(hb.hybrid_bo, **vars(HYBRID["saa-False"])),
            "standard": METHODS["standard"],
        }.items()
    }
    ((hybrid_gp,),) = runs["hybrid"].surrogates
    ((standard_gp,),) = runs["standard"].surrogates

    # Check: the same predictions over the decision variable's range
    grid = np.linspace(-2.0, 2.0, 41)[:, np.newaxis]
    hybrid_mean, hybrid_std = hb.gp.predict(hybrid_gp, grid)
    standard_mean, standard_std = hb.gp.predict(standard_gp, grid)
    np.testing.assert_allclose(hybrid_mean, standard_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hybrid_std, standard_std, rtol=1e-12, atol=1e-12)

    # SAA-EI over 500 scenarios at a fixed u, against the analytic EI there
    incumbent, point = float(runs["standard"].history.f[:4].min()), np.array([0.9])
    scenarios = saa_scenarios(500, 1, np.random.default_rng(1))
    saa = hybrid_acquisition(
        problem,
        (hybrid_gp,),
        scenarios,
        incumbent,
        hb.EI(),
        hb.SAA(n_scenarios=500, unknown_states_as_variables=False),
    )
    saa_ei = -float(saa.nlp.f(_pack(point, np.zeros((500, 0)))))
    analytic = standard_acquisition(problem, standard_gp, incumbent, hb.EI())
    analytic_ei = -float(analytic.nlp.f(_pack(point)))
    mean, std = hb.gp.predict(standard_gp, point[np.newaxis])
    improvement = np.maximum(incumbent - (mean + std * scenarios.z[:, 0]), 0)
    assert analytic_ei > 1e-3  # a point where EI is far from 0, so the check bites
    assert abs(saa_ei - analytic_ei) < 4 * improvement.std() / np.sqrt(500)
