"""Evaluation: measurements, the oracle simulation, the root-finding problem, History.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value; `monkeypatch` is a fixture built into pytest.
"""

import casadi as cas
import numpy as np
import pytest
from scipy.special import lambertw

import hybrid_bo
from hybrid_bo import (
    History,
    Problem,
    ReconstructionWarning,
    RootFindingError,
    System,
    Var,
    incumbents,
    oracle_incumbents,
    oracle_measure,
)
from hybrid_bo.evaluation import OracleMeasurement, evaluate
from hybrid_bo.history import n_iterations


def illustrative_measure(problem: Problem, noise_std: float):
    """The oracle simulation of the illustrative example, with Gaussian noise of
    `noise_std` on the measurement."""
    return oracle_measure(
        problem.system, oracle=lambda v: [cas.sin(v.u)], noise_std=noise_std
    )


def two_branch_problem() -> Problem:
    """y = x**2 with y measured: x = +sqrt(y) and x = -sqrt(y) are both roots.

    The start bounds of x only cover the positive branch.
    """
    u = Var("u", -1.0, 1.0)
    x = Var("x", start=(0.5, 2.0))
    y = Var("y", start=(0.0, 4.0))
    system = System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x**2 - v.y],
        h_inputs=[u],
        measured=[y],
    )
    return Problem(system, f=lambda v: v.x + v.u)


def closed_form(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The illustrative example solved by hand, for u (n,): y = sin(u), and
    x + exp(x) = y gives x = y - W(exp(y)) (Lambert W)."""
    y = np.sin(u)
    x = y - lambertw(np.exp(y)).real
    return x, y, (6 * x - 2) ** 2 * np.sin(12 * x - 4)


# `filterwarnings("error")` turns warnings into failures, so the test also checks
# that the reconstruction agrees with the oracle
@pytest.mark.filterwarnings("error")
def test_noise_free_evaluation_matches_the_closed_form(illustrative_problem):
    """Without noise, the oracle values and the reconstruction both solve the
    illustrative example exactly."""
    # Trial points across the bounds of u
    u = np.array([-2.0, -0.5, 0.0, 1.0, 2.0])
    measure = illustrative_measure(illustrative_problem, noise_std=0.0)

    history = evaluate(
        illustrative_problem, measure, u[:, np.newaxis], np.random.default_rng(0)
    )

    # Measured, reconstructed and oracle values all match the closed form
    x, y, f = closed_form(u)
    assert history.x_oracle is not None and history.y_oracle is not None
    assert history.f_oracle is not None
    for actual, expected in [
        (history.x, x),
        (history.y, y),
        (history.x_oracle, x),
        (history.y_oracle, y),
    ]:
        np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(history.f, f, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(history.f_oracle, f, rtol=1e-8, atol=1e-10)
    assert history.n_initial == len(u)


def test_noise_is_the_first_stream_draw(illustrative_problem):
    """The measurement noise is the first draw from the evaluation's generator."""
    # The root-finding starts come from lhs, which spawns instead of consuming the
    # stream
    measure = illustrative_measure(illustrative_problem, noise_std=0.1)

    history = evaluate(
        illustrative_problem, measure, np.array([[-0.5]]), np.random.default_rng(0)
    )

    noise = np.random.default_rng(0).normal(0.0, 0.1, size=1)
    assert history.y_oracle is not None
    np.testing.assert_allclose(history.y - history.y_oracle, [noise], rtol=1e-12)


def test_root_finding_starts_cover_the_unmeasured_variables(monkeypatch):
    """The root-finding starts are a Latin hypercube over the start bounds of the
    variables that are not measured, and the root found solves g."""
    # Two mechanistic states and one measured unknown state: the starts are a
    # two-dimensional Latin hypercube over the start bounds of x1 and x2
    u = Var("u", -1.0, 1.0)
    x1, x2 = Var("x1", start=(0.0, 1.0)), Var("x2", start=(2.0, 3.0))
    y = Var("y", start=(-1.0, 1.0))
    system = System(
        u=[u],
        x=[x1, x2],
        y=[y],
        g=lambda v: [v.x1 - v.y - v.u, v.x2 - v.x1**2],
        h_inputs=[u],
        measured=[y],
    )
    problem = Problem(system, f=lambda v: v.x2)

    # Record the arguments of each lhs call. `monkeypatch.setattr` replaces lhs in
    # the evaluation module for this test only
    calls = []

    def spy(lower, upper, n, rng):
        calls.append((lower, upper, n))
        return hybrid_bo.sampling.lhs(lower, upper, n, rng)

    monkeypatch.setattr(hybrid_bo.evaluation, "lhs", spy)
    history = evaluate(
        problem,
        lambda u, rng: np.array([0.5]),
        np.array([[0.25]]),
        np.random.default_rng(0),
        n_starts=7,
    )

    # One lhs call over the start bounds of x1 and x2, and the root x1 = y + u
    [(lower, upper, n)] = calls
    np.testing.assert_array_equal(lower, [0.0, 2.0])
    np.testing.assert_array_equal(upper, [1.0, 3.0])
    assert n == 7
    np.testing.assert_allclose(history.x, [[0.75, 0.75**2]])
    np.testing.assert_allclose(history.f, [0.75**2])


def test_measure_without_an_oracle_gives_no_oracle_values():
    """A measurement that returns only the measured variables gives reconstructed
    values and no oracle values."""
    problem = two_branch_problem()

    history = evaluate(
        problem,
        lambda u, rng: np.array([u[0] + 2.0]),
        np.array([[-1.0], [0.0]]),
        np.random.default_rng(0),
    )

    np.testing.assert_allclose(history.x, [[1.0], [np.sqrt(2.0)]])
    np.testing.assert_allclose(history.y, [[1.0], [2.0]])
    np.testing.assert_allclose(history.f, [0.0, np.sqrt(2.0)])
    assert history.x_oracle is None
    assert history.y_oracle is None
    assert history.f_oracle is None


def test_disagreement_with_the_oracle_warns():
    """A reconstruction on a different branch from the oracle values warns, and both
    are kept."""

    # The oracle is on the negative branch; the reconstruction can only find the
    # positive one
    def measure(u, rng):
        return OracleMeasurement(
            measured=np.array([1.0]), x=np.array([-1.0]), y=np.array([1.0])
        )

    # `pytest.warns` fails the test unless the block warns with a message matching
    # `match`
    with pytest.warns(ReconstructionWarning, match="oracle"):
        history = evaluate(
            two_branch_problem(),
            measure,
            np.array([[0.0]]),
            np.random.default_rng(0),
        )

    np.testing.assert_allclose(history.x, [[1.0]])
    assert history.x_oracle is not None and history.f_oracle is not None
    np.testing.assert_allclose(history.x_oracle, [[-1.0]])
    np.testing.assert_allclose(history.f_oracle, [-1.0])


@pytest.mark.filterwarnings("error")
def test_noisy_reconstruction_on_the_oracle_branch_does_not_warn():
    """A reconstruction moved by noise, but on the oracle's branch, does not warn."""

    # Noise moves the reconstruction away from the oracle values, but along the
    # same branch
    def measure(u, rng):
        return OracleMeasurement(
            measured=np.array([1.1]), x=np.array([1.0]), y=np.array([1.0])
        )

    history = evaluate(
        two_branch_problem(), measure, np.array([[0.0]]), np.random.default_rng(0)
    )

    np.testing.assert_allclose(history.x, [[np.sqrt(1.1)]])


def test_failed_root_finding_raises_with_the_points_before():
    """When the root-finding fails, the error carries the history of the points
    evaluated before."""

    # x**2 = -1 has no real root
    def measure(u, rng):
        return np.array([1.0]) if u[0] < 0 else np.array([-1.0])

    # `pytest.raises` fails the test unless the block raises that exception; `info`
    # holds the exception
    with pytest.raises(RootFindingError) as info:
        evaluate(
            two_branch_problem(),
            measure,
            np.array([[-0.5], [0.5], [-0.5]]),
            np.random.default_rng(0),
            n_starts=5,
        )

    # Only the first point, before the failing second one
    history = info.value.history
    assert history is not None
    np.testing.assert_array_equal(history.u, [[-0.5]])
    np.testing.assert_allclose(history.x, [[1.0]])


def test_failed_oracle_simulation_raises():
    """When the oracle simulation has no solution, evaluation raises with an empty
    history."""
    # y = -1 - x**2 and y = x**2 have no common real root
    problem = two_branch_problem()
    measure = oracle_measure(problem.system, oracle=lambda v: -1 - v.x**2, n_starts=5)

    with pytest.raises(RootFindingError) as info:
        evaluate(problem, measure, np.array([[0.0]]), np.random.default_rng(0))

    assert info.value.history is not None
    assert info.value.history.u.shape == (0, 1)


def test_measure_returning_the_wrong_shape_raises():
    """A measurement returning more values than there are measured variables raises."""
    with pytest.raises(ValueError, match="measured"):
        evaluate(
            two_branch_problem(),
            lambda u, rng: np.array([1.0, 2.0]),
            np.array([[0.0]]),
            np.random.default_rng(0),
        )


def test_oracle_must_give_one_value_per_unknown_state():
    """An oracle with more expressions than unknown states is rejected."""
    with pytest.raises(ValueError, match="oracle"):
        oracle_measure(two_branch_problem().system, oracle=lambda v: [v.u, 2 * v.u])


def history_of(f: list[float], f_oracle: list[float] | None = None) -> History:
    """A history with the given measured and oracle objective values, the first two
    of them initial points; the other fields are zero."""
    n = len(f)
    return History(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 1)),
        y=np.zeros((n, 1)),
        f=np.array(f),
        x_oracle=None if f_oracle is None else np.zeros((n, 1)),
        y_oracle=None if f_oracle is None else np.zeros((n, 1)),
        f_oracle=None if f_oracle is None else np.array(f_oracle),
        n_initial=2,
    )


def test_n_iterations_counts_the_points_after_the_initial_ones():
    """A history of five points, two of them initial, has three iterations."""
    assert n_iterations(history_of([3.0, 1.0, 2.0, 0.5, 0.2])) == 3


def test_incumbents_are_the_running_minimum_of_measured_f():
    """The incumbent after each evaluation is the best measured f so far."""
    history = history_of([3.0, 1.0, 2.0, 0.5])

    np.testing.assert_array_equal(incumbents(history), [3.0, 1.0, 1.0, 0.5])


def test_oracle_incumbents_are_the_oracle_f_at_the_measured_incumbent():
    """Under noise, the benchmark metric follows the point the measured f ranks best,
    not the best oracle f so far."""
    # The third point measures worse than the second but is better at the oracle;
    # the fourth measures best but is worse at the oracle
    history = history_of(f=[3.0, 1.0, 2.0, 0.5], f_oracle=[3.0, 1.5, 0.5, 2.0])

    np.testing.assert_array_equal(oracle_incumbents(history), [3.0, 1.5, 1.5, 2.0])


def test_oracle_incumbents_keep_the_earlier_point_on_a_tie():
    """A later point that only equals the incumbent's measured f doesn't replace it."""
    history = history_of(f=[1.0, 1.0, 1.0], f_oracle=[2.0, 0.0, 3.0])

    np.testing.assert_array_equal(oracle_incumbents(history), [2.0, 2.0, 2.0])


def test_oracle_incumbents_without_noise_are_the_running_minimum_of_oracle_f(
    illustrative_problem,
):
    """Without noise the measured and oracle f agree, so the benchmark metric is the
    running minimum of the oracle f (the paper's convergence figures)."""
    u = np.array([[2.0], [-0.5], [1.0], [0.0], [-2.0], [0.4]])
    measure = illustrative_measure(illustrative_problem, noise_std=0.0)

    history = evaluate(illustrative_problem, measure, u, np.random.default_rng(0))

    assert history.f_oracle is not None
    np.testing.assert_allclose(
        oracle_incumbents(history), np.minimum.accumulate(history.f_oracle)
    )


def test_oracle_incumbents_need_oracle_values():
    """A history from a measurement without an oracle has no benchmark metric."""
    with pytest.raises(ValueError, match="oracle"):
        oracle_incumbents(history_of([3.0, 1.0]))
