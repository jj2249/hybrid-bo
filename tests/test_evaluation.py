"""Evaluation: experiments, the oracle simulation, the root-finding problem, History."""

import casadi as cas
import numpy as np
import pytest
from scipy.special import lambertw

import hybrid_bo
from hybrid_bo import (
    History,
    OracleMeasurement,
    Problem,
    ReconstructionWarning,
    RootFindingError,
    System,
    Var,
    evaluate,
    incumbents,
    oracle_experiment,
    oracle_incumbents,
)


def illustrative_experiment(problem: Problem, noise_std: float):
    return oracle_experiment(
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
    experiment = illustrative_experiment(illustrative_problem, noise_std=0.0)

    history = evaluate(
        illustrative_problem, experiment, u[:, np.newaxis], np.random.default_rng(0)
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
    # The root-finding starts come from lhs, which spawns instead of consuming the
    # stream, so the noise is the evaluation's first stream draw
    experiment = illustrative_experiment(illustrative_problem, noise_std=0.1)

    history = evaluate(
        illustrative_problem, experiment, np.array([[-0.5]]), np.random.default_rng(0)
    )

    noise = np.random.default_rng(0).normal(0.0, 0.1, size=1)
    assert history.y_oracle is not None
    np.testing.assert_allclose(history.y - history.y_oracle, [noise], rtol=1e-12)


def test_root_finding_starts_cover_the_unmeasured_variables(monkeypatch):
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

    [(lower, upper, n)] = calls
    np.testing.assert_array_equal(lower, [0.0, 2.0])
    np.testing.assert_array_equal(upper, [1.0, 3.0])
    assert n == 7
    np.testing.assert_allclose(history.x, [[0.75, 0.75**2]])
    np.testing.assert_allclose(history.f, [0.75**2])


def test_experiment_without_an_oracle_gives_no_oracle_values():
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
    # The oracle is on the negative branch; the reconstruction can only find the
    # positive one
    def experiment(u, rng):
        return OracleMeasurement(
            measured=np.array([1.0]), x=np.array([-1.0]), y=np.array([1.0])
        )

    with pytest.warns(ReconstructionWarning, match="oracle"):
        history = evaluate(
            two_branch_problem(),
            experiment,
            np.array([[0.0]]),
            np.random.default_rng(0),
        )

    np.testing.assert_allclose(history.x, [[1.0]])
    assert history.x_oracle is not None and history.f_oracle is not None
    np.testing.assert_allclose(history.x_oracle, [[-1.0]])
    np.testing.assert_allclose(history.f_oracle, [-1.0])


@pytest.mark.filterwarnings("error")
def test_noisy_reconstruction_on_the_oracle_branch_does_not_warn():
    # Noise moves the reconstruction away from the oracle values, but along the
    # same branch
    def experiment(u, rng):
        return OracleMeasurement(
            measured=np.array([1.1]), x=np.array([1.0]), y=np.array([1.0])
        )

    history = evaluate(
        two_branch_problem(), experiment, np.array([[0.0]]), np.random.default_rng(0)
    )

    np.testing.assert_allclose(history.x, [[np.sqrt(1.1)]])


def test_failed_root_finding_raises_with_the_points_before():
    # x**2 = -1 has no real root
    def experiment(u, rng):
        return np.array([1.0]) if u[0] < 0 else np.array([-1.0])

    with pytest.raises(RootFindingError) as info:
        evaluate(
            two_branch_problem(),
            experiment,
            np.array([[-0.5], [0.5], [-0.5]]),
            np.random.default_rng(0),
            n_starts=5,
        )

    history = info.value.history
    assert history is not None
    np.testing.assert_array_equal(history.u, [[-0.5]])
    np.testing.assert_allclose(history.x, [[1.0]])


def test_failed_oracle_simulation_raises():
    # y = -1 - x**2 and y = x**2 have no common real root
    problem = two_branch_problem()
    experiment = oracle_experiment(
        problem.system, oracle=lambda v: -1 - v.x**2, n_starts=5
    )

    with pytest.raises(RootFindingError) as info:
        evaluate(problem, experiment, np.array([[0.0]]), np.random.default_rng(0))

    assert info.value.history is not None
    assert info.value.history.u.shape == (0, 1)


def test_experiment_returning_the_wrong_shape_raises():
    with pytest.raises(ValueError, match="measured"):
        evaluate(
            two_branch_problem(),
            lambda u, rng: np.array([1.0, 2.0]),
            np.array([[0.0]]),
            np.random.default_rng(0),
        )


def test_oracle_must_give_one_value_per_unknown_state():
    with pytest.raises(ValueError, match="oracle"):
        oracle_experiment(two_branch_problem().system, oracle=lambda v: [v.u, 2 * v.u])


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
    experiment = illustrative_experiment(illustrative_problem, noise_std=0.0)

    history = evaluate(illustrative_problem, experiment, u, np.random.default_rng(0))

    assert history.f_oracle is not None
    np.testing.assert_allclose(
        oracle_incumbents(history), np.minimum.accumulate(history.f_oracle)
    )


def test_oracle_incumbents_need_oracle_values():
    """A history from an experiment without an oracle has no benchmark metric."""
    with pytest.raises(ValueError, match="oracle"):
        oracle_incumbents(history_of([3.0, 1.0]))
