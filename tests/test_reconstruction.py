"""Reconstruction: the root-finding problem, and the `Dataset` it gives.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

import dataclasses
from copy import deepcopy

import numpy as np
import pytest
from scipy.special import lambertw

from hybrid_bo import (
    Dataset,
    Problem,
    RootFindingError,
    System,
    Var,
    extend,
    incumbents,
)
from hybrid_bo.evaluation import reconstruct_states


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


def no_known_equations_problem() -> Problem:
    """y measured and no mechanistic states, so the root-finding problem is empty."""
    u, y = Var("u", -1.0, 1.0), Var("y", start=(-1.0, 1.0))
    system = System(u=[u], y=[y], h_inputs=[u], measured=[y])
    return Problem(system, f=lambda v: v.y)


def measured_outside_g_problem() -> Problem:
    """x = u, with y measured but in no known equation: only f sees it, so the
    root-finding problem has a root whatever y is."""
    u = Var("u", -1.0, 1.0)
    x = Var("x", start=(-1.0, 1.0))
    y = Var("y", start=(-1.0, 1.0))
    system = System(
        u=[u], x=[x], y=[y], g=lambda v: [v.x - v.u], h_inputs=[u], measured=[y]
    )
    return Problem(system, f=lambda v: v.x + v.y)


def dataset_of(f: list[float]) -> Dataset:
    """A dataset with the given measured objective values; the other fields are zero."""
    n = len(f)
    return Dataset(
        u=np.zeros((n, 1)),
        x=np.zeros((n, 1)),
        y=np.zeros((n, 1)),
        f=np.array(f),
    )


def test_extend_joins_the_rows_in_order():
    """The rows of the first dataset come first, then those of the second, in every
    field."""
    # Two rows, then one row whose u tells it apart
    first = dataset_of([3.0, 1.0])
    new = dataclasses.replace(dataset_of([2.0]), u=np.array([[7.0]]))

    extended = extend(first, new)

    # The first dataset's rows, then the second's
    np.testing.assert_array_equal(extended.f, [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(extended.u, [[0.0], [0.0], [7.0]])
    assert extended.x.shape == extended.y.shape == (3, 1)


def test_extend_leaves_its_inputs_alone():
    """The inputs keep their values, and the result shares no array with them, so
    writing to it can't reach them."""
    # Two datasets with every field, and copies to compare them with afterwards
    first = dataset_of([3.0, 1.0])
    new = dataset_of([2.0])
    first_before, new_before = deepcopy(first), deepcopy(new)
    fields = ("u", "x", "y", "f")

    # Extend, then write to the result
    extended = extend(first, new)
    extended.f[:] = -1.0

    # Every input keeps its values and shares no memory with the result
    for after, before in ((first, first_before), (new, new_before)):
        for field in fields:
            np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
            assert not np.shares_memory(getattr(extended, field), getattr(after, field))


def test_incumbents_are_the_running_minimum_of_measured_f():
    """The incumbent after each evaluation is the best measured f so far."""
    dataset = dataset_of([3.0, 1.0, 2.0, 0.5])

    np.testing.assert_array_equal(incumbents(dataset), [3.0, 1.0, 1.0, 0.5])


def test_reconstruct_states_solves_the_unmeasured_states_of_given_measurements(
    illustrative_problem,
):
    """Given u and y, `reconstruct_states` gives x from x + exp(x) = y, in closed form
    with the Lambert W function, and f at those states, calling no measurement."""
    # Three measurements of y
    u = np.array([[-1.0], [0.0], [1.0]])
    measured = np.array([[-0.5], [0.3], [1.0]])

    dataset = reconstruct_states(
        illustrative_problem, u=u, measured=measured, rng=np.random.default_rng(0)
    )

    # The closed form of the unmeasured state, and f there
    x = measured - lambertw(np.exp(measured)).real
    f = ((6 * x - 2) ** 2 * np.sin(12 * x - 4)).ravel()
    np.testing.assert_allclose(dataset.x, x, atol=1e-8)
    np.testing.assert_array_equal(dataset.y, measured)
    np.testing.assert_array_equal(dataset.u, u)
    np.testing.assert_allclose(dataset.f, f, atol=1e-6)


# `parametrize` runs the test once per entry, with the entry as the argument
@pytest.mark.parametrize(
    ("u", "measured", "message"),
    [
        (np.zeros(4), np.zeros((4, 1)), "u need shape"),
        (np.zeros((4, 2)), np.zeros((4, 1)), "u need shape"),
        (np.zeros((4, 1)), np.zeros(4), "measured variables need shape"),
        (np.zeros((4, 1)), np.zeros((3, 1)), "measured variables need shape"),
    ],
    ids=["flat u", "too many u", "flat measured", "fewer measured"],
)
def test_reconstruct_states_rejects_malformed_measurements(
    illustrative_problem, u, measured, message
):
    """Measurements need a row per point and a column per variable."""
    with pytest.raises(ValueError, match=message):
        reconstruct_states(
            illustrative_problem, u=u, measured=measured, rng=np.random.default_rng(0)
        )


def square_root_problem() -> Problem:
    """x^2 = y with x in [0, 3]: y < 0 has no root and y > 9 has none within the bounds."""
    u_var, x_var = Var("u", -1.0, 1.0), Var("x", 0.0, 3.0)
    y_var = Var("y", start=(-2.0, 2.0))
    system = System(
        u=[u_var],
        x=[x_var],
        y=[y_var],
        g=lambda v: [v.x**2 - v.y],
        h_inputs=[u_var],
        measured=[y_var],
    )
    return Problem(system, f=lambda v: v.x)


def test_reconstruct_states_names_every_bad_row_and_keeps_the_others():
    """Every row is tried, one error names the bad rows, and `error.dataset` holds the
    rows that worked. A row is bad with no root, with a root outside the bounds of the
    unmeasured states, or with a value that isn't finite."""
    # Good, no root, good, root outside the bounds, not finite
    problem = square_root_problem()
    u = np.array([[-0.5], [0.0], [0.25], [0.5], [0.75]])
    measured = np.array([[1.0], [-1.0], [4.0], [16.0], [np.nan]])

    with pytest.raises(RootFindingError, match=r"rows \[1, 3, 4\]") as caught:
        reconstruct_states(
            problem, u=u, measured=measured, rng=np.random.default_rng(0)
        )

    # The two good rows, in order, with their roots
    dataset = caught.value.dataset
    assert dataset is not None
    np.testing.assert_array_equal(dataset.u, u[[0, 2]])
    np.testing.assert_allclose(dataset.x, [[1.0], [2.0]], atol=1e-8)


def test_a_reconstruction_with_no_good_rows_has_an_empty_dataset():
    """When every row fails, `error.dataset` has no points."""
    problem = square_root_problem()
    measured = np.array([[-1.0], [-2.0]])

    with pytest.raises(RootFindingError, match=r"rows \[0, 1\]") as caught:
        reconstruct_states(
            problem, u=np.zeros((2, 1)), measured=measured, rng=np.random.default_rng(0)
        )

    dataset = caught.value.dataset
    assert dataset is not None
    assert len(dataset.u) == 0


def test_a_row_that_is_not_finite_is_a_bad_row_with_no_known_equations():
    """With M = 0 nothing is solved, so no root can fail: a NaN is still a bad row."""
    u_var, y_var = Var("u", -1.0, 1.0), Var("y", -1.0, 1.0)
    system = System(u=[u_var], y=[y_var], h_inputs=[u_var], measured=[y_var])
    problem = Problem(system, f=lambda v: v.y)
    measured = np.array([[0.5], [np.nan]])

    with pytest.raises(RootFindingError, match=r"rows \[1\]") as caught:
        reconstruct_states(
            problem, u=np.zeros((2, 1)), measured=measured, rng=np.random.default_rng(0)
        )

    dataset = caught.value.dataset
    assert dataset is not None
    np.testing.assert_array_equal(dataset.y, [[0.5]])
