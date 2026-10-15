"""Var, System and Problem: building the CasADi functions and checking the inputs.

pytest runs every function whose name starts with `test_`. A test argument named like
a fixture in `conftest.py` (here `illustrative_problem`) is filled in by pytest with
that fixture's value.
"""

from collections.abc import Callable, Sequence
from typing import Any

import casadi as cas
import numpy as np
import pytest

from hybrid_bo import Problem, System, Var, initial_points
from hybrid_bo.sampling import lhs


def test_illustrative_equations_take_x_y_u(illustrative_problem):
    """The traced g and f of the illustrative example take their inputs in the order
    (x, y, u) and compute the example's equations."""
    problem = illustrative_problem
    rng = np.random.default_rng(0)

    for u, x, y in rng.uniform(-2.0, 2.0, size=(10, 3)):
        np.testing.assert_allclose(problem.system.g(x, y, u).full(), x + np.exp(x) - y)
        np.testing.assert_allclose(
            problem.f(x, y, u).full(), (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        )


def test_flash_unit_shaped_system():
    """A system shaped like the flash unit, with two decision variables, traces g
    correctly and counts its variables."""
    # h takes a decision variable and a mechanistic state; the sensor reads a state
    F, D, z1 = 1.0, 0.3, 0.5

    def p1_sat(T):
        return 10 ** (10.19590302 - 1730.6 / (-39.75 + T * 1000)) / 1e5

    T, p = Var("T", 0.36315, 0.40315), Var("p", 0.8, 2.6)
    x1, ln_gamma1 = Var("x1", 0.0, 1.0), Var("ln_gamma1", start=(-0.7, 0.7))

    def g(v):
        vapor_fraction = (F * z1 - (F - D) * v.x1) / D
        return [v.p * vapor_fraction - p1_sat(v.T) * v.x1 * cas.exp(v.ln_gamma1)]

    system = System(
        u=[T, p], x=[x1], y=[ln_gamma1], g=g, h_inputs=[T, x1], measured=[x1]
    )

    # g at one point, by hand and traced. `pytest.approx` compares with a tolerance
    T_, p_, x1_, ln_gamma1_ = 0.38, 1.2, 0.4, 0.1
    vapor_fraction = (F * z1 - (F - D) * x1_) / D
    expected = p_ * vapor_fraction - p1_sat(T_) * x1_ * np.exp(ln_gamma1_)
    actual = system.g(x1_, ln_gamma1_, [T_, p_]).full().item()
    assert actual == pytest.approx(expected, rel=1e-12)
    assert (system.n_u, system.n_x, system.n_y) == (2, 1, 1)


def test_var_start_defaults_to_finite_bounds():
    """A Var's start bounds default to its bounds when they are finite."""
    assert Var("u", -2.0, 2.0).start == (-2.0, 2.0)
    assert Var("x", start=(0.0, 1.0)).start == (0.0, 1.0)


# `parametrize` runs the test once per entry of the list, each as its own test case
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "2x"}, "identifier"),
        ({"name": 3}, "identifier"),
        ({"name": "lambda", "lower": 0.0, "upper": 1.0}, "identifier"),
        ({"name": "_symbols", "lower": 0.0, "upper": 1.0}, "start with '_'"),
        ({"name": "x", "lower": 1.0, "upper": 0.0}, "lower"),
        ({"name": "x"}, "infinite bound"),
        ({"name": "x", "lower": 0.0, "start": (0.0, np.inf)}, "finite"),
        ({"name": "x", "lower": 0.0, "upper": 1.0, "start": (-1.0, 1.0)}, "inside"),
        ({"name": "x", "lower": 0.0, "upper": 1.0, "start": (0.8, 0.2)}, "ordered"),
    ],
)
def test_var_rejects(kwargs, match):
    """Var raises ValueError for each invalid argument, with a message saying what is
    wrong."""
    # `pytest.raises` fails the test unless the block raises that exception, with a
    # message matching `match`
    with pytest.raises(ValueError, match=match):
        Var(**kwargs)


U, X, Y, Z = (Var(n, 0.0, 1.0) for n in ["u", "x", "y", "z"])


def system(
    u: Sequence[Var] = (U,),
    x: Sequence[Var] = (X,),
    y: Sequence[Var] = (Y,),
    g: Callable[[Any], Any] = lambda v: [v.x - v.y],
    h_inputs: Sequence[Var] = (U,),
    measured: Sequence[Var] = (Y,),
) -> System:
    """A valid one-of-each system; tests override one argument to break it."""
    return System(u, x, y, g, h_inputs, measured)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"u": []}, "at least one variable in u"),
        ({"x": []}, "at least one variable in x"),
        ({"y": []}, "at least one variable in y"),
        ({"x": [X, Var("x", 0.0, 1.0)]}, "u, x and y together must not repeat .*: x"),
        ({"g": lambda v: [v.x, v.y]}, "one equation per mechanistic state"),
        ({"h_inputs": [Y]}, "h_inputs contains 'y'"),
        ({"h_inputs": [U, U]}, "h_inputs must not repeat a variable; repeated: u"),
        ({"measured": [U]}, "measured contains 'u'"),
        ({"measured": [Y, Y]}, "measured must not repeat a variable; repeated: y"),
        ({"measured": [X, Y]}, "as many measured variables"),
    ],
)
def test_system_rejects(overrides, match):
    """System raises ValueError for each invalid argument, with a message saying what
    is wrong."""
    with pytest.raises(ValueError, match=match):
        system(**overrides)


def test_more_than_one_unknown_state_is_not_supported_yet():
    """A system with two unknown states raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="one unknown state"):
        system(y=[Y, Z], measured=[Y, Z])


def test_unknown_name_in_a_callable():
    """Using a name that isn't a variable in g raises, listing the available
    ones."""
    with pytest.raises(
        AttributeError, match="Unknown variable 'w'; available: x, y, u"
    ):
        system(g=lambda v: [v.w])


@pytest.mark.parametrize(
    "g", [lambda v: None, lambda v: [[v.x - v.y]], lambda v: "x - y"]
)
def test_callable_output_that_is_not_an_expression(g):
    """A g that returns something other than a list of CasADi expressions raises
    TypeError."""
    with pytest.raises(TypeError, match="g must return a CasADi expression"):
        system(g=g)


def test_problem_rejects_a_non_scalar_objective():
    """Problem raises when f returns more than one value."""
    with pytest.raises(ValueError, match="f must be a scalar"):
        Problem(system(), f=lambda v: [v.x, v.y])


def test_initial_points_is_lhs_over_the_start_bounds():
    """initial_points is a Latin hypercube over the start bounds of the variables."""
    variables = [Var("a", -2.0, 2.0), Var("b", start=(0.0, 5.0))]

    points = initial_points(variables, 4, np.random.default_rng(0))

    expected = lhs(
        np.array([-2.0, 0.0]), np.array([2.0, 5.0]), 4, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(points, expected)
