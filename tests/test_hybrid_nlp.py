"""Var, HybridNLP, its methods and objectives, and Dataset on the illustrative example."""

import casadi as cas
import illustrative_example as ex
import numpy as np
import pytest
from scipy.special import lambertw
from scipy.stats import norm

import hybrid_bo as hb


def training_data() -> hb.Dataset:
    """Five points of the illustrative example."""
    return ex.measure(np.linspace(-1.8, 1.8, 5)[:, None], np.random.default_rng(1))


def surrogate(data: hb.Dataset) -> list[hb.gp.FittedGP]:
    """The GP of y over u."""
    return [ex.gp.fit(data, inputs=[ex.u], output=ex.y, rng=np.random.default_rng(0))]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "1x", "lower": 0, "upper": 1}, "identifier"),
        ({"name": "x", "lower": 1, "upper": 0}, "lower"),
    ],
)
def test_var_rejects(kwargs, match):
    """A bad name, or lower > upper, raise."""
    with pytest.raises(ValueError, match=match):
        hb.Var(**kwargs)


def test_hybrid_nlp_rejects():
    """A u, x or y with no start, a g without one equation per state of x, a vector f,
    or a repeated name raise."""
    with pytest.raises(ValueError, match="start"):
        hb.HybridNLP(u=[hb.Var("w")], y=[ex.y], f=lambda v: v.y)
    with pytest.raises(ValueError, match="one equation per state"):
        hb.HybridNLP(u=[ex.u], x=[ex.x], y=[ex.y], f=lambda v: v.x)
    with pytest.raises(ValueError, match="scalar"):
        hb.HybridNLP(u=[ex.u], y=[ex.y], f=lambda v: [v.y, v.u])
    with pytest.raises(ValueError, match="unique"):
        hb.HybridNLP(u=[ex.u], y=[ex.u], f=lambda v: v.u)


def test_saa_weights_default_to_equal_and_are_checked():
    """w defaults to 1/S, and must be S entries >= 0 that sum to 1."""
    np.testing.assert_array_equal(hb.SAA(np.zeros((4, 1))).w, np.full(4, 0.25))
    with pytest.raises(ValueError, match="sum to 1"):
        hb.SAA(np.zeros((2, 1)), np.array([1.5, -0.5]))
    with pytest.raises(ValueError, match="SAA needs z"):
        hb.SAA(np.zeros(3))


def test_dataset_from_measurements_solves_g():
    """x solves x + exp(x) = y, f is evaluated, and the dataset extends, indexes, saves
    and loads."""
    data = training_data()

    # x + exp(x) = y is x = y - W(exp(y))
    y = np.sin(data.u)
    np.testing.assert_allclose(data.x, y - lambertw(np.exp(y)).real, atol=1e-9)
    np.testing.assert_array_equal(data[ex.y], y[:, 0])
    point = ex.measure(np.array([0.3]), np.random.default_rng(0))
    assert len(data.extend(point).f) == 6
    np.testing.assert_array_equal(data.extend(point).incumbents[-1], data.f.min())


def test_dataset_saves_and_loads(tmp_path):
    """A saved dataset loads equal, with its configuration, and a v0.1.0 file without
    one loads with K = 0."""
    data = hb.Dataset.from_measurements(
        configured_nlp(),
        u=np.zeros((2, 1)),
        c=np.array([[1.0], [2.0]]),
        measured={ex.y: np.zeros(2)},
        rng=np.random.default_rng(0),
    )
    data.save(tmp_path / "data.npz")
    loaded = hb.Dataset.load(tmp_path / "data.npz")
    assert loaded.names == data.names
    np.testing.assert_array_equal(loaded.c, data.c)
    np.testing.assert_array_equal(loaded.f, data.f)

    old = training_data()
    names = np.array(old.names)
    np.savez(tmp_path / "old.npz", u=old.u, x=old.x, y=old.y, f=old.f, names=names)
    assert hb.Dataset.load(tmp_path / "old.npz").c.shape == (5, 0)


def test_a_failed_measurement_keeps_the_other_rows():
    """A row with no root raises, with the rows that worked on the error."""
    with pytest.raises(hb.RootFindingError, match=r"rows \[1\]") as error:
        hb.Dataset.from_measurements(
            ex.nlp,
            u=np.array([[0.0], [0.0]]),
            measured={ex.y: np.array([0.0, np.nan])},
            rng=np.random.default_rng(0),
        )
    assert error.value.dataset is not None and len(error.value.dataset.f) == 1


def test_measuring_x_solves_g_for_y():
    """With x measured, y solves x + exp(x) = y, and matches measuring that y."""
    u, x = np.array([[0.3], [-1.0]]), np.array([0.2, -0.5])
    data = hb.Dataset.from_measurements(
        ex.nlp, u=u, measured={ex.x: x}, rng=np.random.default_rng(0)
    )

    np.testing.assert_allclose(data.y[:, 0], x + np.exp(x), atol=1e-9)
    from_y = hb.Dataset.from_measurements(
        ex.nlp, u=u, measured={ex.y: data.y[:, 0]}, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(from_y.x, data.x, atol=1e-9)
    np.testing.assert_allclose(from_y.f, data.f, atol=1e-9)


@pytest.mark.parametrize(
    ("measured", "match"),
    [
        ({}, "exactly 1"),
        ({ex.x: [0.0], ex.y: [1.0]}, "exactly 1"),
        ({ex.u: [0.0]}, "exactly 1"),
        ({hb.Var("z", 0, 1): [0.0]}, "exactly 1"),
        ({ex.y: [0.0, 1.0]}, r"\(n,\)"),
    ],
)
def test_from_measurements_rejects(measured, match):
    """Measuring other than P states, or the wrong number of rows, raises."""
    with pytest.raises(ValueError, match=match):
        hb.Dataset.from_measurements(
            ex.nlp, u=np.zeros((1, 1)), measured=measured, rng=np.random.default_rng(0)
        )


def configured_nlp() -> hb.HybridNLP:
    """The illustrative example with a configuration a in g and f: x + a exp(x) = y,
    and f scaled by a. At a = 1 it is the illustrative example."""
    a = hb.Var("a")  # known, not chosen: no bounds or start
    assert a.start is None
    return hb.HybridNLP(
        u=[ex.u],
        c=[a],
        x=[ex.x],
        y=[ex.y],
        g=lambda v: [v.x + v.a * cas.exp(v.x) - v.y],
        f=lambda v: v.a * (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4),
    )


def test_configuration_enters_g_and_f():
    """Each row's c enters g and f, and at c = 1 the dataset and the solution are the
    illustrative example's."""
    nlp, u = configured_nlp(), np.array([[0.3], [-1.0]])
    data = hb.Dataset.from_measurements(
        nlp,
        u=u,
        c=np.array([[1.0], [2.0]]),
        measured={ex.y: np.sin(u[:, 0])},
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(data.x + data.c * np.exp(data.x), np.sin(u), atol=1e-9)
    np.testing.assert_array_equal(data[nlp.c[0]], [1.0, 2.0])
    plain = ex.measure(u, np.random.default_rng(0))
    np.testing.assert_allclose(data.x[0], plain.x[0], atol=1e-9)
    x = data.x[1, 0]
    assert data.f[1] == pytest.approx(2 * (6 * x - 2) ** 2 * np.sin(12 * x - 4))

    saa, fitted = hb.SAA(np.array([[0.0], [1.0]])), surrogate(training_data())
    solution = nlp.solve(
        saa, np.random.default_rng(2), surrogate=fitted, c=np.array([1.0])
    )
    reference = ex.nlp.solve(saa, np.random.default_rng(2), surrogate=fitted)
    np.testing.assert_allclose(solution.u, reference.u, atol=1e-6)
    np.testing.assert_array_equal(solution.c, [1.0])
    at_two = nlp.solve(
        saa, np.random.default_rng(2), surrogate=fitted, c=np.array([2.0])
    )
    np.testing.assert_allclose(at_two.x + 2 * np.exp(at_two.x), at_two.y, atol=1e-6)


def test_configuration_is_checked():
    """A missing c, a c of the wrong shape or outside its bounds, or a c for a problem
    with none raise."""
    bounded = hb.HybridNLP(
        u=[ex.u], c=[hb.Var("a", 0, 1)], y=[ex.y], f=lambda v: v.y + v.a
    )
    saa, fitted = hb.SAA(np.zeros((1, 1))), surrogate(training_data())
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="K=1"):
        bounded.solve(saa, rng, surrogate=fitted)
    with pytest.raises(ValueError, match="K=1"):
        bounded.solve(saa, rng, surrogate=fitted, c=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="bounds"):
        bounded.solve(saa, rng, surrogate=fitted, c=np.array([2.0]))
    with pytest.raises(ValueError, match="K=0"):
        ex.nlp.solve(saa, rng, surrogate=fitted, c=np.array([1.0]))
    with pytest.raises(ValueError, match="K=1"):
        hb.Dataset.from_measurements(
            bounded, u=np.zeros((1, 1)), measured={ex.y: np.zeros(1)}, rng=rng
        )


@pytest.mark.parametrize("unknown_states_as_variables", [True, False])
def test_saa_solution_satisfies_g_and_the_gp(unknown_states_as_variables):
    """Each scenario's states satisfy g, and y is the GP's prediction there."""
    data = training_data()
    saa = hb.SAA(np.random.default_rng(3).standard_normal((4, 1)))

    solution = ex.nlp.solve(
        saa,
        np.random.default_rng(2),
        surrogate=surrogate(data),
        objective=hb.obj.EI(data.incumbents[-1]),
        unknown_states_as_variables=unknown_states_as_variables,
    )

    assert solution.x.shape == solution.y.shape == (4, 1)
    np.testing.assert_allclose(solution.x + np.exp(solution.x), solution.y, atol=1e-6)
    mean, std = surrogate(data)[0].predict(solution.u[None])
    np.testing.assert_allclose(solution.y, mean + std * saa.z, atol=1e-6)


def test_solve_needs_one_gp_per_unknown_state():
    """A surrogate without a GP of each unknown state raises."""
    with pytest.raises(ValueError, match="one GP per unknown state"):
        ex.nlp.solve(hb.SAA(np.zeros((1, 1))), np.random.default_rng(0), surrogate=[])


def test_exact_ei_is_the_closed_form():
    """Standard BO: Exact with EI minimizes -EI of the GP of f."""
    data = training_data()
    observed = hb.Dataset.from_measurements(
        ex.standard_nlp,
        u=data.u,
        measured={ex.fhat: data.f},
        rng=np.random.default_rng(0),
    )
    fitted = ex.gp.fit(
        observed, inputs=[ex.u], output=ex.fhat, rng=np.random.default_rng(0)
    )
    incumbent = data.incumbents[-1]

    solution = ex.standard_nlp.solve(
        hb.Exact(),
        np.random.default_rng(2),
        surrogate=[fitted],
        objective=hb.obj.EI(incumbent),
    )

    mean, std = fitted.predict(solution.u[None])
    gamma = (incumbent - mean[0]) / std[0]
    ei = (incumbent - mean[0]) * norm.cdf(gamma) + std[0] * norm.pdf(gamma)
    assert solution.value == pytest.approx(-ei, rel=1e-6)
    np.testing.assert_allclose(solution.y, [[mean[0]]])


def test_exact_rejects_what_has_no_closed_form():
    """Exact raises for mechanistic states, and for SoftEI."""
    data = training_data()
    with pytest.raises(ValueError, match="Exact needs"):
        ex.nlp.solve(hb.Exact(), np.random.default_rng(0), surrogate=surrogate(data))
    fhat_data = hb.Dataset.from_measurements(
        ex.standard_nlp,
        u=data.u,
        measured={ex.fhat: data.f},
        rng=np.random.default_rng(0),
    )
    fitted = ex.gp.fit(
        fhat_data, inputs=[ex.u], output=ex.fhat, rng=np.random.default_rng(0)
    )
    with pytest.raises(ValueError, match="no exact form"):
        ex.standard_nlp.solve(
            hb.Exact(),
            np.random.default_rng(0),
            surrogate=[fitted],
            objective=hb.obj.SoftEI(0.0, temperature=0.1),
        )


def test_objectives_over_scenarios():
    """Mean, EI and SoftEI of f under weighted scenarios match numpy."""
    f, w = np.array([-1.0, 0.5, 2.0]), np.array([0.2, 0.3, 0.5])

    def value(objective):
        """The objective at f and w, as a number."""
        return float(cas.evalf(objective.saa(cas.DM(f), w)))

    assert value(hb.obj.Mean()) == pytest.approx(w @ f)
    assert value(hb.obj.EI(1.0)) == pytest.approx(w @ np.minimum(f - 1.0, 0))
    smooth = 0.5 * np.log1p(np.exp((1.0 - f) / 0.5))
    assert value(hb.obj.SoftEI(1.0, temperature=0.5)) == pytest.approx(
        -np.log(w @ smooth)
    )
    with pytest.raises(ValueError, match="temperature"):
        hb.obj.SoftEI(0.0, temperature=0.0)
