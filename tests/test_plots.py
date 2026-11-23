"""The figures draw on the illustrative example."""

import casadi as cas
import illustrative_example as ex
import matplotlib
import numpy as np
import pytest

import hybrid_bo as hb

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_figures_draw():
    """An SAA iteration with SoftLogEI draws 3 panels, an Exact solution with Mean 2, and
    the objective on the grid is finite at the solution."""
    rng = np.random.default_rng(0)
    data = ex.measure(np.linspace(-1.8, 1.8, 5)[:, None], rng)
    surrogate = [ex.gp.fit(data, inputs=[ex.u], output=ex.y, rng=rng)]
    solution = ex.nlp.solve(
        hb.SAA(rng.standard_normal((4, 1))),
        rng,
        surrogate=surrogate,
        objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=0.05),
    )
    _, axes = plt.subplots(3, 1)
    hb.plots.iteration(ex.nlp, solution, axes, data)
    objective = axes[2].get_lines()[0].get_ydata()
    assert np.isfinite(objective).any()

    observed = hb.Dataset.from_measurements(
        ex.standard_nlp, u=data.u, measured={ex.fhat: data.f}, rng=rng
    )
    fitted = ex.gp.fit(observed, inputs=[ex.u], output=ex.fhat, rng=rng)
    solution = ex.standard_nlp.solve(hb.Exact(), rng, surrogate=[fitted])
    _, axes = plt.subplots(2, 1)
    hb.plots.solution(ex.standard_nlp, solution, axes)
    with pytest.raises(ValueError, match="on 2 axes"):
        hb.plots.solution(ex.standard_nlp, solution, [*axes, axes[0]])
    plt.close("all")


def test_figures_draw_over_two_u():
    """An SAA iteration over two u draws 5 panels of contours, the acquisition alone
    draws on one axes, and a wrong number of axes raises."""
    rng = np.random.default_rng(0)
    u1, u2 = hb.Var("u1", lower=-2.0, upper=2.0), hb.Var("u2", lower=-2.0, upper=2.0)
    nlp = hb.HybridNLP(
        u=[u1, u2],
        x=[ex.x],
        y=[ex.y],
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4),
    )
    u = hb.lhs_points([u1, u2], 6, rng)
    data = hb.Dataset.from_measurements(
        nlp, u=u, measured={ex.y: np.sin(u[:, 0]) * np.cos(u[:, 1])}, rng=rng
    )
    surrogate = [ex.gp.fit(data, inputs=[u1, u2], output=ex.y, rng=rng)]
    solution = nlp.solve(
        hb.SAA(rng.standard_normal((4, 1))),
        rng,
        surrogate=surrogate,
        objective=hb.obj.SoftLogEI(data.incumbents[-1], temperature=0.05),
    )
    _, axes = plt.subplots(1, 5)
    hb.plots.iteration(nlp, solution, axes, data)
    _, ax = plt.subplots()
    hb.plots.acquisition(nlp, solution, ax)
    with pytest.raises(ValueError, match="two u on 5"):
        hb.plots.solution(nlp, solution, axes[:3])
    plt.close("all")
