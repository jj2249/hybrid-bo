"""The figures draw on the illustrative example."""

import illustrative_example as ex
import matplotlib
import numpy as np
import pytest

import hybrid_bo as hb

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_figures_draw():
    """An SAA iteration with SoftEI draws 3 panels, an Exact solution with Mean 2, and
    the objective on the grid is finite at the solution."""
    rng = np.random.default_rng(0)
    data = ex.measure(np.linspace(-1.8, 1.8, 5)[:, None], rng)
    surrogate = [ex.gp.fit(data, inputs=[ex.u], output=ex.y, rng=rng)]
    solution = ex.nlp.solve(
        hb.SAA(rng.standard_normal((4, 1))),
        rng,
        surrogate=surrogate,
        objective=hb.obj.SoftEI(data.incumbents[-1], temperature=0.05),
    )
    _, axes = plt.subplots(3, 1)
    hb.plots.iteration(ex.nlp, solution, axes, data)
    objective = axes[2].get_lines()[0].get_ydata()
    assert np.isfinite(objective).any()

    observed = hb.Dataset.from_measurements(
        ex.standard_nlp, u=data.u, y=data.f[:, None], rng=rng
    )
    fitted = ex.gp.fit(observed, inputs=[ex.u], output=ex.fhat, rng=rng)
    solution = ex.standard_nlp.solve(hb.Exact(), rng, surrogate=[fitted])
    _, axes = plt.subplots(2, 1)
    hb.plots.solution(ex.standard_nlp, solution, axes)
    with pytest.raises(ValueError, match="on 2 axes"):
        hb.plots.solution(ex.standard_nlp, solution, [*axes, axes[0]])
    plt.close("all")
