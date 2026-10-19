"""The illustrative example of the paper as an experiment, run with every method.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows. Run it, then plot the newest study:

    pixi run experiment illustrative_example [--n-runs 3 --plot ...]
    pixi run plot illustrative_example

`pixi run experiment illustrative_example --help` lists the flags.
"""

import functools

import casadi as cas
import cli

import hybrid_bo as hb

# Problem: what we know about the physics, and what we want
u = hb.Var("u", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))
y = hb.Var("y", start=(-2.0, 2.0))

system = hb.System(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],  # 0 = g(x, y, u)
    h_inputs=[u],  # y = h(u)
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equation y = h(u), hidden from the methods."""
    return [cas.sin(v.u)]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum, at u = -0.45: the study measures regret from it, the methods
# never see it. Found once with
# hb.oracle_optimum(problem, oracle, np.random.default_rng(0), n_starts=200), and
# checked: hybrid BO runs reach it, and none evaluates a point below it
optimum = -50.4608044704652

# Methods: each takes (problem, measure, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baselines. Hybrid BO substitutes
# the GP prediction for the unknown states, as this experiment always has
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            approximation=hb.SAA(unknown_states_as_variables=False)
        ),
    ),
    "standard_bo": hb.standard_bo,
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    "uniform_sampling": hb.uniform_sampling,
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, n_initial=2, n_iterations=5, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the iteration
# figures draw its true curves
if __name__ == "__main__":
    cli.main(
        "illustrative_example",
        problem,
        measure,
        methods,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
