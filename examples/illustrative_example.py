"""The illustrative example of the paper, run with every method.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows. Run it, then plot the newest study:

    pixi run example illustrative_example
    pixi run plot illustrative_example
"""

import functools

import casadi as cas

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


# Test case: an oracle plays the real system
def oracle(v):
    """The unknown equation y = h(u), known only to the test case."""
    return [cas.sin(v.u)]


experiment = hb.oracle_experiment(system, oracle, noise_std=0.0)

# The oracle optimum, which regret is measured from, at u = -0.45. Found once with
# hb.oracle_optimum(problem, oracle, np.random.default_rng(0), n_starts=200), and
# checked: hybrid BO runs reach it, and none evaluates a point below it
optimum = -50.4608044704652

# Methods: each takes (problem, experiment, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baselines. Hybrid BO substitutes
# the GP prediction for the unknown states, as this example always has
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(unknown_states_as_variables=False),
    ),
    "standard_bo": functools.partial(hb.standard_bo, settings=hb.StandardBOSettings()),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    "uniform_sampling": hb.uniform_sampling,
}

# The study, the CLI's defaults. Seed None draws a fresh seed, which the saved study
# keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, n_initial=2, n_iterations=5, seed=None)
