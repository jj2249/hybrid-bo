"""The illustrative example of the paper, run with every method.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows. Run and plot it with the CLI in `examples/main.py`:

    pixi run -e dev python examples/main.py run illustrative_example
    pixi run -e dev python examples/main.py plot examples/results/illustrative_example/<UTC time>
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
experiment = hb.oracle_experiment(
    system, oracle=lambda v: [cas.sin(v.u)], noise_std=0.0
)

# Methods: each takes (problem, experiment, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baselines. The plots take the BO
# methods' settings too. Hybrid BO substitutes the GP prediction for the unknown
# states, as this example always has
settings = {
    "hybrid_bo": hb.HybridBOSettings(unknown_states_as_variables=False),
    "standard_bo": hb.StandardBOSettings(),
}
methods = {
    "hybrid_bo": functools.partial(hb.hybrid_bo, settings=settings["hybrid_bo"]),
    "standard_bo": functools.partial(hb.standard_bo, settings=settings["standard_bo"]),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    "uniform_sampling": hb.uniform_sampling,
}

# Runs, the CLI's defaults
seed = None  # None draws a fresh seed; every run records the seed it used
n_runs = 1
n_initial = 2
n_iterations = 5
