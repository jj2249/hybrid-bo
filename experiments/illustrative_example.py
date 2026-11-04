"""The illustrative example of the paper as an experiment, run with every method.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows.

Run it with `pixi run benchmark -d "experiment[module=illustrative_example]"`.
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

# The prior GP of every BO method, which each fits to its own data: a zero mean and an
# output scale times a Matern-5/2 kernel, both fitted within these bounds (in scaled
# units), and no noise, as the measurement has none
outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
    noise_std=0.0,
)

# Methods: each takes (problem, measure, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baselines. Hybrid BO substitutes
# the GP prediction for the unknown states, as this experiment always has, with SAA-EI,
# with Lin-EI and with SoftSAA-EI at a temperature of 50e-3, about 1e-3 times the
# optimum's magnitude (about 50)
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp,
            acquisition=hb.EI(),
            approximation=hb.SAA(n_scenarios=25, unknown_states_as_variables=False),
        ),
    ),
    "hybrid_bo_lin": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp,
            acquisition=hb.EI(),
            approximation=hb.Linearization(unknown_states_as_variables=False),
        ),
    ),
    "hybrid_bo_soft": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp,
            acquisition=hb.EI(),
            approximation=hb.SoftSAA(
                temperature=50e-3, n_scenarios=25, unknown_states_as_variables=False
            ),
        ),
    ),
    "standard_bo": functools.partial(
        hb.standard_bo, settings=hb.StandardBOSettings(gp=gp, acquisition=hb.EI())
    ),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    "uniform_sampling": hb.uniform_sampling,
}

# Every method's budget: 7 evaluations, the first 2 of them initial points (so 5 iterations)
n_evaluations = 7
n_initial = 2
