"""The illustrative example with two decision variables, run with SAA-EI, Lin-EI,
SoftSAA-EI, standard BO and Latin hypercube sampling.

Two decision variables u1 and u2, one mechanistic state x and one unknown state y.
The known equation and the objective are the illustrative example's, x + exp(x) = y
and f = (6x - 2)^2 sin(12x - 4); the unknown equation is y = sin(u1) cos(u2), which
only the oracle knows. f depends on u only through y, so its minimum is attained
along a curve in the plane of u. With one unknown state and two decision variables,
the iteration figures draw every panel: a row of y (oracle, GP mean, GP std) and a
row of f. Run it, then plot the newest study:

    pixi run experiment illustrative_example_2d [--n-runs 3 --plot ...]
    pixi run plot illustrative_example_2d
    pixi run experiment illustrative_example_2d plot --plot-runs all

`pixi run experiment illustrative_example_2d --help` lists the flags.
"""

import functools

import casadi as cas
import cli

import hybrid_bo as hb

# Problem: what we know about the physics, and what we want. y = sin(u1) cos(u2) is in
# [-1, 1] over the box, as in the illustrative example, so the states share its start
# bounds
u1 = hb.Var("u1", lower=-2.0, upper=2.0)
u2 = hb.Var("u2", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))
y = hb.Var("y", start=(-2.0, 2.0))

system = hb.System(
    u=[u1, u2],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],  # 0 = g(x, y, u)
    h_inputs=[u1, u2],  # y = h(u1, u2)
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equation y = h(u1, u2), hidden from the methods."""
    return [cas.sin(v.u1) * cas.cos(v.u2)]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum: the study measures regret from it, the methods never see it.
# Found once with
# hb.oracle_optimum(problem, oracle, np.random.default_rng(0), n_starts=200), and
# checked: it is the illustrative example's to rounding, since y takes the same values,
# so f is smallest along the curve sin(u1) cos(u2) = sin(-0.45)
optimum = -50.460804470465206

# The prior GP of every BO method, which each fits to its own data: a zero mean and an
# output scale times a Matern-5/2 kernel with a lengthscale per input, both fitted
# within these bounds (in scaled units), and no noise, as the measurement has none
outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
    noise_std=0.0,
)

# Methods: each takes (problem, measure, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baseline. Hybrid BO substitutes
# the GP prediction for the unknown state, with SAA-EI, with Lin-EI and with
# SoftSAA-EI at a temperature of 1e-6, as in rosenbrock.py
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
                temperature=1e-6, n_scenarios=25, unknown_states_as_variables=False
            ),
        ),
    ),
    "standard_bo": functools.partial(
        hb.standard_bo, settings=hb.StandardBOSettings(gp=gp, acquisition=hb.EI())
    ),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
}

# Configurations: every method with the same budget, 16 evaluations, the first 6 of
# them initial points, 2(N + 1) (so 10 iterations)
configurations = {
    label: hb.Configuration(method=method, n_evaluations=16, n_initial=6)
    for label, method in methods.items()
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the
# iteration figures draw its oracle values over the grid of u
if __name__ == "__main__":
    cli.main(
        "illustrative_example_2d",
        problem,
        measure,
        configurations,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
