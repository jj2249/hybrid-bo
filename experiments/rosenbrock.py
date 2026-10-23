"""The 2D Rosenbrock function as a composite function, run with SAA-EI, SoftSAA-EI,
standard BO and Latin hypercube sampling.

Two decision variables u1 and u2, no mechanistic states or known equations, and two
unknown states, y1 = u2 - u1^2 and y2 = u1, which only the oracle knows. The objective
is f = 100 y1^2 + (y2 - 1)^2 = 100 (u2 - u1^2)^2 + (u1 - 1)^2, whose minimum f = 0
at u = (1, 1) lies at the end of a flat curved valley. The question is whether
SoftSAA-EI reaches a lower regret than SAA-EI there: once the incumbent is in the
valley, few scenarios improve anywhere, so SAA-EI is exactly zero over most of the
domain and the acquisition multistart gets no gradient, while SoftSAA-EI has no flat
regions.

It adapts Appendix D.2 of Astudillo & Frazier, "Bayesian optimization of composite
functions" (arXiv 1906.01537), to 2D. Run it, then plot the newest study:

    pixi run experiment rosenbrock [--n-runs 3 --plot ...]
    pixi run plot rosenbrock
    pixi run experiment rosenbrock plot --plot-runs all

`pixi run experiment rosenbrock --help` lists the flags. Hybrid BO's iteration figures
have no GP panels, since those draw one unknown state.
"""

import functools

import cli

import hybrid_bo as hb

# Problem: what we know about the physics, and what we want. The decision variables
# span the restricted box of the Rosenbrock function in Surjanovic & Bingham's Virtual
# Library of Simulation Experiments (SFU). The unknown states' start bounds cover their
# range over it, rounded out: y1 = u2 - u1^2 is in [-6.242, 2.048] and y2 = u1 in
# [-2.048, 2.048]
u1 = hb.Var("u1", lower=-2.048, upper=2.048)
u2 = hb.Var("u2", lower=-2.048, upper=2.048)
y1 = hb.Var("y1", start=(-6.3, 2.1))
y2 = hb.Var("y2", start=(-2.1, 2.1))

system = hb.System(
    u=[u1, u2],
    y=[y1, y2],
    h_inputs=[u1, u2],  # y = h(u1, u2)
    measured=[y1, y2],
)
problem = hb.Problem(system, f=lambda v: 100 * v.y1**2 + (v.y2 - 1) ** 2)


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equations y1 = u2 - u1^2 and y2 = u1, hidden from the methods."""
    return [v.u2 - v.u1**2, v.u1]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum, f = 0 at u = (1, 1): the study measures regret from it, the
# methods never see it. It is analytic, since f is a sum of squares that are both zero
# there, so it needs no hb.oracle_optimum search
optimum = 0.0

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
# the GPs' predictions for the unknown states, with SAA-EI and with SoftSAA-EI at a
# temperature of 1e-6. Its -log keeps a gradient at any temperature, and a temperature
# near the improvements makes it re-measure the incumbent, which scores tau log 2
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp,
            acquisition=hb.EI(),
            approximation=hb.SAA(n_scenarios=25, unknown_states_as_variables=False),
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

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, n_initial=5, n_iterations=20, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the
# iteration figures draw its oracle values over the grid of u
if __name__ == "__main__":
    cli.main(
        "rosenbrock",
        problem,
        measure,
        methods,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
