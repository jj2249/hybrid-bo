"""The 2D Rosenbrock function as a hybrid model, run with SAA-EI, SoftSAA-EI and
standard BO.

Two decision variables u1 and u2, one mechanistic state x and one unknown state y. The
objective is f = 100 y^2 + x^2 = 100 (u2 - u1^2)^2 + (u1 - 1)^2, whose minimum f = 0
at u = (1, 1) lies at the end of a flat curved valley. The question is whether
SoftSAA-EI reaches a lower regret than SAA-EI there: once the incumbent is in the
valley, few scenarios improve anywhere, so SAA-EI is exactly zero over most of the
domain and the acquisition multistart gets no gradient, while SoftSAA-EI has no flat
regions.

It adapts Appendix D.2 of Astudillo & Frazier, "Bayesian optimization of composite
functions" (arXiv 1906.01537), to 2D. D.2 has two unknown outputs, y1 = u2 - u1^2 and
y2 = u1, and no known equations. The library supports only one unknown state, so the
identity output y2 = u1 moves to the known side as the known equation x = u1 - 1, and
the unknown equation is y = u2 - u1^2, which only the oracle knows. Run it, then plot
the newest study:

    pixi run experiment rosenbrock [--n-runs 3 --plot ...]
    pixi run plot rosenbrock

`pixi run experiment rosenbrock --help` lists the flags.
"""

import functools

import cli

import hybrid_bo as hb

# Problem: what we know about the physics, and what we want. The decision variables
# span the restricted box of the SFU library's Rosenbrock function, and the states'
# start bounds cover their range over it
u1 = hb.Var("u1", lower=-2.048, upper=2.048)
u2 = hb.Var("u2", lower=-2.048, upper=2.048)
x = hb.Var("x", start=(-3.1, 1.1))
y = hb.Var("y", start=(-6.3, 2.1))

system = hb.System(
    u=[u1, u2],
    x=[x],
    y=[y],
    g=lambda v: [v.x - (v.u1 - 1)],  # 0 = g(x, y, u)
    h_inputs=[u1, u2],  # y = h(u1, u2)
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: 100 * v.y**2 + v.x**2)


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equation y = h(u1, u2), hidden from the methods."""
    return [v.u2 - v.u1**2]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum, f = 0 at u = (1, 1): the study measures regret from it, the
# methods never see it. It is analytic, since f is a sum of squares that are both zero
# there, so it needs no hb.oracle_optimum search
optimum = 0.0

# Methods: each takes (problem, measure, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baseline. Hybrid BO substitutes
# the GP prediction for the unknown state, with SAA-EI and with SoftSAA-EI at a
# temperature of 10, deliberately high so late-run improvements still have a gradient
# (#113 scales it with f)
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            approximation=hb.SAA(unknown_states_as_variables=False)
        ),
    ),
    "hybrid_bo_soft": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            approximation=hb.SoftSAA(10.0, unknown_states_as_variables=False)
        ),
    ),
    "standard_bo": hb.standard_bo,
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, n_initial=5, n_iterations=20, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the
# iteration figures would draw its true curves (they need one decision variable, so
# plot skips them here for now)
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
