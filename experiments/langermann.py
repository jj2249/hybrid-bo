"""The Langermann function as a composite function, run with SAA-EI, SoftSAA-EI,
standard BO and Latin hypercube sampling.

Two decision variables u1 and u2, no mechanistic states or known equations, and five
unknown states, the squared distances y_j = (u1 - a1j)^2 + (u2 - a2j)^2 from five
centers, which only the oracle knows. The objective is
f = sum_j c_j exp(-y_j / pi) cos(pi y_j), a sum of ripples around the centers with many
local minima. Hybrid BO models each y_j with its own GP and knows how f combines them;
standard BO models f directly.

It is Appendix D.1 of Astudillo & Frazier, "Bayesian optimization of composite
functions" (arXiv 1906.01537). Run it, then plot the newest study:

    pixi run experiment langermann [--n-runs 3 --plot ...]
    pixi run plot langermann
    pixi run experiment langermann plot --plot-runs all

`pixi run experiment langermann --help` lists the flags. Hybrid BO's iteration figures
have no GP panels, since those draw one unknown state.
"""

import functools

import casadi as cas
import cli

import hybrid_bo as hb

# The centers (a1j, a2j) and the weights c_j of the Langermann function
A = [[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]
C = [1, 2, 5, 2, 3]

# Problem: what we know about the physics, and what we want. The unknown states' start
# bounds cover their range over the box, rounded out: each center lies in the box, so
# every y_j has minimum 0, and its maximum is at the farthest corner
u1 = hb.Var("u1", lower=0.0, upper=10.0)
u2 = hb.Var("u2", lower=0.0, upper=10.0)
y = [
    hb.Var(f"y{j + 1}", start=(0.0, upper))
    for j, upper in enumerate([74.0, 89.0, 145.0, 117.0, 130.0])
]

system = hb.System(
    u=[u1, u2],
    y=y,
    h_inputs=[u1, u2],  # y = h(u1, u2)
    measured=y,
)
problem = hb.Problem(
    system,
    f=lambda v: sum(
        c * cas.exp(-y_j / cas.pi) * cas.cos(cas.pi * y_j)
        for c, y_j in zip(C, [v.y1, v.y2, v.y3, v.y4, v.y5], strict=True)
    ),
)


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equations y_j = (u1 - a1j)^2 + (u2 - a2j)^2, hidden from the
    methods."""
    return [(v.u1 - a1) ** 2 + (v.u2 - a2) ** 2 for a1, a2 in zip(*A, strict=True)]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum, near u = (2.795, 1.595): the study measures regret from it, the
# methods never see it. Found once with
# hb.oracle_optimum(problem, oracle, np.random.default_rng(0), n_starts=1000), and
# checked on a 2001 x 2001 grid of the box; 200 starts stop at a local minimum, -4.1009
optimum = -4.155809291847786

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
# temperature of 1e-6, as in the rosenbrock experiment
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

# Configurations: every method with the same budget, 26 evaluations, the first 6 of
# them initial points, the paper's 2(N + 1) (so 20 iterations)
configurations = {
    label: hb.Configuration(method=method, n_evaluations=26, n_initial=6)
    for label, method in methods.items()
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the
# iteration figures draw its oracle values over the grid of u
if __name__ == "__main__":
    cli.main(
        "langermann",
        problem,
        measure,
        configurations,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
