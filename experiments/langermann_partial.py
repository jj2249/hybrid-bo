"""The Langermann function with two of its five squared distances known: a benchmark
with known and unknown equations, run with SAA-EI, SoftSAA-EI, standard BO and Latin
hypercube sampling.

Two decision variables u1 and u2. The Langermann function is
f = sum_j c_j exp(-d_j / pi) cos(pi d_j), a sum of ripples around five centers with
many local minima, where d_j = (u1 - a1j)^2 + (u2 - a2j)^2 is the squared distance
from center j. The distances from centers 1 and 4 are known: they are the mechanistic
states x1 and x4, with the known equations x_j = d_j. The other three are the unknown
states y2, y3 and y5, which only the oracle knows. Hybrid BO models each unknown state
with its own GP and knows the known equations and f; standard BO models f directly,
so it sees the plain Langermann function and behaves exactly as on `langermann`.

Centers 1 and 4 have small weights and lie far from the optimum, so the terms that
shape it are still learned. The known equations depend on u only, so no GP's
uncertainty passes through them. f, the box and the optimum are those of Appendix D.1
of Astudillo & Frazier, "Bayesian optimization of composite functions" (arXiv
1906.01537). Run it, then plot the newest study:

    pixi run experiment langermann_partial [--n-runs 3 --plot ...]
    pixi run plot langermann_partial
    pixi run experiment langermann_partial plot --plot-runs all

`pixi run experiment langermann_partial --help` lists the flags. Hybrid BO's iteration
figures have no GP panels, since those draw one unknown state.
"""

import functools

import casadi as cas
import cli

import hybrid_bo as hb

# The centers (a1j, a2j) and the weights c_j of the Langermann function
A = [[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]
C = [1, 2, 5, 2, 3]


def squared_distance(v, j: int):
    """The squared distance d_j of the point (v.u1, v.u2) from center j (from 1)."""
    return (v.u1 - A[0][j - 1]) ** 2 + (v.u2 - A[1][j - 1]) ** 2


# Problem: what we know about the physics, and what we want. The states' start bounds
# cover their range over the box, rounded out: each center lies in the box, so every
# d_j has minimum 0, and its maximum is at the farthest corner
u1 = hb.Var("u1", lower=0.0, upper=10.0)
u2 = hb.Var("u2", lower=0.0, upper=10.0)
x1 = hb.Var("x1", start=(0.0, 74.0))
x4 = hb.Var("x4", start=(0.0, 117.0))
y2 = hb.Var("y2", start=(0.0, 89.0))
y3 = hb.Var("y3", start=(0.0, 145.0))
y5 = hb.Var("y5", start=(0.0, 130.0))

system = hb.System(
    u=[u1, u2],
    x=[x1, x4],
    y=[y2, y3, y5],
    g=lambda v: [v.x1 - squared_distance(v, 1), v.x4 - squared_distance(v, 4)],
    h_inputs=[u1, u2],  # y = h(u1, u2)
    measured=[y2, y3, y5],
)
problem = hb.Problem(
    system,
    f=lambda v: sum(
        c * cas.exp(-d / cas.pi) * cas.cos(cas.pi * d)
        for c, d in zip(C, [v.x1, v.y2, v.y3, v.x4, v.y5], strict=True)
    ),
)


# Measurement: an oracle plays the real system
def oracle(v):
    """The unknown equations y_j = (u1 - a1j)^2 + (u2 - a2j)^2 for centers 2, 3 and 5,
    hidden from the methods."""
    return [squared_distance(v, j) for j in (2, 3, 5)]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum, near u = (2.795, 1.595): the study measures regret from it, the
# methods never see it. f is the Langermann function of u, whose optimum was found once
# with hb.oracle_optimum from 1000 starts and checked on a 2001 x 2001 grid of the box
# (200 starts stop at a local minimum, -4.1009). On this problem,
# hb.oracle_optimum(problem, oracle, np.random.default_rng(0), n_starts=1000) finds
# the same value
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
# temperature of 5e-3, about 1e-3 times the optimum's magnitude (about 4.2)
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
                temperature=5e-3, n_scenarios=25, unknown_states_as_variables=False
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
        "langermann_partial",
        problem,
        measure,
        configurations,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
