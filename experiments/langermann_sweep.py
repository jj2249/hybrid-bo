"""A sweep over initial fractions on `langermann_partial`: how the best split of a
budget between initial points and BO iterations changes with the budget, for
SoftSAA-EI.

The problem, measurement, optimum and GP are those of `langermann_partial`: the
Langermann function of two decision variables u1 and u2,
f = sum_j c_j exp(-d_j / pi) cos(pi d_j), where d_j = (u1 - a1j)^2 + (u2 - a2j)^2 is
the squared distance from center j. The distances from centers 1 and 4 are the
mechanistic states x1 and x4, with the known equations x_j = d_j, and the other three
are the unknown states y2, y3 and y5, which only the oracle knows. f, the box and the
optimum are those of Appendix D.1 of Astudillo & Frazier, "Bayesian optimization of
composite functions" (arXiv 1906.01537). Langermann's many local minima make it a
harder global optimization problem than Rosenbrock.

There is one configuration per number of initial points, n_initial = 2, 4, ..., 38,
each run to 40 evaluations. A BO run with a larger budget passes through the same
first evaluations as one with a smaller budget (the prefix property), so each run
gives the regret at every budget B from n_initial + 1 to 40, and the initial fraction
n_initial / B is never 0 or 1. The full study is 19 configurations x 10 runs, about
5 hours on one core. Run it in parallel, then plot the newest study:

    pixi run experiment langermann_sweep --n-workers 8
    pixi run plot langermann_sweep

`--method n_initial_16 --method n_initial_36` runs only those configurations.

`plot` draws the budget heatmap, the regret against the initial fraction at a budget
of 20, and the regret over evaluations by number of initial points, besides the
iteration figures of run 0 of each configuration (without GP panels, since those
draw one unknown state). `pixi run experiment langermann_sweep --help` lists the
flags.
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

# The oracle optimum, near u = (2.795, 1.595), as in langermann_partial: found once
# with hb.oracle_optimum from 1000 starts and checked on a 2001 x 2001 grid of the box
# (200 starts stop at a local minimum, -4.1009)
optimum = -4.155809291847786

# The prior GP, which each run fits to its own data: a zero mean and an output scale
# times a Matern-5/2 kernel with a lengthscale per input, both fitted within these
# bounds (in scaled units), and no noise, as the measurement has none
outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
    noise_std=0.0,
)

# The method: hybrid BO with SoftSAA-EI at a temperature of 5e-3, about 1e-3 times the
# optimum's magnitude (about 4.2), with the unknown states not as variables
method = functools.partial(
    hb.hybrid_bo,
    settings=hb.HybridBOSettings(
        gp=gp,
        acquisition=hb.EI(),
        approximation=hb.SoftSAA(
            temperature=5e-3, n_scenarios=25, unknown_states_as_variables=False
        ),
    ),
)

# Configurations: the method with 2, 4, ..., 38 initial points, each to 40
# evaluations, labeled with n_initial padded to two digits so they sort
configurations = {
    f"n_initial_{n_initial:02d}": hb.Configuration(
        method=method, n_evaluations=40, n_initial=n_initial
    )
    for n_initial in range(2, 40, 2)
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, seed=None)

# Run or plot from the command line, drawing the sweep's figures instead of the
# convergence and regret of every configuration. The oracle measurement is cheap, so
# the iteration figures draw its oracle values over the grid of u
if __name__ == "__main__":
    cli.main(
        "langermann_sweep",
        problem,
        measure,
        configurations,
        study_settings,
        optimum=optimum,
        true_curves=True,
        figures={
            "budget": hb.plots.budget_heatmap,
            "at_budget_20": functools.partial(hb.plots.regret_at_budget, budget=20),
            "regret_by_initial_fraction": hb.plots.regret_by_initial_fraction,
        },
    )
