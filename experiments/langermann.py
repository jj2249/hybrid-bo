"""The Langermann function as a composite function, run with SAA-EI, SoftSAA-EI,
standard BO and Latin hypercube sampling.

Two decision variables u1 and u2, no mechanistic states or known equations, and five
unknown states, the squared distances y_j = (u1 - a1j)^2 + (u2 - a2j)^2 from five
centers, which only the oracle knows. The objective is
f = sum_j c_j exp(-y_j / pi) cos(pi y_j), a sum of ripples around the centers with many
local minima. Hybrid BO models each y_j with its own GP and knows how f combines them;
standard BO models f directly.

It is Appendix D.1 of Astudillo & Frazier, "Bayesian optimization of composite
functions" (arXiv 1906.01537).

Run it with `pixi run -e experiments python experiments/langermann.py NAME`.

Hybrid BO's iteration figures have no GP panels, since those draw one unknown state.
"""

import sys
from pathlib import Path

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

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

# Every method's budget: 6 initial points, 2(N + 1), then 20 iterations
N_INITIAL = 6
N_ITERATIONS = 20
N_RUNS = 10

# Methods: each is one run from the generator `rng`. Hybrid BO substitutes the GPs'
# predictions for the unknown states, with SAA-EI and with SoftSAA-EI at a temperature
# of 5e-3, about 1e-3 times the optimum's magnitude (about 4.2)


def hybrid_bo(rng: np.random.Generator) -> hb.Run:
    """Hybrid BO with SAA-EI."""
    return hb.hybrid_bo(
        problem,
        measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=rng,
        gp=gp,
        acquisition=hb.EI(),
        approximation=hb.SAA(n_scenarios=25, unknown_states_as_variables=False),
    )


def hybrid_bo_soft(rng: np.random.Generator) -> hb.Run:
    """Hybrid BO with SoftSAA-EI."""
    return hb.hybrid_bo(
        problem,
        measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=rng,
        gp=gp,
        acquisition=hb.EI(),
        approximation=hb.SoftSAA(
            temperature=5e-3, n_scenarios=25, unknown_states_as_variables=False
        ),
    )


def standard_bo(rng: np.random.Generator) -> hb.Run:
    """Standard BO with EI."""
    return hb.standard_bo(
        problem,
        measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=rng,
        gp=gp,
        acquisition=hb.EI(),
    )


def latin_hypercube_sampling(rng: np.random.Generator) -> hb.Run:
    """Latin hypercube sampling."""
    return hb.latin_hypercube_sampling(
        problem, measure, n_initial=N_INITIAL, n_iterations=N_ITERATIONS, rng=rng
    )


# Every method's runs from the same seeds, so run i of each starts from the same initial
# points, then the results and their figure
if __name__ == "__main__":
    # The study's name, its folder made first so a name in use fails before the runs
    folder = Path("results/langermann", sys.argv[1])
    folder.mkdir(parents=True)

    methods = (hybrid_bo, hybrid_bo_soft, standard_bo, latin_hypercube_sampling)
    runs = {
        method.__name__: [method(np.random.default_rng(seed)) for seed in range(N_RUNS)]
        for method in methods
    }
    hb.save_runs(runs, folder)
    figure, ax = plt.subplots()
    hb.plots.regret(runs, optimum, ax)
    figure.savefig(folder / "regret.png")
