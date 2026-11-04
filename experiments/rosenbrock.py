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
functions" (arXiv 1906.01537), to 2D.

Run it with `pixi run -e experiments python experiments/rosenbrock.py`.

Hybrid BO's iteration figures have no GP panels, since those draw one unknown state.
"""

import matplotlib.pyplot as plt
import numpy as np

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

# Every method's budget: 5 initial points, then 20 iterations
N_INITIAL = 5
N_ITERATIONS = 20
N_RUNS = 10

# Methods: each is one run from the generator `rng`. Hybrid BO substitutes the GPs'
# predictions for the unknown states, with SAA-EI and with SoftSAA-EI at a temperature
# of 100e-3, chosen from trial runs: the other experiments take about 1e-3 times the
# optimum's magnitude, but this optimum is 0. Its -log keeps a gradient at any
# temperature, and a temperature near the improvements makes it re-measure the
# incumbent, which scores tau log 2


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
            temperature=100e-3, n_scenarios=25, unknown_states_as_variables=False
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
    methods = (hybrid_bo, hybrid_bo_soft, standard_bo, latin_hypercube_sampling)
    runs = {
        method.__name__: [method(np.random.default_rng(seed)) for seed in range(N_RUNS)]
        for method in methods
    }
    folder = hb.save_runs(runs, "results/rosenbrock")
    figure, ax = plt.subplots()
    hb.plots.regret(runs, optimum, ax)
    figure.savefig(folder / "regret.png")
