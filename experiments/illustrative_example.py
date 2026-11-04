"""The illustrative example of the paper as an experiment, run with every method.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows.

Run it with `pixi run -e experiments python experiments/illustrative_example.py`.
"""

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

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

# Every method's budget: 2 initial points, then 5 iterations
N_INITIAL = 2
N_ITERATIONS = 5
N_RUNS = 10

# Methods: each is one run from the generator `rng`. Hybrid BO substitutes the GP
# prediction for the unknown states, as this experiment always has, with SAA-EI, with
# Lin-EI and with SoftSAA-EI at a temperature of 50e-3, about 1e-3 times the optimum's
# magnitude (about 50)


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


def hybrid_bo_lin(rng: np.random.Generator) -> hb.Run:
    """Hybrid BO with Lin-EI."""
    return hb.hybrid_bo(
        problem,
        measure,
        n_initial=N_INITIAL,
        n_iterations=N_ITERATIONS,
        rng=rng,
        gp=gp,
        acquisition=hb.EI(),
        approximation=hb.Linearization(unknown_states_as_variables=False),
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
            temperature=50e-3, n_scenarios=25, unknown_states_as_variables=False
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


def uniform_sampling(rng: np.random.Generator) -> hb.Run:
    """Uniform sampling."""
    return hb.uniform_sampling(
        problem, measure, n_initial=N_INITIAL, n_iterations=N_ITERATIONS, rng=rng
    )


# Every method's runs from the same seeds, so run i of each starts from the same initial
# points, then the results and their figure
if __name__ == "__main__":
    methods = (
        hybrid_bo,
        hybrid_bo_lin,
        hybrid_bo_soft,
        standard_bo,
        latin_hypercube_sampling,
        uniform_sampling,
    )
    runs = {
        method.__name__: [method(np.random.default_rng(seed)) for seed in range(N_RUNS)]
        for method in methods
    }
    folder = hb.save_runs(runs, "results/illustrative_example")
    figure, ax = plt.subplots()
    hb.plots.regret(runs, optimum, ax)
    figure.savefig(folder / "regret.png")
