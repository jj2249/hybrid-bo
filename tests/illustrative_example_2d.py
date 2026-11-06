"""The illustrative example with two decision variables, run with SAA-EI, Lin-EI,
SoftSAA-EI, standard BO and Latin hypercube sampling.

Two decision variables u1 and u2, one mechanistic state x and one unknown state y.
The known equation and the objective are the illustrative example's, x + exp(x) = y
and f = (6x - 2)^2 sin(12x - 4); the unknown equation is y = sin(u1) cos(u2), which
only the oracle knows. f depends on u only through y, so its minimum is attained
along a curve in the plane of u. With one unknown state and two decision variables,
the iteration figures draw every panel: a row of y (oracle, GP mean, GP std) and a
row of f.

The tests share it, and the docs quote it. Run it with
`pixi run -e experiments python tests/illustrative_example_2d.py NAME`.
"""

import sys
from pathlib import Path

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

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


measure = hb.measure_oracle(problem, oracle, noise_std=0.0)

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

# Every method's budget: 6 initial points, 2(N + 1), then 10 iterations
N_INITIAL = 6
N_ITERATIONS = 10
N_RUNS = 10

# Methods: each is one run from the generator `rng`. Hybrid BO substitutes the GP
# prediction for the unknown state, with SAA-EI, with Lin-EI and with SoftSAA-EI at a
# temperature of 50e-3, about 1e-3 times the optimum's magnitude (about 50)


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


# Every method's runs from the same seeds, so run i of each starts from the same initial
# points, then the results and their figure
if __name__ == "__main__":
    # The study's name, its folder made first so a name in use fails before the runs
    folder = Path("results/illustrative_example_2d", sys.argv[1])
    folder.mkdir(parents=True)

    methods = (
        hybrid_bo,
        hybrid_bo_lin,
        hybrid_bo_soft,
        standard_bo,
        latin_hypercube_sampling,
    )
    runs = {
        method.__name__: [method(np.random.default_rng(seed)) for seed in range(N_RUNS)]
        for method in methods
    }
    hb.save_runs(runs, folder)

    # Score each run at the oracle, with a generator of its own
    rng = np.random.default_rng(0)
    curves = {
        label: [
            hb.oracle_incumbents(
                run.dataset, hb.oracle_values(problem, oracle, run.dataset.u, rng).f
            )
            for run in group
        ]
        for label, group in runs.items()
    }
    figure, ax = plt.subplots()
    hb.plots.regret(curves, optimum, ax)
    figure.savefig(folder / "regret.png")
