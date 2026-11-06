"""Hybrid BO with SoftSAA-EI against standard BO on the illustrative example.

The known equation is x + exp(x) = y, and the unknown one is y = sin(u), which the
methods never see. Writes regret.png and iteration.png to the current folder:

    pixi run -e dev python examples/illustrative_example.py
"""

import logging

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

import hybrid_bo as hb

logging.basicConfig(level=logging.INFO)

# Show what is being fitted, and whether it succeeds
logging.basicConfig()
logging.getLogger("hybrid_bo").setLevel(logging.INFO)

N_INITIAL, N_ITERATIONS, N_RUNS = 2, 5, 10
OPTIMUM = -50.4608044704652  # the lowest f, at u = -0.446

# The problem: what we know about the system, and what we want
u = hb.Var("u", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))
y = hb.Var("y", start=(-2.0, 2.0))
system = hb.System(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],
    h_inputs=[u],
    measured=[y],
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))


def measure(u):
    """The system: here an oracle, but it could run a plant or a simulator."""
    return np.sin(u)


# The prior GP, and the approximation of EI
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=hb.gp.Param((0.1, 10.0)) * hb.gp.Matern52(),
    noise_std=0.0,
)
approximation = hb.SoftSAA(temperature=50e-3, unknown_states_as_variables=False)

figure, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8), layout="constrained")

# Hybrid BO: a GP of the unknown state, inside the known equations
hybrid_curves = []
for seed in range(N_RUNS):
    rng = np.random.default_rng(seed)
    points = hb.lhs_points(system.u, N_INITIAL, rng)
    dataset = hb.reconstruct_states(
        problem, u=points, measured=measure(points), rng=rng
    )
    scenarios = hb.saa_scenarios(25, system.n_y, rng)  # once, before the loop
    for _ in range(N_ITERATIONS):
        before = dataset
        incumbent = hb.incumbents(dataset)[-1]
        fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
        solution = hb.hybrid_trial_point(
            problem, fitted, scenarios, incumbent, rng, approximation=approximation
        )
        u_new = solution.u[None]
        new = hb.reconstruct_states(problem, u=u_new, measured=measure(u_new), rng=rng)
        dataset = hb.extend(dataset, new)
    hybrid_curves.append(hb.incumbents(dataset))

# The last iteration of the last run
built = hb.hybrid_acquisition(problem, fitted, scenarios, incumbent, approximation)
hb.plots.iteration(built, solution, fitted[0], before, axes[:, 0], scenarios)
axes[0, 0].set(title="Hybrid BO, SoftSAA-EI", ylabel="y")

# Standard BO: a GP of the objective
standard_curves = []
for seed in range(N_RUNS):
    rng = np.random.default_rng(seed)
    points = hb.lhs_points(system.u, N_INITIAL, rng)
    dataset = hb.reconstruct_states(
        problem, u=points, measured=measure(points), rng=rng
    )
    for _ in range(N_ITERATIONS):
        before = dataset
        incumbent = hb.incumbents(dataset)[-1]
        fitted = hb.fit_gp(gp, dataset.u, dataset.f, rng)
        solution = hb.standard_trial_point(problem, fitted, incumbent, rng)
        u_new = solution.u[None]
        new = hb.reconstruct_states(problem, u=u_new, measured=measure(u_new), rng=rng)
        dataset = hb.extend(dataset, new)
    standard_curves.append(hb.incumbents(dataset))

axes[0, 1].remove()
axes[1, 1].set_prop_cycle(color=["tab:orange"])
built = hb.standard_acquisition(problem, fitted, incumbent)
hb.plots.iteration(built, solution, fitted, before, axes[1:, 1])
axes[1, 1].set(title="Standard BO", ylabel="f")

# The true curves: the states and f at a grid of u, from the system itself
grid = np.linspace(-2.0, 2.0, 201)[:, None]
truth = hb.reconstruct_states(problem, u=grid, measured=measure(grid), rng=rng)
axes[0, 0].plot(grid, truth.y, "k--", label="oracle")
axes[1, 0].plot(grid, truth.f, "k--", label="oracle")
axes[1, 1].plot(grid, truth.f, "k--", label="oracle")
for ax in figure.axes:
    ax.legend(fontsize="small")
figure.savefig("iteration.png", dpi=150)

# Regret over the runs
figure, ax = plt.subplots(layout="constrained")
curves = {"hybrid BO": hybrid_curves, "standard BO": standard_curves}
hb.plots.regret(curves, OPTIMUM, ax)
figure.savefig("regret.png", dpi=150)
