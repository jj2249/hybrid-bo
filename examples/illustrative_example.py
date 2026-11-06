"""Hybrid BO with SAA-SoftEI against standard BO on the illustrative example.

The known equation is x + exp(x) = y, and the unknown one is y = sin(u), which the
methods never see. Writes regret.png and iteration.png to the current folder:

    pixi run -e dev python examples/illustrative_example.py
"""

import logging

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

import hybrid_bo as hb

# Show what is being fitted, and whether it succeeds
logging.basicConfig()
logging.getLogger("hybrid_bo").setLevel(logging.INFO)

plt.style.use("hybrid_bo.default")

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


def measure(u, rng):
    """The system: here an oracle, but it could run a plant or a simulator."""
    return np.sin(u)


# The prior GP, and the acquisition function
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=hb.gp.Scale(hb.gp.Matern52(), hb.gp.Param((0.1, 10.0))),
    noise_std=0.0,
)
acquisition = hb.SoftEI(temperature=50e-3)

figure, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8), layout="constrained")

# Hybrid BO: a GP of the unknown state, inside the known equations
hybrid_curves = []
for seed in range(N_RUNS):
    rng = np.random.default_rng(seed)
    points = hb.lhs_points(system.u, N_INITIAL, rng)
    dataset = hb.reconstruct_states(
        problem, u=points, measured=measure(points, rng), rng=rng
    )
    z = rng.standard_normal((25, system.n_y))  # the scenarios, once, before the loop
    w = np.full(25, 1 / 25)
    for _ in range(N_ITERATIONS):
        before = dataset
        incumbent = hb.incumbents(dataset)[-1]
        fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
        equivalent = hb.hybrid_deterministic_equivalent(
            problem,
            fitted,
            z,
            w,
            incumbent,
            acquisition=acquisition,
            unknown_states_as_variables=False,
        )
        solution = equivalent.solve(rng)
        u_new = solution.u[None]
        new = hb.reconstruct_states(
            problem, u=u_new, measured=measure(u_new, rng), rng=rng
        )
        dataset = hb.extend(dataset, new)
    hybrid_curves.append(hb.incumbents(dataset))

# The last iteration of the last run
hb.plots.iteration(equivalent, solution, fitted[0], before, axes[:, 0])
axes[0, 0].set(title="Hybrid BO, SAA-SoftEI", ylabel="y")

# Standard BO: a GP of the objective
standard_curves = []
for seed in range(N_RUNS):
    rng = np.random.default_rng(seed)
    points = hb.lhs_points(system.u, N_INITIAL, rng)
    dataset = hb.reconstruct_states(
        problem, u=points, measured=measure(points, rng), rng=rng
    )
    for _ in range(N_ITERATIONS):
        before = dataset
        incumbent = hb.incumbents(dataset)[-1]
        fitted = gp.fit(dataset.u, dataset.f, rng)
        standard = hb.standard_acquisition(problem, fitted, incumbent)
        solution = standard.solve(rng)
        u_new = solution.w[None]
        new = hb.reconstruct_states(
            problem, u=u_new, measured=measure(u_new, rng), rng=rng
        )
        dataset = hb.extend(dataset, new)
    standard_curves.append(hb.incumbents(dataset))

axes[0, 1].remove()
axes[1, 1].set_prop_cycle(color=["tab:orange"])
hb.plots.iteration(standard, solution, fitted, before, axes[1:, 1])
axes[1, 1].set(title="Standard BO", ylabel="f")

# The true curves: the states and f at a grid of u, from the system itself
grid = np.linspace(-2.0, 2.0, 201)[:, None]
truth = hb.reconstruct_states(problem, u=grid, measured=measure(grid, rng), rng=rng)
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
