"""Hybrid BO with SoftSAA-EI against standard BO on the illustrative example.

The known equation is x + exp(x) = y, and the unknown one is y = sin(u), which the
methods never see. Writes regret.png and iteration.png to the current folder:

    pixi run -e dev python examples/illustrative_example.py
"""

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.special import lambertw

import hybrid_bo as hb

N_INITIAL, N_ITERATIONS, N_RUNS, N_SCENARIOS = 2, 5, 10, 25
OPTIMUM = -50.4608044704652  # the lowest f, at u = -0.446
TEMPERATURE = 50e-3  # about 1e-3 times the optimum's magnitude
HYBRID_COLOR, STANDARD_COLOR, ORACLE_COLOR = "#2a78d6", "#eb6834", "#5f5e58"

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

outputscale = hb.gp.Param((0.1, 10.0))
lengthscale = hb.gp.Param((1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
    noise_std=0.0,
)
approximation = hb.SoftSAA(temperature=TEMPERATURE, unknown_states_as_variables=False)


def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The system: an oracle playing it, with no noise."""
    return np.sin(u)


def oracle_f(u: np.ndarray) -> np.ndarray:
    """The objective at u, solved by hand, for the oracle curve."""
    y = np.sin(u)
    x = y - lambertw(np.exp(y)).real
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def evaluate(u: np.ndarray, rng: np.random.Generator) -> hb.Dataset:
    """A one-row dataset: measure the system at u, and solve for the other states."""
    return hb.reconstruct_states(
        problem, u=u[None], measured=measure(u, rng)[None], rng=rng
    )


def initial_dataset(rng: np.random.Generator) -> hb.Dataset:
    """The initial points, measured."""
    points = hb.lhs_points(system.u, N_INITIAL, rng)
    return hb.reconstruct_states(problem, u=points, measured=np.sin(points), rng=rng)


def hybrid_bo(seed: int):
    """One run of hybrid BO; returns the dataset and what the last iteration used."""
    rng = np.random.default_rng(seed)
    dataset = initial_dataset(rng)
    scenarios = hb.saa_scenarios(N_SCENARIOS, system.n_y, rng)
    for _ in range(N_ITERATIONS):
        before = dataset
        incumbent = hb.incumbents(dataset)[-1]
        fitted = hb.fit_surrogate(system, dataset, gp=gp, rng=rng)
        solution = hb.hybrid_trial_point(
            problem, fitted, scenarios, incumbent, rng, approximation=approximation
        )
        dataset = hb.extend(dataset, evaluate(solution.u, rng))
    built = hb.hybrid_acquisition(problem, fitted, scenarios, incumbent, approximation)
    return dataset, before, fitted[0], built, solution, scenarios


def standard_bo(seed: int):
    """One run of standard BO; returns the dataset and what the last iteration used."""
    rng = np.random.default_rng(seed)
    dataset = initial_dataset(rng)
    for _ in range(N_ITERATIONS):
        before = dataset
        incumbent = hb.incumbents(dataset)[-1]
        fitted = hb.fit_gp(gp, dataset.u, dataset.f, rng)
        solution = hb.standard_trial_point(problem, fitted, incumbent, rng)
        dataset = hb.extend(dataset, evaluate(solution.u, rng))
    built = hb.standard_acquisition(problem, fitted, incumbent)
    return dataset, before, fitted, built, solution


def draw_regret(hybrid: list, standard: list) -> None:
    """The median regret of each method over its runs, with the interquartile range."""
    plt.rcParams["axes.prop_cycle"] = cycler(color=[HYBRID_COLOR, STANDARD_COLOR])
    figure, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    curves = {
        "hybrid BO, SoftSAA-EI": [hb.incumbents(run[0]) for run in hybrid],
        "standard BO": [hb.incumbents(run[0]) for run in standard],
    }
    hb.plots.regret(curves, OPTIMUM, ax)
    ax.grid(True, alpha=0.3)
    ax.set_title("Regret over 10 runs (median and interquartile range)")
    figure.savefig("regret.png", dpi=150)


def draw_iteration(hybrid: tuple, standard: tuple) -> None:
    """The last iteration of a run of each method, with the oracle's curves."""
    figure, axes = plt.subplots(
        3, 2, sharex=True, figsize=(10, 8), layout="constrained"
    )
    figure.suptitle(f"Iteration {N_ITERATIONS}: the state before the last trial point")

    # The library draws each method's panels; standard BO has no GP of y
    _, before, fitted, built, solution, scenarios = hybrid
    axes[0, 0].set_prop_cycle(color=[HYBRID_COLOR])
    hb.plots.iteration(built, solution, fitted, before, axes[:, 0], scenarios)
    axes[0, 0].set(title="Hybrid BO, SoftSAA-EI", ylabel="y")

    _, before, fitted, built, solution = standard
    axes[0, 1].remove()
    axes[1, 1].set_prop_cycle(color=[STANDARD_COLOR])
    hb.plots.iteration(built, solution, fitted, before, axes[1:, 1])
    axes[1, 1].set(title="Standard BO", ylabel="f")

    # The oracle's curves, which only a benchmark has
    grid = np.linspace(-2.0, 2.0, 201)
    axes[0, 0].plot(grid, np.sin(grid), "--", color=ORACLE_COLOR, label="oracle")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.plot(grid, oracle_f(grid), "--", color=ORACLE_COLOR, label="oracle")
    for ax in figure.axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    figure.savefig("iteration.png", dpi=150)


if __name__ == "__main__":
    hybrid_runs = [hybrid_bo(seed) for seed in range(N_RUNS)]
    standard_runs = [standard_bo(seed) for seed in range(N_RUNS)]
    draw_regret(hybrid_runs, standard_runs)
    draw_iteration(hybrid_runs[0], standard_runs[0])
