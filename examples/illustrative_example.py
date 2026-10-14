"""The illustrative example of the paper, run with every method.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows. Run it with `pixi run -e dev python examples/illustrative_example.py`; the runs
of each method are pickled to `examples/results/illustrative_example/<UTC time>/`.
"""

import functools
import pickle
from datetime import UTC, datetime
from pathlib import Path

import casadi as cas
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

# Test case: an oracle plays the real system
experiment = hb.oracle_experiment(
    system, oracle=lambda v: [cas.sin(v.u)], noise_std=0.0
)

# Methods: each takes (problem, experiment, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baselines. Hybrid BO substitutes
# the GP prediction for the unknown states, as this example always has
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(unknown_states_as_variables=False),
    ),
    "standard_bo": functools.partial(hb.standard_bo, settings=hb.StandardBOSettings()),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    "uniform_sampling": hb.uniform_sampling,
}

# Runs
seed = None  # None draws a fresh seed; every run records the seed it used
n_runs = 1
n_initial = 2
n_iterations = 5


if __name__ == "__main__":
    results_dir = (
        Path(__file__).parent
        / "results"
        / "illustrative_example"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    results_dir.mkdir(parents=True)

    # One seed for every method, so run i of each method starts from the same
    # initial points
    entropy = np.random.SeedSequence(seed).entropy
    for name, method in methods.items():
        runs = []
        for rng in np.random.default_rng(entropy).spawn(n_runs):
            initial_points = hb.initial_points(system.u, n_initial, rng)

            # A failed run still saves the points evaluated before the error
            try:
                runs.append(
                    method(problem, experiment, initial_points, n_iterations, rng=rng)
                )
            except hb.HybridBOError as error:
                print(f"{name}: run failed: {error!r}")
                if error.run is not None:
                    runs.append(error.run)

        with (results_dir / f"{name}.pkl").open("wb") as file:
            pickle.dump(runs, file)
        print(f"{name}: saved {len(runs)} of {n_runs} runs to {results_dir}")
