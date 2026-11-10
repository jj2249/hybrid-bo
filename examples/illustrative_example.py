"""Hybrid BO with SAA-SoftEI against standard BO on the illustrative example.

The known equation is x + exp(x) = y, and the unknown one is y = sin(u), which the
methods never see. Prints each method's median regret over the runs:

    pixi run -e dev python examples/illustrative_example.py
"""

import logging

import casadi as cas
import numpy as np

import hybrid_bo as hb

# Show what is being fitted, and whether it succeeds
logging.basicConfig()
logging.getLogger("hybrid_bo").setLevel(logging.INFO)

N_INITIAL, N_ITERATIONS, N_RUNS = 2, 5, 10
OPTIMUM = -50.4608044704652  # the lowest f, at u = -0.446

# What we know about the system, and what we want
u = hb.Var("u", lower=-2.0, upper=2.0)
x = hb.Var("x", start=(-2.5, 1.0))
y = hb.Var("y", start=(-2.0, 2.0))
nlp = hb.HybridNLP(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],
    f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4),
)


def measure(u, rng):
    """The system: here an oracle, but it could run a plant or a simulator."""
    return hb.Dataset.from_measurements(nlp, u=u, y=np.sin(u), rng=rng)


gp = hb.gp.GP(kernel=hb.gp.Matern52(), mean=hb.gp.Zero(), noise_std=0.0)

# Hybrid BO: a GP of the unknown state, inside the known equations
hybrid_regret = []
for seed in range(N_RUNS):
    rng = np.random.default_rng(seed)
    data = measure(hb.lhs_points([u], N_INITIAL, rng), rng)
    saa = hb.SAA(rng.standard_normal((25, 1)))  # the scenarios, once, before the loop
    for _ in range(N_ITERATIONS):
        surrogate = [gp.fit(data, inputs=[u], output=y, rng=rng)]
        solution = nlp.solve(
            saa,
            rng,
            surrogate=surrogate,
            objective=hb.obj.SoftEI(data.incumbents[-1], temperature=0.05),
            unknown_states_as_variables=False,
        )
        data = data.extend(measure(solution.u, rng))
    hybrid_regret.append(data.incumbents - OPTIMUM)

# Standard BO: a GP of the measured f, an unknown state of a problem with nothing known
fhat = hb.Var("fhat", start=(-10.0, 10.0))
standard_nlp = hb.HybridNLP(u=[u], y=[fhat], f=lambda v: v.fhat)
standard_regret = []
for seed in range(N_RUNS):
    rng = np.random.default_rng(seed)
    data = measure(hb.lhs_points([u], N_INITIAL, rng), rng)
    for _ in range(N_ITERATIONS):
        observed = hb.Dataset.from_measurements(
            standard_nlp, u=data.u, y=data.f[:, None], rng=rng
        )
        surrogate = [gp.fit(observed, inputs=[u], output=fhat, rng=rng)]
        solution = standard_nlp.solve(
            hb.Exact(),
            rng,
            surrogate=surrogate,
            objective=hb.obj.EI(data.incumbents[-1]),
        )
        data = data.extend(measure(solution.u, rng))
    standard_regret.append(data.incumbents - OPTIMUM)

for label, regret in (("hybrid BO", hybrid_regret), ("standard BO", standard_regret)):
    print(f"{label} median regret per point:", np.median(regret, axis=0).round(3))
