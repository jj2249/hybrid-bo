"""The illustrative example of the paper, which the tests share: x + exp(x) = y is
known, and y = sin(u) is known only to `measure`."""

import casadi as cas
import numpy as np

import hybrid_bo as hb

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

# Standard BO: a GP of f itself, an unknown state fhat
fhat = hb.Var("fhat", start=(-10.0, 10.0))
standard_nlp = hb.HybridNLP(u=[u], y=[fhat], f=lambda v: v.fhat)


def measure(u: np.ndarray, rng: np.random.Generator) -> hb.Dataset:
    """The system at u, a point or a set: y = sin(u), with x solved from g."""
    return hb.Dataset.from_measurements(nlp, u=u, y=np.sin(u), rng=rng)


gp = hb.gp.GP(kernel=hb.gp.Matern52(), mean=hb.gp.Zero(), noise_std=0.0)

N_INITIAL = 2
