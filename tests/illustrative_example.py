"""The illustrative example of the paper, which the tests share.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the
measurement knows.
"""

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


# Measurement: this plays the real system
def measure(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The measured y = h(u) at u (N,), hidden from the library."""
    return np.sin(u)


# The prior GP of every BO method, which each fits to its own data: a zero mean and an
# output scale times a Matern-5/2 kernel, both fitted within these bounds (in scaled
# units), and no noise, as the measurement has none
outputscale = hb.gp.Param((0.1, 10.0))
lengthscale = hb.gp.Param((1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=hb.gp.Scale(hb.gp.Matern52(lengthscale=lengthscale), outputscale),
    noise_std=0.0,
)

# The initial points of every run
N_INITIAL = 2
