"""The illustrative example with observation noise: the problem and oracle of
`illustrative_example`, measured with normal noise of standard deviation 0.01 on y.

One decision variable u, one mechanistic state x and one unknown state y. The known
equation is x + exp(x) = y; the unknown equation is y = sin(u), which only the oracle
knows. Everything is stated here rather than imported, so this experiment and the
noise-free one change separately; the illustrative example stays noise-free, since
the regression tests reuse its measurement. Run it, then plot the newest study:

    pixi run experiment illustrative_example_noisy [--n-runs 3 --plot ...]
    pixi run plot illustrative_example_noisy

`pixi run experiment illustrative_example_noisy --help` lists the flags.
"""

import functools

import casadi as cas
import cli

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


# Measurement: an oracle plays the real system, with noise on y
def oracle(v):
    """The unknown equation y = h(u), hidden from the methods."""
    return [cas.sin(v.u)]


measure = hb.oracle_measure(system, oracle, noise_std=0.01)

# The oracle optimum, at u = -0.45: the study measures regret from it, the methods
# never see it. It is the noise-free oracle's, as in illustrative_example, where it
# was found with hb.oracle_optimum(problem, oracle, np.random.default_rng(0),
# n_starts=200) and checked
optimum = -50.4608044704652

# The prior GP of every BO method, which each fits to its own data: a zero mean and an
# output scale times a Matern-5/2 kernel, both fitted within these bounds (in scaled
# units), and the measurement's noise as its observation noise. noise_std is in
# measured units, so it is the same number
outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
    noise_std=0.01,
)

# Methods: each takes (problem, measure, initial_points, n, rng), where n is the
# number of iterations, or of design points for the baselines. Hybrid BO substitutes
# the GP prediction for the unknown states, with SAA-EI and with SoftSAA-EI at a
# temperature of 10 (f is in the tens here)
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp, approximation=hb.SAA(unknown_states_as_variables=False)
        ),
    ),
    "hybrid_bo_soft": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp, approximation=hb.SoftSAA(10.0, unknown_states_as_variables=False)
        ),
    ),
    "standard_bo": functools.partial(
        hb.standard_bo, settings=hb.StandardBOSettings(gp=gp)
    ),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, n_initial=3, n_iterations=15, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the iteration
# figures draw its true curves, which are the noise-free oracle's
if __name__ == "__main__":
    cli.main(
        "illustrative_example_noisy",
        problem,
        measure,
        methods,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
