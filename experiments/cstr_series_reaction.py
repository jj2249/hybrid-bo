"""A CSTR running the series reactions A -> B -> C, with the temperature to choose.

One decision variable, the temperature T; two mechanistic states, the outlet
concentrations c_A and c_B; and one unknown state, the rate constant k_1 of the
catalytic step A -> B. The known equations are the steady-state mole balances on A and
B, with the known Arrhenius kinetics of the breakdown B -> C; the catalyst's kinetics
k_1(T) are known only to the oracle. The objective is the most B at the outlet. Since
the breakdown has the higher activation energy, the best temperature lies inside the
bounds. Run it, then plot the newest study:

    pixi run experiment cstr_series_reaction [--n-runs 3 --plot ...]
    pixi run plot cstr_series_reaction
"""

import functools

import casadi as cas
import cli

import hybrid_bo as hb

# Fixed values: residence time V/F [min], feed concentration of A [mol/L], and the
# reference temperature [K] the Arrhenius laws are written about, which keeps their
# exponents of order 1
tau = 10.0
c_A0 = 1.0
T_ref = 350.0


def k_2(T):
    """The known rate constant of the breakdown B -> C [1/min], E_2/R = 10000 K."""
    return 0.02 * cas.exp(-10000.0 * (1 / T - 1 / T_ref))


# Problem: what we know about the physics, and what we want. Concentrations lie between
# 0 and the feed's. The unknown state is ln k_1 rather than k_1, so every GP scenario
# has a positive rate constant (a negative one would push c_A above the feed's, where
# the balances have no solution), and by Arrhenius ln k_1 is linear in 1/T
T = hb.Var("T", lower=300.0, upper=400.0)
c_A = hb.Var("c_A", lower=0.0, upper=c_A0)
c_B = hb.Var("c_B", lower=0.0, upper=c_A0)
ln_k_1 = hb.Var("ln_k_1", start=(-4.5, 0.5))

system = hb.System(
    u=[T],
    x=[c_A, c_B],
    y=[ln_k_1],
    g=lambda v: [
        c_A0 - v.c_A - tau * cas.exp(v.ln_k_1) * v.c_A,  # mole balance on A
        tau * cas.exp(v.ln_k_1) * v.c_A
        - v.c_B
        - tau * k_2(v.T) * v.c_B,  # mole balance on B
    ],
    h_inputs=[T],  # ln k_1 = h(T)
    measured=[ln_k_1],
)
problem = hb.Problem(system, f=lambda v: -v.c_B)


# Measurement: an oracle plays the real reactor. In a plant, k_1 would be inferred from
# the measured outlet c_A as (c_A0 - c_A) / (tau c_A)
def oracle(v):
    """The catalyst's kinetics ln k_1(T), k_1 in 1/min and E_1/R = 5000 K, hidden from
    the methods."""
    return [cas.log(0.2) - 5000.0 * (1 / v.T - 1 / T_ref)]


measure = hb.oracle_measure(system, oracle, noise_std=0.0)

# The oracle optimum, at T = 350 K: the study measures regret from it, the methods
# never see it. Found once with
# hb.oracle_optimum(problem, oracle, np.random.default_rng(0), n_starts=200), and
# checked against the closed form of c_B on a grid of T, which peaks at c_B = 5/9
optimum = -0.5555555555900221

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

# Methods: hybrid BO with SoftSAA-EI at a temperature of 5e-4, about 1e-3 times the
# optimum's magnitude (about 0.56), standard BO, and Latin hypercube sampling
methods = {
    "hybrid_bo_soft": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp,
            acquisition=hb.EI(),
            approximation=hb.SoftSAA(
                temperature=5e-4, n_scenarios=25, unknown_states_as_variables=False
            ),
        ),
    ),
    "standard_bo": functools.partial(
        hb.standard_bo, settings=hb.StandardBOSettings(gp=gp, acquisition=hb.EI())
    ),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
}

# Configurations: every method with the same budget, 7 evaluations, the first 3 of
# them initial points (so 4 iterations)
configurations = {
    label: hb.Configuration(method=method, n_evaluations=7, n_initial=3)
    for label, method in methods.items()
}

# The study, which the command line's flags override. Seed None draws a fresh seed,
# which the saved study keeps, so it can be run again exactly
study_settings = hb.StudySettings(n_runs=10, seed=None)

# Run or plot from the command line. The oracle measurement is cheap, so the iteration
# figures draw its true curves
if __name__ == "__main__":
    cli.main(
        "cstr_series_reaction",
        problem,
        measure,
        configurations,
        study_settings,
        optimum=optimum,
        true_curves=True,
    )
