"""The illustrative example with observation noise: the same problem and oracle as
`illustrative_example`, measured with normal noise of standard deviation 0.01 on y.

The illustrative example itself stays noise-free, since the regression tests reuse its
measurement. Every BO method's GP states the same noise. Run it, then plot the newest
study:

    pixi run experiment illustrative_example_noisy [--n-runs 3 --plot ...]
    pixi run plot illustrative_example_noisy

`pixi run experiment illustrative_example_noisy --help` lists the flags.
"""

import functools

import cli
from illustrative_example import optimum, oracle, problem, study_settings, system

import hybrid_bo as hb

# Measurement: the oracle, with noise on y
measure = hb.oracle_measure(system, oracle, noise_std=0.01)

# The prior GP of every BO method: the illustrative example's, with the measurement's
# noise, in the same (measured) units
outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))
lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))
gp = hb.gp.GP(
    mean=hb.gp.Zero(),
    kernel=outputscale * hb.gp.Matern52(lengthscale=lengthscale),
    noise_std=0.01,
)

# Methods: those of the illustrative example, with the noisy GP
methods = {
    "hybrid_bo": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp, approximation=hb.SAA(unknown_states_as_variables=False)
        ),
    ),
    "hybrid_bo_lin": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp, approximation=hb.Linearization(unknown_states_as_variables=False)
        ),
    ),
    "hybrid_bo_soft": functools.partial(
        hb.hybrid_bo,
        settings=hb.HybridBOSettings(
            gp=gp, approximation=hb.SoftSAA(1.0, unknown_states_as_variables=False)
        ),
    ),
    "standard_bo": functools.partial(
        hb.standard_bo, settings=hb.StandardBOSettings(gp=gp)
    ),
    "latin_hypercube_sampling": hb.latin_hypercube_sampling,
    "uniform_sampling": hb.uniform_sampling,
}

# Run or plot from the command line, with the illustrative example's study settings
# and oracle optimum. The true curves are the noise-free oracle's
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
