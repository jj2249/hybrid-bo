"""Defaults never choose the method (ADR 0009): the arguments that say what we search
for have no default, and the settings are passed by keyword."""

from typing import Any

import casadi as cas
import illustrative_example as ex
import pytest

from hybrid_bo import (
    EI,
    LCB,
    SAA,
    Linearization,
    SoftSAA,
    hybrid_trial_point,
    oracle_measure,
    standard_trial_point,
)
from hybrid_bo.gp import GP, Matern52, Zero
from hybrid_bo.settings import HybridBOSettings, StandardBOSettings

# Stand-ins for the positional arguments of the trial point functions: each call below
# fails on its signature, so they are never read
STANDARD: Any = (None,) * 4
HYBRID: Any = (None,) * 5

# Each call leaves out one argument, so the type checker flags it as the test intends
REQUIRED = {
    "acquisition of standard BO": lambda: StandardBOSettings(gp=ex.gp),  # ty: ignore[missing-argument]
    "acquisition of hybrid BO": lambda: HybridBOSettings(  # ty: ignore[missing-argument]
        gp=ex.gp, approximation=SAA(n_scenarios=25)
    ),
    "approximation of hybrid BO": lambda: HybridBOSettings(gp=ex.gp, acquisition=EI()),  # ty: ignore[missing-argument]
    "acquisition of standard_trial_point": lambda: standard_trial_point(*STANDARD),  # ty: ignore[missing-argument]
    "acquisition of hybrid_trial_point": lambda: hybrid_trial_point(  # ty: ignore[missing-argument]
        *HYBRID, approximation=Linearization()
    ),
    "approximation of hybrid_trial_point": lambda: hybrid_trial_point(  # ty: ignore[missing-argument]
        *HYBRID, acquisition=EI()
    ),
    "n_scenarios of SAA": lambda: SAA(),  # ty: ignore[missing-argument]
    "n_scenarios of SoftSAA": lambda: SoftSAA(temperature=1.0),  # ty: ignore[missing-argument]
    "std_factor of LCB": lambda: LCB(),  # ty: ignore[missing-argument]
    "noise_std of GP": lambda: GP(mean=Zero(), kernel=Matern52()),  # ty: ignore[missing-argument]
    "noise_std of the oracle measurement": lambda: oracle_measure(  # ty: ignore[missing-argument]
        ex.problem.system, oracle=lambda v: [cas.sin(v.u)]
    ),
}

# Each call passes the fields by position
POSITIONAL = {
    "LCB": lambda: LCB(1.0),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "SAA": lambda: SAA(25),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "SoftSAA": lambda: SoftSAA(1.0, 25),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "Linearization": lambda: Linearization(False),  # ty: ignore[too-many-positional-arguments]
    "StandardBOSettings": lambda: StandardBOSettings(ex.gp, EI()),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "HybridBOSettings": lambda: HybridBOSettings(ex.gp, EI(), SAA(n_scenarios=25)),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "GP": lambda: GP(Zero(), Matern52(), 0.0),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "standard_trial_point": lambda: standard_trial_point(*STANDARD, EI()),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "hybrid_trial_point": lambda: hybrid_trial_point(*HYBRID, EI(), Linearization()),  # ty: ignore[missing-argument, too-many-positional-arguments]
}


# `parametrize` runs the test once per entry, with the entry as the argument
@pytest.mark.parametrize("build", REQUIRED.values(), ids=REQUIRED.keys())
def test_method_choice_has_no_default(build):
    """Leaving out an argument that chooses the method raises `TypeError`."""
    # `raises` passes only if the block raises that exception
    with pytest.raises(TypeError, match="argument"):
        build()


@pytest.mark.parametrize("build", POSITIONAL.values(), ids=POSITIONAL.keys())
def test_settings_are_keyword_only(build):
    """Passing the fields of a settings or choice dataclass, or the choices of a trial
    point function, by position raises `TypeError`."""
    with pytest.raises(TypeError, match="argument"):
        build()
