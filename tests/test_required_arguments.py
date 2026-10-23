"""Defaults never choose the method (ADR 0009): the arguments that say what we search
for have no default, and the settings are passed by keyword."""

import casadi as cas
import illustrative_example as ex
import pytest

from hybrid_bo import (
    EI,
    LCB,
    SAA,
    Configuration,
    HybridBOSettings,
    Linearization,
    SoftSAA,
    StandardBOSettings,
    StudySettings,
    oracle_measure,
)
from hybrid_bo.gp import GP, Matern52, Zero

# Each call leaves out one argument, so the type checker flags it as the test intends
REQUIRED = {
    "acquisition of standard BO": lambda: StandardBOSettings(gp=ex.gp),  # ty: ignore[missing-argument]
    "acquisition of hybrid BO": lambda: HybridBOSettings(  # ty: ignore[missing-argument]
        gp=ex.gp, approximation=SAA(n_scenarios=25)
    ),
    "approximation of hybrid BO": lambda: HybridBOSettings(gp=ex.gp, acquisition=EI()),  # ty: ignore[missing-argument]
    "n_scenarios of SAA": lambda: SAA(),  # ty: ignore[missing-argument]
    "n_scenarios of SoftSAA": lambda: SoftSAA(temperature=1.0),  # ty: ignore[missing-argument]
    "std_factor of LCB": lambda: LCB(),  # ty: ignore[missing-argument]
    "n_evaluations of a configuration": lambda: Configuration(  # ty: ignore[missing-argument]
        method=ex.methods["standard_bo"], n_initial=2
    ),
    "n_initial of a configuration": lambda: Configuration(  # ty: ignore[missing-argument]
        method=ex.methods["standard_bo"], n_evaluations=7
    ),
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
    "StudySettings": lambda: StudySettings(1, 0),  # ty: ignore[missing-argument, too-many-positional-arguments]
    "Configuration": lambda: Configuration(ex.methods["standard_bo"], 7, 2),  # ty: ignore[missing-argument, too-many-positional-arguments]
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
    """Passing the fields of a settings or choice dataclass by position raises
    `TypeError`."""
    with pytest.raises(TypeError, match="argument"):
        build()
