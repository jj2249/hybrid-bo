"""The settings of the BO methods, standard BO and hybrid BO.

They live apart from the methods (`bo.py`) because a run holds the settings it ran
with (`Run.settings`), and `bo.py` builds runs.
"""

from dataclasses import dataclass, field

from hybrid_bo.acquisition import (
    EI,
    LCB,
    SAA,
    Linearization,
    SoftSAA,
    _check_acquisition,
    _check_approximation,
)
from hybrid_bo.gp import GP
from hybrid_bo.solvers import Ipopt


@dataclass(frozen=True, kw_only=True)
class StandardBOSettings:
    """Settings of standard BO: the prior GP of u -> f, the acquisition function,
    the number of multistart starts for the acquisition problem and for fitting the
    GP, and the solver for both.

    Raises `TypeError` unless `acquisition` is an `EI` or an `LCB`.
    """

    gp: GP
    acquisition: EI | LCB
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = field(default_factory=Ipopt)

    def __post_init__(self) -> None:
        _check_acquisition(self.acquisition)


@dataclass(frozen=True, kw_only=True)
class HybridBOSettings:
    """Settings of hybrid BO: the prior GP of the unknown states over the arguments
    of h (`h_inputs`), the acquisition function (what is computed), its
    approximation (how it is computed, with its own settings), the number of
    multistart starts for the acquisition problem and for fitting the GP, and the
    solver for both.

    Raises `TypeError` unless `acquisition` is an `EI` or an `LCB` and
    `approximation` an `SAA`, a `SoftSAA` or a `Linearization`, `ValueError` for
    `LCB` with `SoftSAA`, and `NotImplementedError` for `LCB` with `Linearization`.
    """

    gp: GP
    acquisition: EI | LCB
    approximation: SAA | SoftSAA | Linearization
    n_starts: int = 10
    n_starts_gp: int = 10
    solver: Ipopt = field(default_factory=Ipopt)

    def __post_init__(self) -> None:
        _check_acquisition(self.acquisition)
        _check_approximation(self.acquisition, self.approximation)


# A BO method's settings: the one place a new method's settings join
MethodSettings = StandardBOSettings | HybridBOSettings

# The settings types, which saving tags by name: the methods' and their parts'
TYPES = (
    EI,
    LCB,
    SAA,
    SoftSAA,
    Linearization,
    StandardBOSettings,
    HybridBOSettings,
)
