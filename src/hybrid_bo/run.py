"""A run of a method, as pure data (ADR 0003)."""

from dataclasses import dataclass, field

from hybrid_bo.acquisition import AcquisitionSolution
from hybrid_bo.dataset import Dataset
from hybrid_bo.gp import FittedGP
from hybrid_bo.hybrid_model import Scenarios
from hybrid_bo.settings import MethodSettings


@dataclass(frozen=True)
class Run:
    """One run of a method: the evaluated points, the fitted surrogate models, the
    acquisition problems' solutions and the method's settings.

    Holds only data (arrays, numbers, strings and frozen dataclasses of them), never
    the problem or any user callable, so it saves as JSON (`hb.save_run`) and loads
    without the code that produced it (ADR 0003).
    `method` is the method's name. `surrogates` has one fitted surrogate model per
    iteration, the one behind that iteration's trial point: a tuple of one fitted GP
    per unknown state, in the order of the system's y, for hybrid BO, and a 1-tuple of
    the GP of f for standard BO. `acquisitions` has the solution of that iteration's
    acquisition problem; both are empty for the baselines. `scenarios` are hybrid BO's,
    the same in every iteration, and None for the other methods. `settings` are the BO
    method's settings in full, which the iteration plots replay, and None for the
    baselines. So every saved run has the same fields.
    """

    dataset: Dataset
    method: str
    surrogates: list[tuple[FittedGP, ...]] = field(default_factory=list)
    acquisitions: list[AcquisitionSolution] = field(default_factory=list)
    scenarios: Scenarios | None = None
    settings: MethodSettings | None = None
