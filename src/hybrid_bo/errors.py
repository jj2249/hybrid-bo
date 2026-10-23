"""The library's exceptions and warnings."""

from typing import TYPE_CHECKING

# Annotations only: run.py imports gp.py, which imports this module
if TYPE_CHECKING:
    from hybrid_bo.history import History
    from hybrid_bo.run import Run


class HybridBOError(Exception):
    """Base of the library's errors.

    `history` holds the points evaluated before the error, when it was raised while
    evaluating; `run` holds the partial run, when it was raised during a method. The
    caller can save either.
    """

    history: "History | None" = None
    run: "Run | None" = None


class RootFindingError(HybridBOError):
    """No start of a root-finding solve gave a root within the bounds."""


class NoSolutionError(HybridBOError):
    """Every start of a multistart solve failed."""


class StudyFormatError(ValueError):
    """A saved study is in a format this version can't load, such as one saved before
    GPs were data. Rerun it, or load it with the version that saved it."""


class ReconstructionWarning(UserWarning):
    """The root-finding problem found a different root from the oracle values.

    Only raised with an oracle. Noise alone doesn't trigger it: the reconstruction is
    compared with the root nearest the oracle values, for the same noisy measured
    variables. A warning means the measured variables don't determine the states
    uniquely, so the method may learn from states the system didn't settle at. Measure
    other variables, or narrow the start bounds of the unmeasured ones to the right
    branch. Silence it with `warnings.filterwarnings("ignore",
    category=ReconstructionWarning)`.
    """


class OracleOptimumWarning(UserWarning):
    """An evaluated point has an oracle objective below the stated oracle optimum.

    The stated value can't then be the optimum, so every regret measured from it is
    wrong. Find the optimum again, e.g. with `oracle_optimum` from more starts.
    """


class ReplayWarning(UserWarning):
    """An iteration figure's replay of hybrid BO's acquisition function didn't find
    the stored solution at the trial point.

    The replay solves each scenario's equations with u fixed by Newton's method,
    starting at the trial point from the stored solution. A warning means the root it
    found there has other states or another acquisition value, or none was found: the root-finder left the
    branch the optimizer chose, so the panel may not show the acquisition function the
    run optimized. Silence it with `warnings.filterwarnings("ignore",
    category=ReplayWarning)`.
    """
