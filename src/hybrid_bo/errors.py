"""The library's exceptions and warnings."""

from hybrid_bo.history import History
from hybrid_bo.run import Run


class HybridBOError(Exception):
    """Base of the library's errors.

    `history` holds the points evaluated before the error, when it was raised while
    evaluating; `run` holds the partial run, when it was raised during a method. The
    caller can save either.
    """

    history: History | None = None
    run: Run | None = None


class RootFindingError(HybridBOError):
    """No start of a root-finding solve gave a root within the bounds."""


class NoSolutionError(HybridBOError):
    """Every start of a multistart solve failed."""


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
