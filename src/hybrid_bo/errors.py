"""The library's exceptions and warnings."""

from hybrid_bo.history import History


class HybridBOError(Exception):
    """Base of the library's errors.

    `history` holds the points evaluated before the error, when it was raised while
    evaluating; the caller can save them.
    """

    history: History | None = None


class RootFindingError(HybridBOError):
    """No start of a root-finding solve gave a root within the bounds."""


class ReconstructionWarning(UserWarning):
    """The root-finding problem found other states than the oracle's.

    The measured variables then do not determine the states uniquely.
    """
