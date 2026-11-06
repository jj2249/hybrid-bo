"""The library's exceptions."""

from typing import TYPE_CHECKING

# Annotations only: dataset.py is imported by modules that import this one
if TYPE_CHECKING:
    from hybrid_bo.dataset import Dataset


class HybridBOError(Exception):
    """Base of the library's errors."""


class RootFindingError(HybridBOError):
    """No start of a root-finding solve gave a root within the bounds, or a measured
    variable or decision variable isn't finite. `dataset` holds the points that worked,
    if any were given."""

    dataset: "Dataset | None" = None


class NoSolutionError(HybridBOError):
    """Every start of a multistart solve failed."""
