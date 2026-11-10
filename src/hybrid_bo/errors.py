"""The library's exceptions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hybrid_bo.dataset import Dataset


class RootFindingError(Exception):
    """No root of g within the bounds, or a value that isn't finite. `dataset` holds
    the points that worked, if any were given."""

    dataset: "Dataset | None" = None


class NoSolutionError(Exception):
    """Every start of a multistart solve failed."""
