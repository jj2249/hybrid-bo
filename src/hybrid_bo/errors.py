"""The library's exceptions."""

from hybrid_bo.dataset import Dataset


class RootFindingError(Exception):
    """No start of a root-finding solve gave a root within the bounds, or a measured
    variable or decision variable isn't finite. `dataset` holds the points that worked,
    if any were given."""

    dataset: Dataset | None = None


class NoSolutionError(Exception):
    """Every start of a multistart solve failed."""
