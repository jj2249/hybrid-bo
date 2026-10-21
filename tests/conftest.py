"""Shared test setup. pytest loads this file before the tests in this directory.

A fixture is a function marked `@pytest.fixture`: a test with an argument of the same
name gets the fixture's return value.
"""

import io
import pickle
from collections.abc import Callable
from typing import Any

import illustrative_example
import matplotlib
import pytest

from hybrid_bo import Problem

# Draw plots without opening windows
matplotlib.use("Agg")


def pytest_addoption(parser):
    """Adds the `--update-baselines` option to the pytest command line (a pytest
    hook, called by pytest at startup)."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        help="Overwrite stored regression results instead of comparing against them.",
    )


@pytest.fixture
def update_baselines(request):
    """Whether `--update-baselines` was passed (`request` is pytest's handle on the
    running test)."""
    return request.config.getoption("--update-baselines")


class _DataOnlyUnpickler(pickle.Unpickler):
    """Loads only the library's data classes, numpy and builtins: no user code."""

    allowed = (
        "hybrid_bo.run",
        "hybrid_bo.history",
        "hybrid_bo.gp",
        "hybrid_bo.acquisition",
        "hybrid_bo.settings",
        "hybrid_bo.solvers",
        "numpy",
        "builtins",
    )

    def find_class(self, module, name):
        """The class `module.name`, if the module is allowed."""
        if not module.startswith(self.allowed):
            msg = f"the pickle needs {module}.{name}"
            raise pickle.UnpicklingError(msg)
        return super().find_class(module, name)


@pytest.fixture
def data_only_pickle() -> Callable[[Any], Any]:
    """A function that pickles a value and loads it back with only the library's
    data classes, numpy and builtins, so loading fails if it needs user code."""
    return lambda value: _DataOnlyUnpickler(io.BytesIO(pickle.dumps(value))).load()


@pytest.fixture
def illustrative_problem() -> Problem:
    """The problem of the illustrative example (experiments/illustrative_example.py)."""
    return illustrative_example.problem
