"""Shared test setup. pytest loads this file before the tests in this directory.

A fixture is a function marked `@pytest.fixture`: a test with an argument of the same
name gets the fixture's return value.
"""

import illustrative_example
import pytest

from hybrid_bo import Problem


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


@pytest.fixture
def illustrative_problem() -> Problem:
    """The problem of the illustrative example (examples/illustrative_example.py)."""
    return illustrative_example.problem
