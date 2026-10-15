import illustrative_example
import matplotlib
import pytest

from hybrid_bo import Problem

# Draw plots without opening windows
matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        help="Overwrite stored regression results instead of comparing against them.",
    )


@pytest.fixture
def update_baselines(request):
    return request.config.getoption("--update-baselines")


@pytest.fixture
def illustrative_problem() -> Problem:
    """The problem of the illustrative example (examples/illustrative_example.py)."""
    return illustrative_example.problem
