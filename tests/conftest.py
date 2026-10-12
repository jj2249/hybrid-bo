from pathlib import Path

import pytest
import yaml

from hybrid_bo import Config

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"

# Small, seeded runs so the tests are fast and deterministic
TEST_SETTINGS = {
    "seed": 0,
    "create_plots": False,
    "n_iterations_bo": 2,
    "n_runs_bo": 1,
    "save_extended_results": False,
}


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
def illustrative_config():
    with (EXAMPLES_DIR / "illustrative_example" / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    return Config(**(cfg | TEST_SETTINGS))
