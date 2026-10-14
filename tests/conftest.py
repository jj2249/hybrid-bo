from pathlib import Path

import casadi as cas
import pytest
import yaml

from hybrid_bo import Problem, System, Var
from hybrid_bo.legacy import Config

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


@pytest.fixture
def illustrative_problem() -> Problem:
    """The illustrative example in the new style (DESIGN.md, target example)."""
    u = Var("u", -2.0, 2.0)
    x = Var("x", start=(-2.5, 1.0))
    y = Var("y", start=(-2.0, 2.0))
    system = System(
        u=[u],
        x=[x],
        y=[y],
        g=lambda v: [v.x + cas.exp(v.x) - v.y],
        h_inputs=[u],
        measured=[y],
    )
    return Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))
