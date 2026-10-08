from importlib import import_module
from pathlib import Path

import yaml

from hybrid_bo import Config
from hybrid_bo.routines import (
    do_hybrid_bo,
    do_latin_hypercube_sampling,
    do_standard_bo,
    do_uniform_sampling,
)

# Options:
# "illustrative_example"
# "flash_unit"
example = "illustrative_example"

# Options:
# "hybrid_bo"
# "standard_bo"
# "latin_hypercube_sampling"
# "uniform_sampling"
routine = "hybrid_bo"


project_dir = Path(__file__).parent.resolve()
example_dir = project_dir / example


if __name__ == "__main__":
    with (example_dir / "config.yaml").open() as file:
        config = Config(**yaml.safe_load(file))

    pkg = import_module(example)

    problem = pkg.get_custom_problem(config)
    initial_points = pkg.get_u_initial(problem, config)

    routines = {
        "hybrid_bo": (
            do_hybrid_bo,
            pkg.get_gp_hybrid_bo,
            pkg.get_optimizer_hybrid_bo,
            pkg.create_plots_hybrid_bo,
        ),
        "standard_bo": (
            do_standard_bo,
            pkg.get_gp_standard_bo,
            pkg.get_optimizer_standard_bo,
            pkg.create_plots_standard_bo,
        ),
        "latin_hypercube_sampling": (
            do_latin_hypercube_sampling,
            pkg.get_gp_standard_bo,
            None,
            pkg.create_plots_latin_hypercube_sampling,
        ),
        "uniform_sampling": (
            do_uniform_sampling,
            pkg.get_gp_standard_bo,
            None,
            pkg.create_plots_uniform_sampling,
        ),
    }

    run, make_gp, make_optimizer, make_plots = routines[routine]

    run(
        problem,
        config,
        make_gp(problem.n_u),
        initial_points,
        make_optimizer() if make_optimizer else None,
        example_dir / routine,
        make_plots if config.create_plots else None,
    )