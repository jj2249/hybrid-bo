from importlib import import_module
from pathlib import Path

import yaml

from hybrid_bo.legacy import Config
from hybrid_bo.legacy.routines import (
    do_hybrid_bo,
    do_latin_hypercube_sampling,
    do_standard_bo,
    do_uniform_sampling,
)

# Options:
# "illustrative_example"
# "flash_unit" (only at the `pre-refactor` tag; see examples/flash_unit/README.md)
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

    # One generator per run; each run's initial points come from its own generator
    rngs = config.rng.spawn(config.n_runs_bo)
    initial_points = [pkg.get_u_initial(problem, rng) for rng in rngs]

    results_dir = example_dir / routine
    # Plot functions are only needed (and looked up) when plotting is on
    plots = getattr(pkg, f"create_plots_{routine}") if config.create_plots else None

    match routine:
        case "hybrid_bo":
            do_hybrid_bo(
                problem,
                config,
                gp=pkg.get_gp_hybrid_bo(problem.n_u),
                u_initial_complete=initial_points,
                rngs=rngs,
                optimizer=pkg.get_optimizer_hybrid_bo(),
                results_dir=results_dir,
                create_plots=plots,
            )
        case "standard_bo":
            do_standard_bo(
                problem,
                config,
                gp=pkg.get_gp_standard_bo(problem.n_u),
                u_initial_complete=initial_points,
                rngs=rngs,
                optimizer=pkg.get_optimizer_standard_bo(),
                results_dir=results_dir,
                create_plots=plots,
            )
        case "latin_hypercube_sampling":
            do_latin_hypercube_sampling(
                problem,
                config,
                u_initial_complete=initial_points,
                rngs=rngs,
                results_dir=results_dir,
                create_plots=plots,
            )
        case "uniform_sampling":
            do_uniform_sampling(
                problem,
                config,
                u_initial_complete=initial_points,
                rngs=rngs,
                results_dir=results_dir,
                create_plots=plots,
            )
        case _:
            msg = f"Unknown routine {routine!r}"
            raise ValueError(msg)
