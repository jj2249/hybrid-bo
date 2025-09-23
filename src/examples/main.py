from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from hybrid_bo import GP, Config, Problem
from hybrid_bo.optimizers import Optimizer
from hybrid_bo.routines import (
    hybrid_bo,
    latin_hypercube_sampling,
    standard_bo,
    uniform_sampling,
)

# Options:
# "forrester"
# "abstract_example"
# "flash_unit"
example: str = "abstract_example"

# Options:
# "hybrid_bo"
# "standard_bo"
# "latin_hypercube_sampling"
# "uniform_sampling"
# "compare_results"
routine: str = "hybrid_bo"

this_dir: Path = Path(__file__).parent.resolve()
results_dir: Path = this_dir / example


if __name__ == "__main__":
    example_pkg: ModuleType = import_module(example)
    problem: Problem = example_pkg.custom_problem()

    config_path: Path = this_dir / example / "config.yaml"
    with config_path.open("r") as stream:
        config_dict: dict = yaml.safe_load(stream)
    config_dict["problem"] = problem
    config: Config = Config(**config_dict)

    gp_hybrid_bo: GP = example_pkg.gp_hybrid_bo(problem.n_u)
    gp_standard_bo: GP = example_pkg.gp_standard_bo(problem.n_u)
    optimizer_hybrid_bo: Optimizer = example_pkg.optimizer_hybrid_bo()
    optimizer_standard_bo: Optimizer = example_pkg.optimizer_standard_bo()
    u_train_initial_complete: list[np.ndarray] = example_pkg.u_train_initial(
        problem,
        config,
    )

    create_plots_hybrid_bo: Callable = example_pkg.create_plots_hybrid_bo
    create_plots_standard_bo: Callable = example_pkg.create_plots_standard_bo
    create_plots_latin_hypercube_sampling: Callable = (
        example_pkg.create_plots_latin_hypercube_sampling
    )
    create_plots_uniform_sampling: Callable = example_pkg.create_plots_uniform_sampling

    match routine:
        case "hybrid_bo":
            hybrid_bo(
                problem,
                config,
                gp_hybrid_bo,
                u_train_initial_complete,
                optimizer_hybrid_bo,
                results_dir,
                create_plots_hybrid_bo,
            )
        case "standard_bo":
            standard_bo(
                problem,
                config,
                gp_standard_bo,
                u_train_initial_complete,
                optimizer_standard_bo,
                results_dir,
                create_plots_standard_bo,
            )
        case "latin_hypercube_sampling":
            latin_hypercube_sampling(
                problem,
                config,
                gp_standard_bo,
                u_train_initial_complete,
                results_dir,
                create_plots_latin_hypercube_sampling,
            )
        case "uniform_sampling":
            uniform_sampling(
                problem,
                config,
                gp_standard_bo,
                u_train_initial_complete,
                results_dir,
                create_plots_uniform_sampling,
            )
