"""Every method of the experiments, as a Benchopt solver."""

from pathlib import Path
from typing import Any

import numpy as np
from benchmark_utils import FOLDER, MODULES, experiment
from benchopt import BaseSolver

import hybrid_bo as hb


class Solver(BaseSolver):
    """Runs the configuration `label` of an experiment and saves the run in full."""

    name = "method"
    sampling_strategy = "run_once"
    parameters = {
        "label": sorted(
            {label for module in MODULES for label in experiment(module).configurations}
        )
    }
    label: str

    def skip(self, **objective: Any) -> tuple[bool, str | None]:
        """Skips an experiment that doesn't compare this method."""
        chosen = objective["experiment"]
        if self.label not in chosen.configurations:
            return True, f"{chosen.__name__} has no method {self.label}"
        return False, None

    def set_objective(self, **objective: Any) -> None:
        """Keeps the experiment and its code. Run i of every method has the same seed,
        so the same initial points."""
        self.experiment = objective["experiment"]
        self.code = objective["code"]
        self.seed = self.get_seed(use_repetition=True)

    def run(self, stop_val: object) -> None:
        """Draws the initial points and runs the method. A `HybridBOError` keeps the
        partial run, which then reports as incomplete."""
        configuration = self.experiment.configurations[self.label]
        problem, measure = self.experiment.problem, self.experiment.measure
        rng = np.random.default_rng(self.seed)
        initial_points = hb.initial_points(
            problem.system.u, configuration.n_initial, rng
        )
        n = configuration.n_evaluations - configuration.n_initial
        try:
            self.result = configuration.method(
                problem, measure, initial_points, n, rng=rng
            )
        except hb.HybridBOError as error:
            if error.run is None:
                raise
            self.result = error.run
        self.budget = configuration.n_evaluations
        hb.save_run(self.result, self._replaced_run_file())

    def _replaced_run_file(self) -> Path:
        """Where the run is saved, with any earlier file of it gone. Not Benchopt's
        run folder, which holds only what its run didn't find in the cache."""
        module = self.experiment.__name__
        folder = FOLDER / "outputs" / "runs" / module / self.code / self.label
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / f"seed-{self.seed}.json"
        file.unlink(missing_ok=True)
        return file

    def get_result(self) -> dict[str, Any]:
        """The run, and the budget it had."""
        return {"run": self.result, "budget": self.budget}
