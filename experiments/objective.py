"""How Benchopt scores a run: one row of results per evaluation."""

from typing import Any

from benchopt import BaseObjective

import hybrid_bo as hb


class Objective(BaseObjective):
    """Reports each evaluation's incumbents and regret, and whether the run spent its
    budget. A run is one call of a method, so Benchopt calls it once (`run_once`)."""

    name = "hybrid-bo"
    sampling_strategy = "run_once"
    min_benchopt_version = "1.10"

    def set_data(self, **data: Any) -> None:
        """Keeps what the dataset gives, the experiment and its code."""
        self.data = data

    def get_objective(self) -> dict[str, Any]:
        """What a method needs: the experiment, and the code it was run with."""
        return self.data

    def evaluate_result(self, **result: Any) -> list[dict[str, float]]:
        """One row per evaluation of the method's `run`, which had the evaluation
        budget `budget`. `value` is Benchopt's main metric: the objective without
        noise at the incumbent."""
        history = result["run"].history
        oracle = hb.oracle_incumbents(history)
        incumbents = hb.incumbents(history)
        complete = float(len(incumbents) == result["budget"])
        optimum = self.data["experiment"].optimum
        return [
            {
                "value": float(oracle[i]),
                "evaluation": i + 1,
                "incumbent": float(incumbents[i]),
                "regret": float(oracle[i] - optimum),
                "complete": complete,
            }
            for i in range(len(incumbents))
        ]

    def get_one_result(self) -> dict[str, Any]:
        """Not used: `benchopt test` has no dummy run to score."""
        return {"run": None, "budget": 0}
