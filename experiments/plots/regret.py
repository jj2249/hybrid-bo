"""The median regret of each method over the evaluations."""

from typing import Any

import pandas as pd
from benchopt import BasePlot


class Plot(BasePlot):
    """Benchopt's own curves need a point per solver call, and a run is one call."""

    name = "regret"
    type = "scatter"
    options = {"dataset": ..., "objective": ...}

    # Benchopt checks that `plot` names the options, so it can't take **kwargs as its
    # base class does
    def plot(  # ty: ignore[invalid-method-override]
        self, df: pd.DataFrame, dataset: str, objective: str
    ) -> list[dict[str, Any]]:
        """A curve per method, the median over its runs at each evaluation."""
        df = df.query("dataset_name == @dataset and objective_name == @objective")
        curves = []
        for method, rows in df.groupby("solver_name"):
            median = rows.groupby("objective_evaluation")["objective_regret"].median()
            curves.append(
                {
                    "x": median.index.tolist(),
                    "y": median.tolist(),
                    "label": method,
                    **self.get_style(method),
                }
            )
        return curves

    def get_metadata(  # ty: ignore[invalid-method-override]
        self, df: pd.DataFrame, dataset: str, objective: str
    ) -> dict[str, str]:
        """The title, the labels of the axes and a log scale for the regret."""
        return {
            "title": f"{objective}\nData: {dataset}",
            "xlabel": "Evaluation",
            "ylabel": "Regret",
            "scale": "semilog-y",
        }
