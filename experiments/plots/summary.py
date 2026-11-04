"""One row per method: its runs and how they ended."""

from typing import Any

import pandas as pd
from benchopt import BasePlot


class Plot(BasePlot):
    """A failed run has no row in the results, and an incomplete one stopped early
    with a `HybridBOError`, so both show here against the other methods' runs."""

    name = "summary"
    type = "table"
    options = {"dataset": ..., "objective": ...}

    # Benchopt checks that `plot` names the options, so it can't take **kwargs as its
    # base class does
    def plot(  # ty: ignore[invalid-method-override]
        self, df: pd.DataFrame, dataset: str, objective: str
    ) -> list[list[Any]]:
        """The number of runs, of complete runs, the final regret's quartiles and the
        number of runs below the stated optimum, whose regret is wrong."""
        df = df.query("dataset_name == @dataset and objective_name == @objective")
        rows = []
        for method, method_rows in df.groupby("solver_name"):
            final = method_rows.groupby("idx_rep").last()
            quartiles = final["objective_regret"].quantile([0.25, 0.5, 0.75])
            rows.append(
                [
                    method,
                    len(final),
                    int(final["objective_complete"].sum()),
                    *quartiles.tolist(),
                    int(final["objective_below_optimum"].sum()),
                ]
            )
        return rows

    def get_metadata(  # ty: ignore[invalid-method-override]
        self, df: pd.DataFrame, dataset: str, objective: str
    ) -> dict[str, Any]:
        """The title and the columns."""
        return {
            "title": f"{objective}\nData: {dataset}",
            "columns": [
                "method",
                "runs",
                "complete",
                "regret Q1",
                "median",
                "Q3",
                "below optimum",
            ],
            "default_order_ascending": True,
            "default_order_column": 3,
        }
