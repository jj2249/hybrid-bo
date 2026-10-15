"""Polars views of a study, for quick analysis: one row per evaluation
(`evaluations`), per method (`summary`) or per fitted hyperparameter entry
(`hyperparameters`).

Polars is optional: it is imported on first use, so the rest of the library works
without it (the `hybrid-bo[analysis]` extra installs it). Regret and convergence are
over evaluations, not iterations, since the initial points are not free.
"""

import re
from typing import TYPE_CHECKING

import numpy as np

from hybrid_bo.history import History, incumbents, oracle_incumbents
from hybrid_bo.run import Run
from hybrid_bo.studies import Study

if TYPE_CHECKING:
    import polars as pl


def _polars():
    """The `polars` module. Raises `ImportError`, naming the extra, when it isn't
    installed."""
    try:
        import polars as pl
    except ImportError as error:
        msg = "the analysis views need Polars: install hybrid-bo[analysis] (or polars)"
        raise ImportError(msg) from error
    return pl


def _names(record: dict[str, str], block: str, n: int) -> list[str]:
    """The names of the n variables of `block` ("u", "x" or "y") in a study's record,
    or their positions as text when the record doesn't name n of them."""
    names = re.findall(r"Var\(name='([^']*)'", record.get(block, ""))
    return names if len(names) == n else [str(i) for i in range(n)]


def _run_evaluations(
    run: Run, record: dict[str, str], optimum: float | None
) -> "pl.DataFrame":
    """The rows of `evaluations` for one run."""
    pl = _polars()
    history: History = run.history
    n = len(history.f)
    n_iterations = n - history.n_initial
    has_oracle = history.f_oracle is not None

    def floats(values: np.ndarray | list[float | None] | None) -> "pl.Series":
        """A Float64 column, all null when `values` is None."""
        return pl.Series([None] * n if values is None else values, dtype=pl.Float64)

    # Where the row is: the method, the run, and its evaluation and iteration
    columns = {
        "method": pl.Series([run.method] * n, dtype=pl.String),
        "run": pl.Series([run.index] * n, dtype=pl.Int64),
        "evaluation": pl.Series(range(1, n + 1), dtype=pl.Int64),
        "iteration": pl.Series(
            [None] * history.n_initial + list(range(1, n_iterations + 1)),
            dtype=pl.Int64,
        ),
    }

    # The variables and f, then their oracle values (null without an oracle)
    blocks = {"u": history.u, "x": history.x, "y": history.y}
    for block, values in blocks.items():
        for name, column in zip(_names(record, block, values.shape[1]), values.T):
            columns[f"{block}_{name}"] = floats(column)
    columns["f"] = floats(history.f)
    oracle_blocks = {"x": history.x_oracle, "y": history.y_oracle}
    for block, values in oracle_blocks.items():
        for i, name in enumerate(_names(record, block, blocks[block].shape[1])):
            columns[f"{block}_oracle_{name}"] = floats(
                None if values is None else values[:, i]
            )
    columns["f_oracle"] = floats(history.f_oracle)

    # The incumbents, the regret and the acquisition value behind each trial point
    oracle_incumbent = oracle_incumbents(history) if has_oracle else None
    has_regret = oracle_incumbent is not None and optimum is not None
    columns["incumbent"] = floats(incumbents(history))
    columns["oracle_incumbent"] = floats(oracle_incumbent)
    columns["regret"] = floats(oracle_incumbent - optimum if has_regret else None)
    values = [solution.value for solution in run.acquisitions[:n_iterations]]
    columns["acquisition"] = floats(
        [None] * history.n_initial + values + [None] * (n_iterations - len(values))
    )
    return pl.DataFrame(columns)


def evaluations(study: Study, optimum: float | None = None) -> "pl.DataFrame":
    """One row per evaluation of every run, in the study's order, with columns:

    - `method`, `run` (its index), `evaluation` (from 1, initial points included)
      and `iteration` (from 1; null for initial points; a baseline's design points
      count as iterations);
    - `u_<name>`, `x_<name>`, `y_<name>`, one per variable, and `f`;
    - `x_oracle_<name>`, `y_oracle_<name>` and `f_oracle`, null without an oracle;
    - `incumbent` (`hb.incumbents`), `oracle_incumbent` (`hb.oracle_incumbents`,
      null without an oracle) and `regret`: the oracle incumbent minus the oracle
      optimum, which is `optimum` when given, else the study's recorded one, else
      there is no regret (null);
    - `acquisition`: the acquisition problem's objective at the trial point (-EI
      for EI); null for initial points and the baselines.

    Variables are named after the study's record, or numbered from 0 when it names
    none. Raises `ImportError` without Polars.
    """
    pl = _polars()
    if optimum is None and "oracle_optimum" in study.record:
        optimum = float(study.record["oracle_optimum"])
    frames = [_run_evaluations(run, study.record, optimum) for run in study.runs]
    return pl.concat(frames) if frames else pl.DataFrame()


def summary(study: Study) -> "pl.DataFrame":
    """One row per method, in the study's order, with columns:

    - `method`, `runs` (the runs in the study) and `failed` (the runs of the
      settings that didn't finish: those that stopped short and those that left
      nothing to keep);
    - `<value>_median` and `<value>_iqr` (interquartile range, with numpy's default
      linear quantiles) over runs of the final `incumbent`, `oracle_incumbent` and
      `regret`, as in `evaluations` (null where those are null).

    Raises `ImportError` without Polars.
    """
    pl = _polars()
    settings = study.settings
    n_evaluations = settings.n_initial + settings.n_iterations
    finals = evaluations(study).group_by("method", "run", maintain_order=True).last()

    def spread(name: str) -> list["pl.Expr"]:
        """The median and interquartile range of a column."""
        column = pl.col(name)
        iqr = column.quantile(0.75, "linear") - column.quantile(0.25, "linear")
        return [column.median().alias(f"{name}_median"), iqr.alias(f"{name}_iqr")]

    return finals.group_by("method", maintain_order=True).agg(
        pl.len().cast(pl.Int64).alias("runs"),
        (
            settings.n_runs
            - (pl.col("evaluation") == n_evaluations).sum().cast(pl.Int64)
        ).alias("failed"),
        *spread("incumbent"),
        *spread("oracle_incumbent"),
        *spread("regret"),
    )


def hyperparameters(study: Study) -> "pl.DataFrame":
    """The fitted hyperparameters over the iterations: one row per run, iteration and
    hyperparameter entry, with columns `method`, `run`, `iteration` (from 1, as in
    `evaluations`), `hyperparameter` (its name), `entry` (its position in the
    hyperparameter, from 0; e.g. one lengthscale per input) and `value`. The
    baselines fit nothing, so they have no rows. Raises `ImportError` without
    Polars.
    """
    pl = _polars()
    rows = [
        (run.method, run.index, iteration, name, entry, float(value))
        for run in study.runs
        for iteration, surrogate in enumerate(run.surrogates, start=1)
        for name, values in surrogate.hyperparameters.items()
        for entry, value in enumerate(values)
    ]
    schema = {
        "method": pl.String,
        "run": pl.Int64,
        "iteration": pl.Int64,
        "hyperparameter": pl.String,
        "entry": pl.Int64,
        "value": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")
