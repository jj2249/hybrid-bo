"""The Polars views of a study: one row per evaluation, per method and per fitted
hyperparameter entry, on a study built by hand.

pytest runs every function whose name starts with `test_`.
"""

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

import hybrid_bo as hb
from hybrid_bo.acquisition import AcquisitionSolution
from hybrid_bo.gp import GP, FittedGP, Matern52, Scale, Scaler, Zero

OPTIMUM = -1.0

# The variables as a study's artifact holds them (`repr` of each `Var`)
ARTIFACT = {
    "u": "Var(name='flow', lower=0.0, upper=1.0, start=(0.0, 1.0))",
    "x": "Var(name='a', lower=-inf, upper=inf, start=(0.0, 1.0)); "
    "Var(name='b', lower=-inf, upper=inf, start=(0.0, 1.0))",
    "y": "Var(name='rate', lower=-inf, upper=inf, start=(0.0, 1.0))",
    "oracle_optimum": repr(OPTIMUM),
}
SETTINGS = hb.StudySettings(n_runs=2, n_initial=2, n_iterations=2, seed=0)


def history(f: list[float], f_oracle: list[float] | None) -> hb.History:
    """A history of 2 initial points and the rest iterations, with the given measured
    and oracle objective values; u counts the points, x and y are copies of it."""
    n = len(f)
    u = np.arange(n, dtype=float)[:, np.newaxis]
    with_oracle = f_oracle is not None
    return hb.History(
        u=u,
        x=np.hstack([u, -u]),
        y=2 * u,
        f=np.array(f),
        x_oracle=np.hstack([u, -u]) + 0.5 if with_oracle else None,
        y_oracle=2 * u + 0.5 if with_oracle else None,
        f_oracle=np.array(f_oracle) if with_oracle else None,
        n_initial=2,
    )


def surrogate(lengthscale: list[float]) -> FittedGP:
    """A fitted GP of two inputs with an output scale of 1 and the given
    lengthscales."""
    scaler = Scaler(np.zeros(1), np.ones(1))
    return FittedGP(
        GP(
            mean=Zero(),
            kernel=Scale(Matern52(tuple(lengthscale)), outputscale=1.0),
            noise_std=0.0,
        ),
        X=np.zeros((2, 2)),
        Y=np.zeros(2),
        x_scaler=scaler,
        y_scaler=scaler,
    )


def bo_run(index: int, f: list[float], f_oracle: list[float]) -> hb.Run:
    """A standard BO run with one surrogate (one GP, of f) and acquisition solution
    per iteration."""
    n_iterations = len(f) - 2
    return hb.Run(
        history(f, f_oracle),
        artifact={},
        method="bo",
        index=index,
        surrogates=[(surrogate([0.1 * i, 0.2 * i]),) for i in range(n_iterations)],
        acquisitions=[AcquisitionSolution(-float(i)) for i in range(n_iterations)],
        settings=hb.StandardBOSettings(
            gp=GP(mean=Zero(), kernel=Matern52(), noise_std=0.0), acquisition=hb.EI()
        ),
    )


def baseline_run(index: int, f: list[float], f_oracle: list[float]) -> hb.Run:
    """A baseline run: no surrogates and no acquisition solutions."""
    return hb.Run(history(f, f_oracle), artifact={}, method="lhs", index=index)


# Noise: the second BO run's best measured point is not its best oracle point. The
# second baseline run failed after its initial points
RUNS = (
    bo_run(0, [3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]),
    bo_run(1, [5.0, 4.0, 1.0, 2.0], [5.0, 4.0, 3.0, 0.0]),
    baseline_run(0, [2.0, 3.0, 1.0, 4.0], [2.0, 3.0, 1.0, 4.0]),
    baseline_run(1, [2.0, 3.0], [2.0, 3.0]),
)
STUDY = hb.Study(RUNS, SETTINGS, name="by hand", artifact=ARTIFACT)


def rows_of(frame: pl.DataFrame, method: str, run: int) -> pl.DataFrame:
    """The rows of one run of one method."""
    return frame.filter((pl.col("method") == method) & (pl.col("run") == run))


# evaluations


def test_evaluations_has_one_row_per_evaluation_with_named_columns():
    """Each evaluation is a row, in study order, with the variables named after the
    recorded `Var`s and the oracle values beside them."""
    frame = hb.evaluations(STUDY)

    assert frame.columns == [
        "method",
        "run",
        "evaluation",
        "iteration",
        "u_flow",
        "x_a",
        "x_b",
        "y_rate",
        "f",
        "x_oracle_a",
        "x_oracle_b",
        "y_oracle_rate",
        "f_oracle",
        "incumbent",
        "oracle_incumbent",
        "regret",
        "acquisition",
    ]
    assert frame.height == sum(len(run.history.f) for run in RUNS)
    run = rows_of(frame, "bo", 1)
    np.testing.assert_array_equal(run["x_b"], RUNS[1].history.x[:, 1])
    np.testing.assert_array_equal(run["y_oracle_rate"], 2 * np.arange(4) + 0.5)


def test_evaluations_count_initial_points_as_evaluations_but_not_iterations():
    """`evaluation` counts from 1 over every point; `iteration` is null for the
    initial points, then counts from 1."""
    run = rows_of(hb.evaluations(STUDY), "bo", 0)

    assert run["evaluation"].to_list() == [1, 2, 3, 4]
    assert run["iteration"].to_list() == [None, None, 1, 2]


def test_incumbents_match_the_history_functions():
    """`incumbent` and `oracle_incumbent` are `hb.incumbents` and
    `hb.oracle_incumbents` of each run's history."""
    frame = hb.evaluations(STUDY)

    for run in RUNS:
        rows = rows_of(frame, run.method, run.index)
        np.testing.assert_array_equal(rows["incumbent"], hb.incumbents(run.history))
        np.testing.assert_array_equal(
            rows["oracle_incumbent"], hb.oracle_incumbents(run.history)
        )


def test_regret_is_the_oracle_incumbent_minus_the_recorded_optimum():
    """Without an `optimum`, regret is measured from the study's recorded one. Under
    noise it follows the incumbent's point, not the best oracle value."""
    run = rows_of(hb.evaluations(STUDY), "bo", 1)

    # The oracle incumbents are 5, 4, 3, 3; the optimum is -1
    assert run["regret"].to_list() == [6.0, 5.0, 4.0, 4.0]


def test_a_given_optimum_overrides_the_recorded_one():
    """`optimum` replaces the study's oracle optimum."""
    run = rows_of(hb.evaluations(STUDY, optimum=0.0), "bo", 1)

    assert run["regret"].to_list() == [5.0, 4.0, 3.0, 3.0]


def test_regret_is_null_without_an_optimum():
    """A study without an oracle optimum and no `optimum` given has no regret."""
    artifact = {k: v for k, v in ARTIFACT.items() if k != "oracle_optimum"}
    study = hb.Study(RUNS, SETTINGS, name=None, artifact=artifact)

    regret = hb.evaluations(study)["regret"]

    assert regret.null_count() == len(regret)


def test_oracle_columns_are_null_without_oracle_values():
    """A run without oracle values has null oracle values, oracle incumbents and
    regret, with the same columns as a run that has them."""
    run = hb.Run(history([1.0, 0.0], None), artifact={}, method="lhs")
    study = hb.Study((run,), SETTINGS, name=None, artifact=ARTIFACT)

    frame = hb.evaluations(study)

    assert frame.columns == hb.evaluations(STUDY).columns
    for column in ["x_oracle_a", "f_oracle", "oracle_incumbent", "regret"]:
        assert frame[column].null_count() == 2, column


def test_acquisition_is_the_stored_value_at_each_iteration():
    """`acquisition` is the stored acquisition value of the iteration, and null for
    the initial points and the baselines."""
    frame = hb.evaluations(STUDY)

    assert rows_of(frame, "bo", 0)["acquisition"].to_list() == [None, None, 0.0, -1.0]
    assert rows_of(frame, "lhs", 0)["acquisition"].null_count() == 4


def test_variables_are_numbered_when_the_artifact_has_no_names():
    """Without variables in the artifact, the columns are numbered from 0."""
    study = hb.Study(RUNS[:1], SETTINGS, name=None, artifact={})

    columns = hb.evaluations(study).columns

    assert {"u_0", "x_0", "x_1", "y_0", "x_oracle_1", "y_oracle_0"} <= set(columns)


# summary


def test_summary_has_one_row_per_method_with_medians_and_iqrs():
    """Per method, in study order: the number of runs and of failed runs, and the
    median and interquartile range (numpy's default quantiles) over runs of the
    final incumbent, oracle incumbent and regret."""
    summary = hb.summary(STUDY)

    assert summary["method"].to_list() == ["bo", "lhs"]
    assert summary["runs"].to_list() == [2, 2]
    # The second baseline run stopped after its initial points
    assert summary["failed"].to_list() == [0, 1]
    bo = summary.row(0, named=True)
    finals = {
        "incumbent": np.array([0.0, 1.0]),
        "oracle_incumbent": np.array([0.0, 3.0]),
        "regret": np.array([1.0, 4.0]),
    }
    for name, values in finals.items():
        q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        assert bo[f"{name}_median"] == pytest.approx(q50), name
        assert bo[f"{name}_iqr"] == pytest.approx(q75 - q25), name


def test_summary_counts_runs_that_left_nothing_as_failed():
    """A run that failed before evaluating anything is missing from the study; the
    summary counts it as failed against the settings' number of runs."""
    study = hb.Study(RUNS[:1], SETTINGS, name=None, artifact=ARTIFACT)

    summary = hb.summary(study)

    assert summary["runs"].to_list() == [1]
    assert summary["failed"].to_list() == [1]


# hyperparameters


def test_hyperparameters_has_one_row_per_fitted_entry():
    """Each iteration's fitted values, one row per hyperparameter entry (each
    lengthscale of an ARD kernel is an entry), named by its path in the GP; the
    baselines fit none."""
    frame = hb.hyperparameters(STUDY)

    assert frame.columns == [
        "method",
        "run",
        "iteration",
        "output",
        "hyperparameter",
        "entry",
        "value",
    ]
    # 2 BO runs, 2 iterations, an output scale and 2 lengthscales
    assert frame.height == 2 * 2 * 3
    lengthscales = frame.filter(
        (pl.col("run") == 0)
        & (pl.col("iteration") == 2)
        & (pl.col("hyperparameter") == "kernel.kernel.lengthscale")
    )
    assert lengthscales["entry"].to_list() == [0, 1]
    assert lengthscales["value"].to_list() == pytest.approx([0.1, 0.2])
    # Standard BO's one GP models the objective
    assert frame["output"].unique().to_list() == ["f"]


def test_hyperparameters_name_the_unknown_state_each_gp_models():
    """With two unknown states, hybrid BO fits one GP per unknown state in each
    iteration, and each row's `output` names the unknown state its GP models."""
    # Setup: a hybrid BO run of one iteration whose GPs have different lengthscales
    artifact = ARTIFACT | {
        "y": "Var(name='rate', lower=-inf, upper=inf, start=(0.0, 1.0)); "
        "Var(name='yield_', lower=-inf, upper=inf, start=(0.0, 1.0))"
    }
    run = hb.Run(
        history([3.0, 2.0, 1.0], None),
        artifact={},
        method="hybrid",
        surrogates=[(surrogate([0.1, 0.2]), surrogate([0.3, 0.4]))],
        acquisitions=[AcquisitionSolution(0.0)],
        settings=hb.HybridBOSettings(
            gp=GP(mean=Zero(), kernel=Matern52(), noise_std=0.0),
            acquisition=hb.EI(),
            approximation=hb.SAA(n_scenarios=25),
        ),
    )
    study = hb.Study((run,), SETTINGS, name="two states", artifact=artifact)

    frame = hb.hyperparameters(study)

    # Check: the output scale and two lengthscales of each GP, in the order of y
    assert frame["output"].to_list() == ["rate"] * 3 + ["yield_"] * 3
    lengthscales = frame.filter(pl.col("hyperparameter") == "kernel.kernel.lengthscale")
    assert lengthscales["value"].to_list() == pytest.approx([0.1, 0.2, 0.3, 0.4])


# Without Polars


def test_the_library_runs_without_polars():
    """With Polars hidden, `import hybrid_bo` and the methods work, and a view raises
    an `ImportError` that names the `analysis` extra. A fresh Python process, so that
    Polars isn't already imported."""
    code = """
import sys
sys.modules["polars"] = None  # `import polars` now raises ImportError
import numpy as np
import hybrid_bo as hb
from hybrid_bo.history import History

u = hb.Var("u", lower=0.0, upper=1.0)
x = hb.Var("x", lower=-10.0, upper=10.0)
y = hb.Var("y", lower=-10.0, upper=10.0)
system = hb.System(
    u=[u], x=[x], y=[y], g=lambda v: [v.x - v.y], h_inputs=[u], measured=[y]
)
problem = hb.Problem(system, f=lambda v: v.x**2)
measure = hb.oracle_measure(system, lambda v: [v.u], noise_std=0.0)
settings = hb.StudySettings(n_runs=1, n_initial=2, n_iterations=1, seed=0)
study = hb.study({"lhs": hb.latin_hypercube_sampling}, problem, measure, settings)
assert "polars" not in [name for name, module in sys.modules.items() if module]
try:
    hb.evaluations(study)
except ImportError as error:
    assert "hybrid-bo[analysis]" in str(error), error
else:
    raise AssertionError("no ImportError")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
