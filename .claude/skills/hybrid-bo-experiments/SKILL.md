---
name: hybrid-bo-experiments
description: Use when writing, modifying or running an experiment under experiments/. Covers the layout of an experiment, how to define the Problem and the measurement, the method settings, saving runs, and common mistakes.
---

# Writing and running experiments

Read `docs/writing_an_experiment.rst` first: it is the user guide (layout, CasADi
rules, `Var`/`System`/`Problem`, oracle vs custom measurement, methods and settings,
`cli.main`, running, common errors). `docs/reading_results.rst` covers the saved study,
the figures and the Polars views. `experiments/illustrative_example.py` is the
reference experiment, and `illustrative_example_2d.py` its version with two decision
variables; `rosenbrock.py` (P = 2) and `langermann.py` (P = 5) have no
known equations (M = 0), and `langermann_partial.py` is `langermann.py` with explicit
known equations of u alone (M = 2, P = 3); `langermann_sweep.py` is its sweep over
initial fractions for one method (one configuration per `n_initial`, and the sweep's
`figures`). `cstr_series_reaction.py` is a 1D chemical engineering example (M = 2,
P = 1) whose unknown state is a log rate constant, so every GP scenario keeps the
known equations solvable. Use GLOSSARY.md's terms. This file adds only what an agent needs on top.

## Layout contract

An experiment is one module under `experiments/`, importable as a top-level module
(pytest and ty put `experiments/` on the path). At module level, in this order:
Problem, measurement (with a module-level `def oracle(v)` and the stated `optimum` for
a benchmark), the BO methods' prior GP (`gp`), `methods` (or one `method`, for a
sweep), `configurations` (each method as an `hb.Configuration(method=...,
n_evaluations=..., n_initial=...)`, the comparisons giving every method the same
budget, and `langermann_sweep.py` one per `n_initial`), `study_settings`
(`hb.StudySettings(n_runs=..., seed=...)`), then a `__main__` block calling
`cli.main` with `configurations`.

- Everything the command line uses is passed in the `cli.main` call (directly, or
  inside `problem` and `measure`); nothing is read from the module by name.
- Importing the module runs nothing, so `tests/test_regression.py` reuses the
  illustrative example's `problem`, `measure`, `methods` and `configurations`, and
  `tests/conftest.py`'s `illustrative_problem` fixture is its `problem`. Changing the
  illustrative experiment's problem, measurement or settings moves the regression
  baselines: that is a *logic* change.
- A custom measurement is `(u (N,), rng) -> measured (P,)`; its docstring (else its
  name) describes it in each run's artifact; a dataclass measurement is described by
  its repr.
- A BO method's settings are required, since they name its GP: bind them with
  `functools.partial(hb.hybrid_bo, settings=hb.HybridBOSettings(gp=gp, ...))`, never
  put `hb.hybrid_bo` or `hb.standard_bo` in `methods` bare. `settings` is
  keyword-only, after `rng`.
- Defaults never choose the method (ADR 0009): a default may set how hard or how
  precisely we search, never what we search for. So these are required and spelled
  out in every experiment: `acquisition` (`hb.EI()` or `hb.LCB(std_factor=...)`) on
  both settings, `approximation` on `HybridBOSettings`, `n_scenarios` on `hb.SAA` and
  `hb.SoftSAA` (with `temperature`), and `noise_std` on `hb.oracle_measure` and
  `hb.gp.GP`. Search effort and numerics keep defaults (`n_starts`, `n_starts_gp`,
  the solver, `jitter`, kernel priors, `ard`, `unknown_states_as_variables=True`). The
  settings and choice dataclasses, `GP`, `Configuration` and `StudySettings` are
  keyword-only:
  `hb.SoftSAA(temperature=5e-3, n_scenarios=25, unknown_states_as_variables=False)`
  (the experiments take a temperature of about 1e-3 times the optimum's magnitude).
  Leaving one out, or passing one by position, raises `TypeError`.
- Define the GP once as a module-level `gp`, each fitted
  hyperparameter its own variable so the experiment shows them all (see
  `illustrative_example.py`): `outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))`,
  `lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))`, then
  `hb.gp.GP(mean=hb.gp.Zero(), kernel=outputscale *
  hb.gp.Matern52(lengthscale=lengthscale), noise_std=0.0)`. The shipped experiments
  share it between methods; each method fits its own GP to its own data.

## Randomness

Every draw of a study comes from `settings.seed`. Run `i` of every configuration gets
the `i`th child of the seed, so run `i` of each configuration with the same `n_initial`
starts from the same initial points. Within `hb.repeat`, run `i` draws
`hb.initial_points(system.u, n_initial, rng)` and runs the method for `n_evaluations -
n_initial` iterations, both from that generator. A BO run with a larger budget passes
through the same first evaluations as one with a smaller budget (the prefix property),
so a budget sweep needs one configuration per `n_initial`, at the largest budget; the
baselines draw their whole design first: Latin hypercube sampling lacks it, and
uniform sampling has it only with a measurement that draws nothing. Outside a study, do the same by hand:
`np.random.default_rng(seed).spawn(n_runs)`, and draw nothing else from a run's
generator.

## Oracle optimum

State it as a module-level `optimum`; the study never computes it (a 10-start search
found a local minimum for the illustrative example, #62). Find it once with
`hb.oracle_optimum(problem, oracle, rng, n_starts=200)`, check it, and say how in a
comment. `hb.study` warns with `OracleOptimumWarning` if a run evaluates a point whose
oracle objective is below it.

## Pitfalls

- `math.exp(v.x)` on a CasADi symbol silently returns `nan`; numpy functions mostly
  work but emit a CasADi `FutureWarning`. Always use `cas.*`.
- `System` takes keyword arguments only. Without known equations, leave out both
  `x` and `g` (M = 0); giving one without the other raises `ValueError`.
- Exactly one measured variable per unknown state. With `P > 1` unknown states,
  `plots.iteration` draws hybrid BO only with `gp=False` (else `ValueError`), so the
  CLI draws those runs' iteration figures without the GP panels.
- With measurement noise, state it in the GP too, as the same number:
  `GP(..., noise_std=0.01)` for `oracle_measure(..., noise_std=0.01)`. `noise_std`
  is in measured units; kernel hyperparameters (`Param` bounds, fixed lengthscales) are in
  scaled units (inputs onto [0, 1], outputs standardized). Don't convert the noise by
  hand.
- `true_curves=True` runs the measurement over the plot grid: only for an oracle.
- The iteration figures need one or two decision variables; otherwise `plot` prints
  why and skips that run's figures. They are drawn for run 0 of each method unless
  `--plot-runs` names other runs (or `all`). They replay each run from what it stored (its fitted
  GPs and its settings), so nothing from the experiment's methods is passed and a
  study plots after the methods change.
- Studies saved before #119 (GPs as data) or #120 (runs hold their settings) don't
  load: `hb.load` raises `StudyFormatError`. Rerun them.

## Running

    pixi run experiment <name> [--method M ...] [--n-runs 10] [--seed 0] [--n-workers 4] [--name ID] [--plot] [--plot-runs 0 3] [--require-clean]
    pixi run experiment <name> prepare [--method M ...] [--n-runs 10] [--seed 0] [--name ID]
    pixi run experiment <name> run-task STUDY --task-index K
    pixi run experiment <name> resume STUDY [--n-workers 4] [--plot]
    pixi run plot <name> [STUDY]
    pixi run experiment <name> plot [STUDY] [--plot-runs all] [--results-root DIR]

`--method` takes configuration labels. There is no `--n-iterations`: budgets belong
to configurations, so change them in the experiment (tests write small experiment
modules, as `tests/test_cli.py`'s `small` does, and run them with its `experiment`
fixture). A study's runs are tasks (one run of one configuration, `hb.Task`). `run`
is `prepare`, then the tasks, then assembly:

- `prepare` reserves the study id by creating the study folder (an atomic `mkdir`,
  so studies started at once never share an id) and writes its task list
  `tasks.json`: the seed (resolved if None), `n_runs`, each chosen configuration's
  budget, the oracle optimum, the code version, the problem's artifact text and the
  tasks as `[label, index]`, by label, then index.
- The launcher (`cli.run_tasks` -> `cli.run_commands`, a generic runner of named
  commands) runs each task as `python -m <name> run-task STUDY --task-index K` from
  `experiments/`, keeping `--n-workers` running (default 1) in task order. The
  `name` passed to `cli.main` must be the experiment's module name (a usage error
  otherwise); tests write their experiment modules to a folder on `PYTHONPATH`. The
  task rebuilds the experiment from its module, so nothing is pickled: lambdas and
  nested functions are fine. A task saves `runs/<label>/run-003.json`
  (temporary file, then renamed), or `run-003.failed` for a run that failed before
  evaluating anything, and `logs/<label>/run-003.log` (DEBUG and above; the console
  gets INFO and above, tagged `<label>/run-003`). It exits 0 when the task is done,
  also if it was already, 1 on an unexpected exception (traceback in its log), 2 if
  the experiment or the code version differs from the task list. With more than one
  at a time, tasks get `OMP/OPENBLAS/MKL/VECLIB_*_NUM_THREADS=1` unless set; `-W`
  options are passed on.
- Assembly loads the run files (`hb.load_run`), calls `hb.assemble`, writes
  `artifact.json` (`hb.saving.save_artifact`; the study is finished exactly when it
  exists, so it is written last, atomically), rebuilds `study.log` as the launcher's
  lines then every per-run log, each line starting with its tag (dropping lines an
  interrupted assembly tagged, so none appear twice), and writes `summary.csv`.

Ctrl-C, a failing task, or any other error once the tasks have started stops the
study at once (running tasks are terminated, finished runs kept), prints the `resume`
command and exits 130 or 1. `resume STUDY [--n-workers N] [--require-clean]` takes
everything else from the task list; it refuses a finished study and, naming the
difference (both values for one-value fields), an experiment whose configurations,
budgets, problem or measurement text, or code version differ from the task list;
then it runs the tasks left and assembles. Helpers raise `cli.UsageError`, which
`main` reports as a usage error. For a cluster, `prepare`, then a SLURM array of
`run-task --task-index $SLURM_ARRAY_TASK_ID`, then `resume`.

No command lists every experiment, and a typo in the name fails with "No module named
...". A relative `--results-root` is relative to where pixi was started (`INIT_CWD`),
not `experiments/`. The study id is `--name`, else the UTC time to the second at the
start. A taken `--name` (by a saved or an unfinished study) is a usage error; a taken
UTC time gets `-2`, `-3`, ... appended. The study log holds the launcher's lines
(each configuration, each task started and finished with a count, warnings) and then
every run's lines, once the study is assembled. `plot` loads the study given (an unfinished one is a usage error
saying to resume it), else the finished one whose artifact was written last
(`artifact.json`, or `study.json` in older studies), and saves `summary.csv` if the
study has none (studies saved before summaries). Plotting draws the study figures
`cli.main`'s `figures` names (default convergence and regret; a sweep over initial
fractions passes `hb.plots.budget_heatmap` and its siblings, file name to function of
the study; one that raises `ValueError` is skipped with a message), and the iteration
figures of the runs `--plot-runs` names (indices or `all`, default `0`, of each
method); on `run` and `resume`, `--plot-runs` implies `--plot`. Running with
uncommitted or untracked files warns that the study won't be reproducible;
`--require-clean` refuses to run instead.
