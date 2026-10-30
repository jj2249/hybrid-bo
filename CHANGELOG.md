# Changelog

## Unreleased

- **Breaking:** The experiments `langermann_partial` and `langermann_sweep` are
  removed, with the sweep's figures `hb.plots.budget_heatmap`,
  `hb.plots.regret_at_budget` and `hb.plots.regret_by_initial_fraction`, and
  `cli.main`'s `figures`: `plot` always draws convergence and regret.
- Study tasks start about a second faster (#172): `hb.sampling.lhs` draws the
  same Latin hypercube as scipy's `qmc.LatinHypercube`, without importing
  `scipy.stats`.

## 0.5.1 (2026-10-26)

- Assembling a study no longer fails on Windows (#163): the study log is closed
  before it is rebuilt, so its lines logged after the rebuild, such as the runs
  saved, also reach the new study log on Linux and macOS.

## 0.5.0 (2026-10-26)

- `hb.plots.budget_heatmap` (#147) draws the budget on y and the initial fraction
  on x, so a row is the choice at one budget, and drops the lines of constant
  initial fraction and of the best one at each budget. Its new `tolerance` colors
  each cell by the fraction of runs whose regret is at most `tolerance` instead of
  the median log10 regret.
- The experiment `langermann_sweep` (#147): a sweep of SoftSAA-EI over initial
  fractions on `langermann_partial`'s problem, with n_initial = 2, 4, ..., 38 and 40
  evaluations each, which plots `budget_heatmap` of the median regret and of the
  runs within 0.027 of the optimum, `regret_at_budget` at budgets of 10, 20, 30 and
  40, and `regret_by_initial_fraction`.
- The experiment `langermann_partial` (#153): the Langermann function, box and
  optimum of `langermann`, with known and unknown equations. The squared distances
  from centers 1 and 4 are mechanistic states with known equations, and those from
  centers 2, 3 and 5 are unknown states, so standard BO and Latin hypercube sampling
  behave exactly as on `langermann`. SoftSAA-EI's temperature is 5e-3.
- **Changes results:** The experiments' SoftSAA-EI temperatures are about 1e-3 times
  each optimum's magnitude, set from trial runs, instead of 1e-6, which took tiny
  local steps: 50e-3 for `illustrative_example` and `illustrative_example_2d`, 5e-3
  for `langermann`, and 100e-3 for `rosenbrock` (whose optimum is 0). The library is
  unchanged; the illustrative example's `hybrid_bo_soft` regression baseline changes.
- **Breaking:** Parallel studies are tasks over files, which checkpoint and resume
  (#143, #145, #156, ADR 0010). A study's runs are its tasks: `hb.study_tasks`,
  `hb.run_task` on each and `hb.assemble`, of which `hb.study` is the composition, so
  a launcher can run them elsewhere; `hb.study` runs them serially in the caller's
  process and has no `n_workers`. `hb.save_run` and `hb.load_run` save and load one
  run file, and `hb.saving.save_artifact` completes a study whose run files are
  saved. The experiment CLI runs each task as its own process
  (`python -m <experiment> run-task`), `--n-workers` at a time (default 1), on
  Windows and macOS alike, with single-threaded BLAS when there are several; each
  run is saved as it finishes, with its per-run log, and lambdas and nested
  functions work. Ctrl-C, or a task that fails, stops the study at once, keeps the
  finished runs and prints the command that finishes it:
  `pixi run experiment <name> resume STUDY [--n-workers N]`, which refuses a
  finished study or a changed experiment, seed or code version. `prepare` and
  `run-task STUDY --task-index K` split a study up for a SLURM array or GNU
  parallel. A study folder holds its task list (`tasks.json`) from the start, and
  is finished when `artifact.json` exists; `plot` skips unfinished studies. The
  study log starts each line of a run with its tag (`hybrid_bo/run-003`) and holds
  the runs' lines after the launcher's, in task order. The `<id>.log` and
  `<id>.logs` beside the study folders are gone. Results are unchanged.
- Faster iteration figures for hybrid BO with mechanistic states (#154): the replay
  of the acquisition function root-finds each scenario's states with Newton's method
  (CasADi's `rootfinder`), mapped over the grid in rings outwards from the trial
  point, instead of one IPOPT solve per grid point, and evaluates the acquisition
  problem's objective there. A grid point whose root falls outside a state's bounds
  is still a gap. A new `hb.ReplayWarning` says when the replay at the trial point
  isn't the run's stored solution. Results are unchanged; figures differ from before
  only by IPOPT's tolerance.
- Figures of an initial-fraction sweep (#146): `hb.plots.budget_heatmap` (median
  log10 regret by budget and number of initial points, with lines of constant
  initial fraction and the best one at each budget), `hb.plots.regret_at_budget`
  (median regret and IQR against the initial fraction at one budget) and
  `hb.plots.regret_by_initial_fraction` (median regret over evaluations, one line
  per number of initial points). Each takes a study of one method and raises
  `ValueError` otherwise. `cli.main` takes `figures`, the study figures `plot`
  draws (default: convergence and regret, as before).
- **Breaking:** Configurations (#144): `hb.Configuration(method=..., n_evaluations=...,
  n_initial=...)` is a method with its own evaluation budget, initial points
  included, so one study can compare splits of a budget. `StudySettings` keeps only
  `n_runs` and `seed`; `hb.study` and `cli.main` take a mapping of labels to
  configurations instead of methods, and `hb.repeat` a configuration.
  `StudySettings(n_runs=10, n_initial=2, n_iterations=5)` with `methods` becomes
  `StudySettings(n_runs=10)` with `{label: hb.Configuration(method=method,
  n_evaluations=7, n_initial=2) for label, method in methods.items()}`. Each run of a
  study records its configuration's budget (`Run.n_evaluations`, saved; studies saved
  before load with their one budget in every run), `summary` counts failed runs
  against each run's budget, and `convergence` and `regret` mark the end of each
  number of initial points. The experiment CLI drops `--n-iterations`, and the study log
  names each configuration with its budget. Results are unchanged.
- **Breaking:** Defaults never choose the method (#140, ADR 0009): `acquisition` of
  `HybridBOSettings` and `StandardBOSettings`, `approximation` of `HybridBOSettings`
  and of `hybrid_acquisition`, `n_scenarios` of `SAA` and `SoftSAA`, `std_factor` of
  `LCB`, and `noise_std` of `oracle_measure` and `GP` have no default. `EI`, `LCB`,
  `SAA`, `SoftSAA`, `Linearization`, both settings, `GP` and `StudySettings` are
  keyword-only. The old defaults were `EI()`, `SAA(n_scenarios=25)`,
  `std_factor=1.0` and `noise_std=0.0`: `HybridBOSettings(gp=gp)` becomes
  `HybridBOSettings(gp=gp, acquisition=hb.EI(), approximation=hb.SAA(n_scenarios=25))`,
  and `SoftSAA(1e-6)` becomes `SoftSAA(temperature=1e-6, n_scenarios=25)`. Results
  and saved studies are unchanged.
- `oracle_measure` returns an `OracleSimulation`, a frozen, callable dataclass that
  holds the system, the traced oracle and the settings (#141). A run's artifact records a dataclass measurement by its repr, so the oracle
  simulation's `measure` entry reads `OracleSimulation(y = ..., noise_std=...,
  n_starts=..., tol=...)` instead of `oracle simulation: y = ..., ...`. Results don't
  change.
- **Changes results:** `multistart` breaks ties toward the earlier start (#136): a
  later start replaces the best only when its objective is lower by more than 1e-13
  (relative, or absolute below 1), so objectives that differ only by rounding give
  the same solution on every platform. Before, macOS and Windows could pick different
  optima of a symmetric or flat acquisition problem. Hybrid BO's illustrative example
  results change; standard BO's match the macOS results.
- The experiment `langermann` (`pixi run experiment langermann`): the Langermann
  function of Astudillo & Frazier's App. D.1 as a composite function, with no known
  equations and five unknown states, comparing SAA-EI, SoftSAA-EI, standard BO and
  Latin hypercube sampling (#132). The experiment `rosenbrock` is now D.2 adapted to
  2D, with no known equations and two unknown states, y1 = u2 - u1^2 and y2 = u1.
- The experiment `illustrative_example_2d`: the illustrative example with two decision
  variables and the unknown equation y = sin(u1) cos(u2), whose iteration figures
  show every 2D panel, the GP of y included, now that `rosenbrock` has two unknown
  states (#132).
- The experiment CLI draws the iteration figures of a hybrid BO run with more than
  one unknown state without the GP panels, instead of skipping them (#132).
- The 2D iteration figure's EI panel computes SoftSAA's log10 EI from its -log EI,
  so where SoftSAA-EI underflows in float64 the note no longer reads `-inf` and the
  map has no gray "EI = 0" cells. The EI color scale covers at most 10 decades below
  the largest EI (#138).

## 0.4.0 (2026-10-21)

- **Breaking:** A system may have no mechanistic states or known equations (#115):
  `System`'s arguments are keyword-only, and `x` and `g` are optional, both left out
  for M = 0, which makes hybrid BO BO of a composite function. Every method, saving
  and the plots work for such a system. An empty array saved where `None` is allowed
  too now loads as an array. Results with known equations don't change.
- **Breaking:** A system may have more than one unknown state (#114). Hybrid BO fits one
  independent GP per unknown state, all from the settings' prior GP over the same
  arguments of h. `Run.surrogates` holds a tuple of fitted GPs per iteration (one per
  unknown state; a 1-tuple for standard BO), and `load` reads older runs' single GPs
  as 1-tuples. `hb.hyperparameters` has an `output` column naming the variable each GP
  models (the unknown state, or `f`). `plots.iteration` with more than one unknown
  state draws only with `gp=False`. Results with one unknown state don't change.
- A study's artifact gives each BO method's prior GP (#128): the run artifact's `gp`,
  on the line after the method's settings in `methods`, so the study artifact alone
  says how to run the study again. Studies saved before have it only in their runs.
- `hb.plots.iteration` draws problems with two decision variables (#117): filled
  contours over a 30 x 30 grid, a row of y and a row of f for hybrid BO (oracle, mean,
  std, log10 EI with SAA-EI's zeros in gray), a row of f for standard BO. Three or more
  still raise `ValueError`. `predict` maps a numpy input over its rows, so predicting at
  many points is fast; its values don't change.
- The experiment CLI's `--plot-runs` (run and `plot`) chooses the runs that get
  iteration figures: indices or `all`, default `0`. Plotting a study now draws run 0's
  iteration figures, not every run's. On the run command it implies `--plot`.

## 0.3.0 (2026-10-21)

- **Breaking:** A run has an artifact (#123): `Run.record` is `Run.artifact`, with the
  same content, and a saved run's JSON names it `artifact`. `run_record` is
  `run_artifact`, and `optimization_loop` takes `artifact` for `record`. Studies whose
  run files hold `record` don't load: `hb.load` raises `StudyFormatError`; rerun them.
- **Breaking:** Runs hold their method's settings (#120): `Run.settings` is the
  `StandardBOSettings` or `HybridBOSettings` a BO run ran with (None for the
  baselines), saved in full. `hb.plots.iteration` replays from them and loses its
  `settings=` argument, so a run with a custom solver plots with nothing passed.
  The run's artifact keeps the settings as text, for people. `optimization_loop` takes the
  settings as a required keyword, `settings=` (None for a method the plots can't
  replay). The settings classes moved to `hybrid_bo.settings`
  (still `hb.HybridBOSettings` and `hb.StandardBOSettings`). Studies saved before
  this change don't load: `hb.load` raises `StudyFormatError`; rerun them.
- **Breaking:** GPs are data (#119, ADR 0008). A prior GP is
  `hb.gp.GP(mean, kernel, noise_std=0.0, jitter=1e-6)` built from a closed set of
  frozen dataclasses: the `Zero` mean and the kernels `Matern12`, `Matern32`,
  `Matern52`, `SquaredExponential` and `Scale` (an output scale, unsquared), which
  `outputscale * kernel` builds. A kernel's hyperparameters are its fields: a number
  fixes one, `hb.gp.Param(init, bounds, log=True)` fits it; a lengthscale has one
  entry per input unless `ard=False`. `noise_std` is a
  known noise in measured units, so it is the same number as the measurement's
  noise. `HybridBOSettings.gp` and `StandardBOSettings.gp` are required, so
  `hb.standard_bo` can't go into `methods` bare:
  `GP(Zero(), Param(1.0, (0.1, 10.0)) * Matern52())` is the old default, with the
  same results bit for bit. A fitted GP is
  `FittedGP(gp, X, Y, x_scaler, y_scaler)`, the prior GP with every `Param` fitted;
  `predict(fitted, X)` and `log_marginal_likelihood(fitted)` take it alone, as do
  `standard_acquisition` and `hybrid_acquisition`. Runs save their fitted GPs whole,
  so `hb.plots.iteration` draws a run with any GP without `settings=`. A run's artifact
  holds the prior GP's repr as `gp` (no `kernel` and `mean` entries), and
  `hb.hyperparameters` names each by its path in the GP (`kernel.outputscale`,
  `kernel.kernel.lengthscale`). `PriorGP`, `Hyper`, `default_gp` and `hb.kernels`
  (with the unused `linear` and `constant` kernels) are removed, with no aliases.
  Studies saved before this change don't load: `hb.load` raises `StudyFormatError`;
  rerun them.
- The experiment `rosenbrock` (`pixi run experiment rosenbrock`): the 2D
  Rosenbrock function as a hybrid model, the first experiment with two decision
  variables, comparing SAA-EI, SoftSAA-EI, standard BO and Latin hypercube
  sampling (#116).
- `multistart` warns when starts stop at IPOPT's iteration limit, giving how many
  of the starts did and the name of the NLP's objective (e.g.
  `negative_log_marginal_likelihood` for a GP fit), and `fit_gp` warns when the
  fitted GP's training covariance has a condition number above 1e10, giving it and
  the jitter. Both appear in the study log; results don't change (#108).
- A study logs its progress: `hb.study` each method, `hb.repeat` each run and the
  optimization loop each iteration (with its time and the incumbent) at INFO, and
  `fit_gp` and `multistart` their times at DEBUG. The library configures no handlers.
  The experiment CLI shows INFO on the console and saves DEBUG and above, with every
  warning, as the study log, `study.log` in the study's folder; a study that doesn't
  finish leaves `<study id>.log` beside the study folders (#109).
- **Breaking:** the experiment CLI reserves the study id before running, and refuses
  a `--name` that is already taken instead of saving to `<name>-2` (#109).
- **Changes results:** `PriorGP.jitter` defaults to 1e-6 instead of 1e-10. With
  1e-10 the GP fit stalled once a run's inputs clustered (cond(K) about 1e11), so a
  15-iteration hybrid BO run on the illustrative example took 160s; it now takes
  4s. Fitted GPs, and so the BO methods' trial points, change (#107).
- SoftSAA-EI: `hb.SoftSAA(temperature)` as hybrid BO's approximation
  (`HybridBOSettings(approximation=hb.SoftSAA(1.0))`) is SAA-EI with each scenario's
  improvement softened by a smooth maximum at the temperature, in units of the
  objective, so the acquisition problem has no flat regions or kinks; it minimizes
  -log SoftSAA-EI, which runs store as their acquisition value (the iteration plot
  draws it as -EI). It takes
  `n_scenarios` and `unknown_states_as_variables` like `SAA` and gives EI only; with
  `LCB` it raises `ValueError`. The illustrative example runs it as
  `hybrid_bo_soft` (#103).
- Lin-EI: `hb.Linearization()` as hybrid BO's approximation
  (`HybridBOSettings(approximation=hb.Linearization())`) computes EI in closed form
  from the objective linearized around the nominal states, over one scenario instead
  of `n_scenarios`. It gives EI only; with `LCB` it raises `NotImplementedError`.
  The illustrative example runs it as `hybrid_bo_lin` (#99).
- **Breaking:** hybrid BO's acquisition approximation is a setting of its own,
  `hb.SAA(n_scenarios, unknown_states_as_variables)`, and `n_scenarios` and
  `unknown_states_as_variables` move from `HybridBOSettings` to it:
  `HybridBOSettings(n_scenarios=5)` becomes
  `HybridBOSettings(approximation=hb.SAA(n_scenarios=5))`.
  `hb.acquisition.hybrid_acquisition` takes `approximation` in place of
  `unknown_states_as_variables`. Results are unchanged (#98).
- **Breaking:** `hb.acquisition.standard_acquisition` and `hybrid_acquisition` return
  an `AcquisitionProblem` (`.nlp`, `.start_bounds` and the order of the decision
  vector) instead of the tuple `(nlp, (start lower, start upper))` (#93).
- **Breaking:** the prior GP and the fitted GP are explicit. `hb.gp.GP` is renamed
  `hb.gp.PriorGP` (`FittedGP` keeps its name), and `hb.gp.log_likelihood` is renamed
  `hb.gp.log_marginal_likelihood`. There are no aliases for the old names (#91).
- A study's artifact (`Study.artifact`, saved as `artifact.json`) replaces
  `Study.record` and `study.json`. `hb.load` still reads studies saved as
  `study.json`.
- The recorded code version ends in `-untracked` when files that aren't ignored were
  never added to git, as it ends in `-dirty` for uncommitted changes, so a study's
  artifact says whether it is reproducible.
- `pixi run experiment` warns when run with uncommitted or untracked files;
  `--require-clean` refuses to run instead.
- Versioning and releases are documented in CONTRIBUTING.md: the version changes
  only in a release PR, and merging that PR tags and publishes the release (#84).

## 0.2.0 (2026-10-16)

The first release after rewriting the original code (tag `pre-refactor`). The
reasons for the rewrite are in [DESIGN.md](DESIGN.md).

- A problem is `hb.Var`s, an `hb.System` of known and unknown equations and an
  `hb.Problem`; a measurement (an `hb.Measure`, or `hb.oracle_measure` to simulate
  one) returns the measured variables.
- Four methods with default settings: `hb.hybrid_bo`, `hb.standard_bo`,
  `hb.latin_hypercube_sampling` and `hb.uniform_sampling`. Each returns an `hb.Run`,
  which is plain data.
- `hb.study` runs every method several times from shared initial points; studies are
  saved and loaded as JSON (`hb.save`, `hb.load`).
- Polars views (`hb.evaluations`, `hb.summary`, `hb.hyperparameters`) and static plots
  (`hb.plots`) of a study.
- Experiments are Python modules in `experiments/`, run with
  `pixi run experiment <name>` and plotted with `pixi run plot <name>`.
- User docs: Getting started, Writing an experiment and Reading results.
- Results pickled by the old code can't be loaded; regenerate them with the new API.
