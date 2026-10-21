# Changelog

## Unreleased

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
