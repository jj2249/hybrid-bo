---
name: hybrid-bo-src
description: Use when working on src/hybrid_bo. Covers what the library computes (hybrid BO, standard BO and the baselines), the module map, data and shape conventions, behavior that surprises people, and the testing rules.
---

# Working on src/hybrid_bo

The workflow is in CONTRIBUTING.md, and open bugs are GitHub issues labeled `bug`.
This skill covers what the code computes and the contracts between its parts;
DESIGN.md holds the design and the reasons. The
code uses CONTEXT.md's terms; use them in PRs, tests and comments too.

## What the code computes ([arXiv 2603.11199](https://arxiv.org/abs/2603.11199))

    min_u f(x, y, u)   s.t.   0 = g(x, y, u),   y = h(x, u)  (unknown)

- **Evaluation** (`evaluate`): the measurement returns the measured variables for each
  trial point. The oracle simulation (`oracle_measure`) solves `g` with
  `y = oracle(x, u)` for every state by multistart `fsolve`, reads the measured
  variables and adds `rng.normal(0, noise_std)` (drawn also when `noise_std` is 0).
  The library then always solves the **root-finding problem**: `g` for the unmeasured
  states given `u` and the measured variables.
- **Hybrid BO** (`bo.py`, `acquisition.py`): the unknown equations are never used in
  the acquisition problem. A GP maps the `h_inputs` columns of `(u, x)` to `y` and is
  trained on the evaluated points. The acquisition problem is a sample-average
  approximation:
  - `n_scenarios` scenarios `mean + std * z_s`. The scenario draws `z_s` are drawn once
    per run and reused in every iteration.
  - Each scenario has its own copy `x_s` of the mechanistic states, and
    `g(x_s, y_s, u) = 0` holds for every scenario.
  - With `unknown_states_as_variables` (default True) the `y_s` are variables with the
    constraint `y_s = scenario s`. Otherwise the GP prediction is substituted directly
    (the illustrative example's setting).
  - Acquisition function: SAA-EI is `sum_s w_s min(f_s - incumbent, 0)`, minimized.
    SAA-LCB is the weighted mean of `f_s` minus `std_factor` times the weighted std.
  - Solved with `multistart`. The trial point is the `u` part of the solution.
- **Standard BO** (`bo.py`): a GP directly on `u -> f`, with analytic EI or LCB over
  `u` only.
- **Latin hypercube / uniform sampling** (`baselines.py`): the non-adaptive baselines.
  They draw the whole design at the start of a run and train no GP.

## Module map

Every module lives at the top level of `hybrid_bo`. `hybrid_bo/__init__.py`
re-exports only what a typical script uses (DESIGN.md, "A slim top level");
`tests/test_namespace.py` pins that set. Everything else, tests included, is imported
from its module (`from hybrid_bo.gp import fit_gp`).

- `problem.py`: `Var` (name, bounds, start bounds), `System` (`u`, `x`, `y`, `g`,
  `h_inputs`, `measured`) and `Problem` (`system`, `f`). `g` and `f` are user
  callables over a namespace `v` of symbols (`v.x`), traced once in `_trace` into
  `cas.Function`s with column-vector inputs in the paper's order `(x, y, u)`. All
  input checks live in the constructors and raise `ValueError` (`NotImplementedError`
  for `P > 1`). `Symbolic` (default `cas.SX`) is the one place the symbol type is
  chosen. `_h_inputs(system, u, x)` gives the `h_inputs` columns of numeric or
  symbolic `u`, `x` (hybrid BO's GP training data and acquisition problem).
- `sampling.py`: `lhs(lower, upper, n, rng)` and `uniform(...)` over a box, shape
  `(n, d)`; `initial_points(variables, n, rng)` is `lhs` over `Var` start bounds. Use
  them for any Latin hypercube or uniform draw.
- `evaluation.py`: a `Measure` is `(u (N,), rng) -> measured (P,)`, or an
  `OracleMeasurement` (measured plus oracle `x`, `y`) in benchmark experiments.
  `oracle_measure(system, oracle, noise_std, ...)` traces the oracle like `g`
  (`_trace_oracle`).
  `oracle_optimum(problem, oracle, rng)` solves min f s.t. g = 0, y = oracle by
  `multistart` from `lhs` starts over the start bounds of (u, x, y).
  `evaluate(problem, measure, u, rng)` runs the measurement per row, then always
  solves the root-finding problem (`_reconstruct`, starts over the unmeasured
  variables), and returns a `History` whose points are all initial points. Every
  fsolve goes through `_find_root` (CasADi Jacobian, accepts residual `< tol` within
  the bounds, silences fsolve's warnings). With an oracle, one extra solve from the
  oracle values (no draws) checks the reconstruction is on the same root, else
  `ReconstructionWarning`. `_start_bounds` (in `sampling.py`) and `_as_text` (in
  `problem.py`, an equation as text in the variable names) are shared helpers.
- `history.py`: `History` (`u`, `x`, `y`, `f`, oracle values or `None`, `n_initial`)
  `incumbents` and `oracle_incumbents` (the benchmark metric); `_concatenate`
  appends one history to another.
- `run.py`: `Run` (`history`, `record`, `method`: its label, `record["method"]`
  unless a study relabels it; `index` in its study; `surrogates`: one `FittedGP` per
  iteration, the one behind its trial point, and `acquisitions`: one
  `AcquisitionSolution` per iteration, both empty for the baselines; `scenarios`,
  hybrid BO only, else None), pure data that loads without user code (ADR 0003).
  `run_record(problem, measure, rng, method, settings)` builds the text record
  from `_problem_record`, `_generator_record` and `_versions_record` (a study's
  record reuses them); call it before the method draws. A measurement is recorded by
  its docstring, else its name (`oracle_measure` sets the docstring).
- `studies.py`: `StudySettings(n_runs, n_initial, n_iterations, seed)`, `Study(runs,
  settings, name, record)`, `Method` (called as `method(problem, measure,
  initial_points, n, rng=rng)`), `repeat` (run i from
  `default_rng(SeedSequence(seed).spawn(n_runs)[i])`, initial points then the method;
  catches `HybridBOError`, logs it and keeps `error.run`) and `study` (resolves a
  None seed to fresh entropy kept in its settings, then each method's `repeat` from
  that seed; relabels runs with their key; records a stated `optimum` as
  `oracle_optimum`, never computes one, and warns with `OracleOptimumWarning` if an
  evaluated point's `f_oracle` is below it). Studies take a seed, not an
  `rng`: the settings alone reproduce a study.
- `saving.py`: `save(study, path)` / `load(path)` (ADR 0004): `study.json` lists the
  run files `runs/<method>/run-<index:03d>.json`. `_encode`/`_decode` convert any
  dataclass by its type hints; arrays are nested lists of floats (an empty one is
  `{"shape": ...}`). A new field on `Run` or its parts round-trips without changes if
  its type is one `_decode` handles. `save` refuses an existing folder.
- `views.py`: `evaluations(study, optimum=None)`, `summary(study)` and
  `hyperparameters(study)`, Polars data frames (one row per evaluation, per method,
  per fitted hyperparameter entry); each docstring lists its columns. Variable
  columns are named from the study's record (`_names` reads `Var(name='...')`).
  Polars is imported inside the functions (`_polars`), so `import hybrid_bo` works
  without it; a missing Polars raises `ImportError` naming `hybrid-bo[analysis]`.
- `baselines.py`: `latin_hypercube_sampling` and `uniform_sampling` draw the whole
  design over the `u` start bounds, then `evaluate` initial points plus design in
  one call.
- `solvers.py`: `NLP` (`f`, bounds `(n,)`, optional `g(w) = 0`), `Ipopt` settings and
  `multistart(nlp, starts, solver) -> Solution` (`w`, `f`), the best start. It uses
  `cas.Opti` with bounds as constraints, as the pre-refactor code did, so IPOPT takes
  the steps the baselines record; failed starts are logged at debug level,
  `NoSolutionError` if all fail. The caller draws the starts with `lhs`.
- `kernels.py`: kernels as plain functions of CasADi matrices (`matern12/32/52`,
  `squared_exponential`, `linear`, `constant`); a lengthscale or scale is a scalar or
  a `(d, 1)` column (ARD).
- `gp.py`: `Hyper` (named, `size`, `log`), `GP` spec (hyperparameters in optimization
  order, `kernel(h, X1, X2)`, `mean(h, X)`, `jitter`), `default_gp(n)` (noise-free
  `outputscale`, unsquared, times Matern-5/2 with ARD), `Scaler` with
  `minmax`/`standard`/`scale`/`unscale`, `FittedGP` (data only), `fit_gp`, `predict`
  (numpy X -> arrays `(n,)`, symbolic X -> `(n, 1)` columns) and `log_likelihood`. The
  hyperparameter named `noise_std`, if declared, is the noise:
  `K = k(X, X) + (noise_std**2 + jitter) I`; without it only `jitter` is added.
  Numeric data enter CasADi as `Symbolic` constants. Keep the `minmax` operation
  order: it reproduces the pre-refactor values bit for bit for [0, 1], which the
  baselines depend on. `_gp_record` gives the GP as text for a run's record.
- `acquisition.py`: `EI`, `LCB(std_factor)` and `standard_acquisition` (analytic EI
  or LCB over `u`; returns the `NLP` and the `u` start bounds). `_check_acquisition`
  is the one check of an acquisition function; every method's settings and builder
  call it (`TypeError` unless an `EI` or `LCB` instance). `Scenarios` (`z (S, P)`,
  `w (S,)`, checked), `saa_scenarios(n, n_y, rng)` and `hybrid_acquisition` (variables
  `u`, then `vec` of the `(S, M)` x copies, then of the `(S, P)` y copies if variables;
  bounds and start bounds from each `Var`, repeated per scenario; `g` per scenario,
  followed by its GP tie). `AcquisitionSolution(value, x, y)` is one iteration's
  solution: the minimized objective at the trial point and, for hybrid BO, the
  scenario states `(S, M)`, `(S, P)`, which `_scenario_states` reads from the
  solution (or predicts, when the y_s are substituted).
- `bo.py`: `optimization_loop(problem, measure, initial_points, n_iterations,
  trial_point, rng, record) -> Run` (evaluate initial points, then per iteration
  `trial_point(problem, history, rng) -> (u, surrogate, acquisition solution)` and
  evaluate u; a
  `HybridBOError` gets the partial run as `error.run`). `standard_bo` and
  `hybrid_bo` take `rng` last positionally and `settings` keyword-only, defaulting to
  `StandardBOSettings()` / `HybridBOSettings()`. `StandardBOSettings` and
  `standard_bo` (fit `u -> f`, `standard_acquisition`, `multistart` from `lhs` over
  the `u` start bounds), `HybridBOSettings` and `hybrid_bo` (record, then
  `saa_scenarios`, then the loop; fit `h_inputs -> y`, `hybrid_acquisition`; adds the
  scenarios to the run and to `error.run`). `_method_record` builds both methods'
  records.
- `plots.py`: static figures, no widgets and no `rng`. `convergence(study, ax)` and
  `regret(study, optimum, ax)` draw `evaluations` (median and IQR per method over
  evaluations, `_over_evaluations`). `iteration(run, problem, i, *, measure, gp,
  scenarios, acquisition, settings) -> Figure` draws the state before trial point i
  over a 101-point grid of one-dimensional `u` (start bounds) plus the trial point,
  with one figure legend (`_figure_legend`). `_replay_settings` reads the acquisition
  function and `unknown_states_as_variables` from the record's settings text; the
  fields the replay uses (`_REPLAYED`) must appear in that text, so a custom solver
  or a non-default GP needs `settings`.
  `_replay`: standard BO evaluates `standard_acquisition`'s NLP; hybrid BO rebuilds
  `hybrid_acquisition` and solves it with u as an `nlpsol` parameter, outwards from
  the stored solution (NaN where a solve fails), then f per scenario at
  `_scenario_states`. Figures are drawn in `STYLE` (`hybrid_bo.mplstyle`).
  matplotlib and Polars are imported inside the functions.
- `errors.py`: `HybridBOError` (with `.history`, the points evaluated before the
  error, and `.run`, the partial run when raised in a method), `RootFindingError`,
  `NoSolutionError`, `ReconstructionWarning`, `OracleOptimumWarning`. The library
  raises these, never bare `Exception` (ruff `TRY002`/`BLE001` apply everywhere).
  It imports `History` and `Run` only for type checking, since `run.py` imports
  `gp.py`, which imports `errors.py`.

## Conventions and contracts

- **Shapes.** Rows are points: a point is `(n,)`, a set `(n_points, n)`, `f`
  `(n_points,)`. Column vectors appear only where a traced `cas.Function` is called
  (`g(x, y, u)` takes columns) and in `predict` with symbolic X.
- **Bounds.** Each `Var` has bounds and start bounds. Bounds box the acquisition
  problem and decide whether an `fsolve` root is accepted; they may be infinite.
  Start bounds are finite and set the ranges of initial points, multistart starts and
  `fsolve` starts.
- **Runs are pure data.** A `Run` holds arrays, numbers and strings only, so it
  saves as JSON (and pickles) without user code; the record is text. Plots are functions of a
  finished run.
- **Runtime aliases** (e.g. `Symbolic`) are used at runtime. Keep them as plain
  assignments, never `type X = ...`.

## Behavior to know

- **Randomness is per run.** The caller passes one generator per run (a study
  spawns them from its seed). A run uses only its own generator, in DESIGN.md's
  "Order of random draws":
  1. initial points (caller): `lhs` over the `u` start bounds;
  2. method setup: hybrid BO draws the scenarios (`rng.standard_normal((S, P))`);
     the baselines draw their whole design; standard BO draws nothing;
  3. each evaluation: oracle `fsolve` starts, then the noise (even when `noise_std`
     is 0), then root-finding starts;
  4. each iteration: GP-fit starts, then acquisition starts, then the evaluation.

  **`lhs` doesn't consume the stream.** scipy's `LatinHypercube(rng=rng)` spawns the
  next child of `rng`'s `SeedSequence`, so `lhs` calls (all `fsolve` starts,
  multistart starts, LHS designs) are ordered only among themselves, and stream draws
  (`normal`, `uniform`, `standard_normal`) only among themselves. The regression
  baselines depend on both orders, so adding, removing or reordering a draw in either
  is a logic change. `tests/test_sampling.py` pins this scipy behavior.
- **Errors propagate** out of a method: a `HybridBOError` carries the partial run
  (`error.run`). Only `repeat` (and so `study`) catches them, per run: it logs the
  error and keeps the partial run.
- **The library never writes files on its own.** Saving is an explicit `save` call.
- **Oracle incumbents.** `incumbents(history)` is the running minimum of the measured
  `f`. The benchmark metric `oracle_incumbents(history)` is `f_oracle` at the
  incumbent's point, not the running minimum of `f_oracle`; they agree only
  without noise.

## Testing rules

- `pixi run -e dev test` runs in about 6s. Run it for every change.
- `tests/test_regression.py` runs each method of `experiments/illustrative_example.py`
  (seed 0, 2 iterations) and compares with `tests/data/regression/*.json`. The
  baselines were recorded by the pre-refactor code, so the test maps each `Run` to
  their `ResultsBO` keys (`x` holds `[x, y]`, `*_initial`/`*_bo` halves), plus
  `acquisition_values` (BO methods) and `acquisition_x`/`acquisition_y` (hybrid BO)
  since #53. A non-logic PR must not change stored values. A logic PR regenerates
  them with `--update-baselines` and explains the diff. Regenerating on another
  machine moves the last digits of stored values, so a PR that only adds fields adds
  their keys to the stored files and leaves the other values as they are.
- The unit tests (`test_kernels.py`, `test_gp.py`, `test_scalers.py`,
  `test_sampling.py`, `test_solvers.py`, `test_system.py`, `test_evaluation.py`,
  `test_baselines.py`, `test_standard_bo.py`, `test_hybrid_bo.py`,
  `test_studies.py`, `test_views.py`) compare against
  numpy/scipy references or closed forms. Add one when changing math.
- Tests follow the comment rules in CLAUDE.md Conventions (docstrings, commented
  blocks, a note on the first use of each pytest feature); `test_standard_bo.py` is
  the example.
- A known bug gets a `pytest.mark.xfail(strict=True)` test and a GitHub issue
  labeled `bug`. The PR that fixes it removes the marker.
