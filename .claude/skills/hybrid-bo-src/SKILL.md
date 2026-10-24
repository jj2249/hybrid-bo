---
name: hybrid-bo-src
description: Use when working on src/hybrid_bo. Covers what the library computes (hybrid BO, standard BO and the baselines), the module map, data and shape conventions, behavior that surprises people, and the testing rules.
---

# Working on src/hybrid_bo

The workflow is in CONTRIBUTING.md, and open bugs are GitHub issues labeled `bug`.
This skill covers what the code computes and the contracts between its parts;
DESIGN.md holds the design and the reasons. The
code uses GLOSSARY.md's terms; use them in PRs, tests and comments too.

## What the code computes ([arXiv 2603.11199](https://arxiv.org/abs/2603.11199))

    min_u f(x, y, u)   s.t.   0 = g(x, y, u),   y = h(x, u)  (unknown)

- **Evaluation** (`evaluate`): the measurement returns the measured variables for each
  trial point. The oracle simulation (`oracle_measure`) solves `g` with
  `y = oracle(x, u)` for every state by multistart `fsolve`, reads the measured
  variables and adds `rng.normal(0, noise_std)` (drawn also when `noise_std` is 0).
  The library then always solves the **root-finding problem**: `g` for the unmeasured
  states given `u` and the measured variables.
- **Hybrid BO** (`bo.py`, `acquisition.py`): the unknown equations are never used in
  the acquisition problem. The surrogate model is one independent GP per unknown state
  (any P >= 1), each mapping the `h_inputs` columns of `(u, x)` to its column of `y`,
  all from the one prior GP in the settings and trained on the evaluated points.
  `HybridBOSettings.approximation` sets how the acquisition function is computed
  (`acquisition` says what): the sample-average approximation `SAA`, `SoftSAA` or
  `Linearization` (below). With SAA:
  - `SAA.n_scenarios` scenarios `mean + std * z_s` (unknown state j:
    `mean_j + std_j * z_sj`, from GP j). The scenario draws `z_s` are drawn once
    per run and reused in every iteration.
  - Each scenario has its own copy `x_s` of the mechanistic states, and
    `g(x_s, y_s, u) = 0` holds for every scenario.
  - With `SAA.unknown_states_as_variables` (default True) the `y_s` are variables with the
    constraint `y_s = scenario s`. Otherwise the GP prediction is substituted directly
    (the illustrative example's setting).
  - Acquisition function: SAA-EI is `sum_s w_s min(f_s - incumbent, 0)`, minimized.
    SAA-LCB is the weighted mean of `f_s` minus `std_factor` times the weighted std.
  - `SoftSAA(temperature, n_scenarios, unknown_states_as_variables)` is SAA over
    the same scenarios with `max(a_s, 0)`, `a_s = incumbent - f_s`, replaced by
    `tau log(1 + exp(a_s / tau))`; the problem minimizes `-log` of the weighted
    average (SoftSAA-EI), a logsumexp of `_log_softplus`, so it is finite with a
    nonzero gradient where SAA-EI is flat. The stored acquisition value is that
    `-log`; the iteration plot draws it as `-EI`, `-exp(-value)`. The temperature is required (positive, finite). EI only: LCB raises
    `ValueError`.
  - With `Linearization` the problem has one scenario, the nominal scenario (`z = 0`,
    weight 1), so `x̄` solves `g` with `ȳ` the GP's mean (or tied to it, by the same
    flag). Lin-EI is analytic EI of `N(f(x̄, ȳ, u), s²)`, with `s² = a aᵀ` and `a` the row
    **total** derivative `df/dz` by the implicit function theorem (`_sensitivity`;
    DESIGN.md), with `L = diag(std_1, ..., std_P)`. The partial `f_y L` is wrong: it
    gives `s ≡ 0` on the illustrative example. EI only: LCB raises
    `NotImplementedError`.
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

- `problem.py`: `Var` (name, bounds, start bounds), `System` (keyword-only `u`, `x`,
  `y`, `g`, `h_inputs`, `measured`; `x` and `g` optional and together: without them
  M = 0 and `g` is traced from no equations, a `cas.Function` with output (0, 1),
  whose artifact text is `none`) and `Problem` (`system`, `f`). `g` and `f` are user
  callables over a namespace `v` of symbols (`v.x`), traced once in `_trace` into
  `cas.Function`s with column-vector inputs in the paper's order `(x, y, u)`. All
  input checks live in the constructors and raise `ValueError`; any number P >= 1 of
  unknown states is allowed. `Symbolic` (`cas.SX`) is the symbol type of all library
  math. It is SX only: switching it to MX breaks the GP (`cas.chol` has no MX overload) and the
  artifact's equation text (DESIGN.md, Decisions). `_h_inputs(system, u, x)` gives the `h_inputs`
  columns of numeric or symbolic `u`, `x` (hybrid BO's GP training data and
  acquisition problem).
- `sampling.py`: `lhs(lower, upper, n, rng)` and `uniform(...)` over a box, shape
  `(n, d)`; `initial_points(variables, n, rng)` is `lhs` over `Var` start bounds. Use
  them for any Latin hypercube or uniform draw.
- `evaluation.py`: a `Measure` is `(u (N,), rng) -> measured (P,)`, or an
  `OracleMeasurement` (measured plus oracle `x`, `y`) in benchmark experiments.
  `oracle_measure(system, oracle, noise_std, ...)` (`noise_std` required) traces
  the oracle like `g` (`_trace_oracle`) and returns an `OracleSimulation`, a frozen,
  callable dataclass holding the system, the traced oracle and the settings (#141);
  its repr is the artifact's text, which `resume` compares across processes, so it
  must not depend on the process (no addresses).
  `oracle_optimum(problem, oracle, rng)` solves min f s.t. g = 0, y = oracle by
  `multistart` from `lhs` starts over the start bounds of (u, x, y).
  `evaluate(problem, measure, u, rng)` runs the measurement per row, then always
  solves the root-finding problem (`_reconstruct`, starts over the unmeasured
  variables), and returns a `History` whose points are all initial points. Every
  fsolve goes through `_find_root` (CasADi Jacobian, accepts residual `< tol` within
  the bounds, silences fsolve's warnings; with no unmeasured states, as with M = 0,
  it returns the empty root without solving; the starts are still drawn). With an
  oracle, one extra solve from the
  oracle values (no draws) checks the reconstruction is on the same root, else
  `ReconstructionWarning`. `_start_bounds` (in `sampling.py`) and `_as_text` (in
  `problem.py`, an equation as text in the variable names) are shared helpers.
- `history.py`: `History` (`u`, `x`, `y`, `f`, oracle values or `None`, `n_initial`)
  `incumbents` and `oracle_incumbents` (the benchmark metric); `_concatenate`
  appends one history to another.
- `run.py`: `Run` (`history`, `artifact`, `method`: its label, `artifact["method"]`
  unless a study relabels it; `index` in its study; `surrogates`: per iteration a
  `tuple[FittedGP, ...]`, the surrogate behind its trial point (hybrid BO: one GP per
  unknown state in the order of `system.y`; standard BO: a 1-tuple, the GP of f), and
  `acquisitions`: one
  `AcquisitionSolution` per iteration, both empty for the baselines; `scenarios`,
  hybrid BO only, else None; `settings`, the BO method's settings in full, else
  None; `n_evaluations`, its configuration's budget, set
  by `repeat`, else None), pure data that loads without user code (ADR 0003).
  `run_artifact(problem, measure, rng, method, settings)` builds the run's artifact,
  text, from `problem_artifact` (public: the CLI's task list holds it),
  `_generator_artifact` and `_versions_artifact` (a
  study's artifact reuses them); call it before the method draws. `code_version(path)` is
  `git describe --always --dirty` plus `-untracked` for untracked files that aren't
  ignored, or `unknown`; `is_clean(version)` says whether it pins the code (the CLI
  checks it before running). A measurement is recorded by
  its docstring, else its name; a dataclass measurement (`OracleSimulation`) by its
  repr.
- `studies.py`: `Configuration(method, n_evaluations, n_initial)` (a method with
  its own budget, `ValueError` unless 1 <= n_initial <= n_evaluations; it runs
  `n_evaluations - n_initial` iterations), `StudySettings(n_runs, seed)`,
  `Study(runs, settings, name, artifact)`, `Method` (called as `method(problem,
  measure, initial_points, n, rng=rng)`), `repeat(configuration, ...)` (run i from
  `default_rng(SeedSequence(seed).spawn(n_runs)[i])`, initial points then the method;
  catches `HybridBOError`, logs it and keeps `error.run`; sets each run's `index`
  and `n_evaluations`, the configuration's budget, since the method is code and
  can't be saved) and `study(configurations, problem, measure, settings, name,
  optimum)`, serial in the caller's process (no `n_workers`; #156, ADR 0010):
  resolves a None seed to fresh entropy kept in its settings, logs each
  configuration, then `study_tasks` -> `run_task` on each -> `assemble`. `Task(label,
  configuration, index, child)` is one run of a configuration; `study_tasks(configurations,
  settings)` gives them by label, then index (`ValueError` if the seed is None);
  `run_task(task, problem, measure, settings) -> Run | None` is one run as `repeat`
  describes (None: failed before evaluating anything; the run keeps its method's
  name); `assemble(tasks, runs, problem, measure, settings, name, optimum)` relabels
  runs with their task's label, drops the Nones (`ValueError` unless one per task),
  builds the artifact (records a stated `optimum` as `oracle_optimum`, never computes
  one, gives a BO method's prior GP, the run artifact's `gp`, on the line after its
  settings in `methods`, `no runs` for a label without runs) and warns with
  `OracleOptimumWarning` if an evaluated point's `f_oracle` is below the optimum.
  Run i of configurations with the same `n_initial` shares initial points. Running
  tasks elsewhere is the experiment CLI's job (processes over files). A study
  doesn't stop on Ctrl-C while IPOPT runs (#155), since CasADi catches the
  interrupt there; the CLI's does. Each configuration spawns its own
  `SeedSequence` children: `lhs` spawns from them, so sharing them across
  configurations changes the later ones' initial points. Studies take a seed, not an
  `rng`: the settings alone reproduce a study. **Prefix property**: no BO method
  depends on its remaining budget, so with the same seed and `n_initial` a longer
  run starts with exactly the shorter run's evaluations, fitted GPs and acquisition
  solutions (`tests/test_studies.py` pins it for hybrid and standard BO; the
  baselines draw their whole design first: Latin hypercube sampling lacks it, uniform
  sampling has it only with a measurement that draws nothing). Keep it: a budget sweep
  reads every budget from the longest run.
- `saving.py`: `save(study, path)` / `load(path)` (ADR 0004): `artifact.json` lists the
  run files `runs/<method>/run-<index:03d>.json`; `save_run(run, path)` (new file
  only, `FileExistsError`) and `load_run(path)` write and read one run file in that
  format (the caller writes atomically), and `save_artifact(study, path)` writes only
  the artifact beside run files already in place (`ValueError` if one is missing or
  its text isn't the run's encoding, `FileExistsError` if an artifact exists); `save`
  is `save_run` per run then the artifact; `load` also reads the older
  `study.json` (with `record` for `artifact`), a study saved before #144 (`n_initial`,
  `n_iterations` in its settings) with that budget as each run's `n_evaluations`, and a run saved before #114, whose
  surrogates are single fitted GPs, as 1-tuples (`_with_surrogate_tuples`, before
  `_check_format`); but a run saved before #119 (fitted
  GPs without `gp`, or a `record` with a `kernel` entry), a BO run saved before #120
  (a `record` with `gp` but no `settings` field) or any run saved before #123 (with
  `record`, the artifact's old name) raises `StudyFormatError`
  (`_check_format`), as do a tag naming no expected type and data that fits no
  member of a union. `_encode`/`_decode`
  convert any dataclass by its type hints; arrays are nested lists of floats (an empty
  one is `{"shape": ...}`). The GP types (`gp.TYPES`) and the settings types
  (`settings.TYPES`: the settings, `EI`, `LCB` and the approximations) are saved with
  `"type"`, their class name, and `_decode` picks a union's member by it (`_union_member` picks
  untagged data by its JSON shape: a list, an empty array's `{"shape": ...}`, another
  dict or a plain value). A new field on `Run` or its parts round-trips without
  changes if its type is one `_decode` handles. `save` refuses an existing folder.
- `views.py`: `evaluations(study, optimum=None)`, `summary(study)` and
  `hyperparameters(study)`, Polars data frames (one row per evaluation, per method,
  per hyperparameter entry of each fitted GP, named by its dotted path in the GP,
  e.g. `kernel.kernel.lengthscale`, with `output` naming the variable the GP models:
  the unknown state for hybrid BO, `f` for standard BO); each docstring lists its
  columns. Variable
  columns are named from the study's artifact (`_names` reads `Var(name='...')`).
  Polars is imported inside the functions (`_polars`), so `import hybrid_bo` works
  without it; a missing Polars raises `ImportError` naming `hybrid-bo[analysis]`.
- `baselines.py`: `latin_hypercube_sampling` and `uniform_sampling` draw the whole
  design over the `u` start bounds, then `evaluate` initial points plus design in
  one call.
- `solvers.py`: `NLP` (`f`, bounds `(n,)`, optional `g(w) = 0`; a `g` of no rows is
  no constraint, since `cas.Opti` refuses an empty one), `Ipopt` settings and
  `multistart(nlp, starts, solver) -> Solution` (`w`, `f`), the best start; objectives
  within `_TIE_TOLERANCE` (1e-13, relative or absolute below 1) tie and the earlier
  start wins, since acquisition problems can have symmetric or flat optima whose
  values differ only by platform-dependent rounding. Loosening it moves the GP fits'
  hyperparameters and every BO baseline. It uses
  `cas.Opti` with bounds as constraints, as the pre-refactor code did, so IPOPT takes
  the steps the baselines record; failed starts, and each call's number of failed
  starts and time under `nlp.f`'s name, are logged at debug level; starts that
  stopped at IPOPT's iteration limit (`opti.stats()["return_status"]` is
  `Maximum_Iterations_Exceeded`) get one warning per call, giving their count and
  `nlp.f`'s name, before `NoSolutionError` if all fail. The caller draws the starts with `lhs`.
- `gp.py` (ADR 0008): GPs are data, frozen dataclasses from a closed set. `Param(init,
  bounds, log=True)` is a hyperparameter to fit (checked: init inside the bounds, positive
  bounds for log). Kernels `Matern12/32/52` and `SquaredExponential` (`lengthscale:
  float | tuple | Param`, `ard=True`) and `Scale(kernel, outputscale)` (unsquared;
  `outputscale * kernel`, a number or a `Param` on either side, builds it through
  `_Scalable`, the kernels' only operator: `kernel * kernel` raises `TypeError`),
  `Kernel` their union; the `Zero` mean (`Mean`); `TYPES` the closed set, which
  saving tags by name. `GP(mean, kernel, noise_std, jitter=1e-6)` (keyword-only) is the prior
  GP; `FittedGP(gp, X, Y, x_scaler, y_scaler)` the fitted GP, its `gp` with no
  `Param` left (checked), scaled data and scalers. `covariance(kernel, X1, X2)` and
  `prior_mean(mean, X)` match on the type; the formulas are the cases' bodies. A
  hyperhyperparameter is a field whose type hint admits `Param`; `_hyperparameters(node, n_inputs)`
  lists them as (dotted path, value, entries or None) in fit order, a node's own
  before those of the kernels it holds (so `Scale(Matern52())` fits outputscale,
  then the lengthscales, as `default_gp` did: the baselines depend on that order),
  and `_with_hyperparameters` sets them by path. A fitted value is a float, or a tuple of
  one per input for an ARD lengthscale; a fixed number applies to every input.
  `fit_gp(gp, X, Y, rng, ...) -> FittedGP` fits every `Param` (log ones in log
  space) by `multistart` from `lhs` starts over the bounds; with no `Param` it draws
  nothing. `predict(fitted, X)` (the noise-free conditional mean and std; numpy X ->
  arrays `(n,)`, built at one row and mapped over the rows, symbolic X -> `(n, 1)`
  columns) and `log_marginal_likelihood(fitted)`,
  which `fit_gp` maximizes. Kernel hyperparameters are in scaled units; `noise_std` is a
  known noise in **measured** units, scaled by the output scaler's slope:
  `K = k(X, X) + ((noise_std * slope)**2 + jitter) I`. Hyperparameters in the fit are SX
  slices put in the dataclass fields. `Scaler` with `minmax`/`standard`/`scale`/
  `unscale`. The jitter defaults to 1e-6, a nugget: much smaller and the fit stalls
  once the inputs cluster (#107, DESIGN.md). After fitting, `fit_gp` warns if the
  condition number of that `K` at the kept values (numpy, scaled data) exceeds
  `CONDITION_THRESHOLD` (1e10), giving it and the jitter.
  Numeric data enter CasADi as `Symbolic` constants. Keep the `minmax` operation
  order: it reproduces the pre-refactor values bit for bit for [0, 1], which the
  baselines depend on.
- `acquisition.py`: `EI`, `LCB(std_factor)` and `standard_acquisition` (analytic EI
  or LCB over `u`). Both builders return an `AcquisitionProblem` (`nlp`,
  `start_bounds`, and the variable order: `n_u`, `n_x`, `n_scenarios`,
  `unknown_states_as_variables`; the defaults are standard BO's, `w = u`).
  `_pack(u, x, y)` (numpy or CasADi) and `_unpack(problem, w) -> (u, x, y | None)`
  are the only code that knows the order of `w`: never slice or reshape `w`
  elsewhere. `_check_acquisition`
  is the one check of an acquisition function; every method's settings and builder
  call it (`TypeError` unless an `EI` or `LCB` instance). `SAA(n_scenarios,
  unknown_states_as_variables)` is hybrid BO's approximation, checked by
  `_check_approximation(acquisition, approximation)` in `HybridBOSettings` and
  `hybrid_acquisition` (`TypeError` unless an `SAA` or a
  `Linearization(unknown_states_as_variables)` or a `SoftSAA(temperature, ...)`,
  `NotImplementedError` for `LCB` with `Linearization`, `ValueError` for `LCB` with
  `SoftSAA`). `nominal_scenario(n_y)` is linearization's one scenario, which
  `hybrid_acquisition` requires with it (`ValueError` otherwise);
  `_sensitivity(problem, fitted)` is the `cas.Function` `(x, y, u) -> a (1, P)`.
  Every hybrid builder (`hybrid_acquisition`, `_scenario_unknown_states`,
  `_sensitivity`, `_scenario_states`) takes `fitted`, the surrogate model: a tuple of
  one fitted GP per unknown state (`ValueError` in `hybrid_acquisition` otherwise).
  `Scenarios` (`z (S, P)`,
  `w (S,)`, checked), `saa_scenarios(n, n_y, rng)` and `hybrid_acquisition(..., acquisition,
  approximation)` (variables
  `u`, then `vec` of the `(S, M)` x copies, then of the `(S, P)` y copies if variables;
  bounds and start bounds from each `Var`, repeated per scenario; `g` per scenario,
  followed by its GP tie). `_scenario_unknown_states` is the one GP scenario
  prediction (`mean_j + std_j * z_sj` per unknown state j), symbolic or numeric.
  `AcquisitionSolution(value, x, y)` is one iteration's solution: the minimized
  objective at the trial point and, for hybrid BO, the scenario states `(S, M)`,
  `(S, P)`, which `_scenario_states(system, fitted, scenarios,
  acquisition_problem, w)` reads from the solution (or predicts, when the y_s are
  substituted).
- `settings.py`: `StandardBOSettings` and `HybridBOSettings`, apart from `bo.py`
  because `run.py` imports them and `bo.py` imports `run.py`; `MethodSettings`, their
  union (`Run.settings`, the loop and the plots use it); `TYPES`, the types saving
  tags.
- `bo.py`: `optimization_loop(problem, measure, initial_points, n_iterations,
  trial_point, rng, artifact, *, settings) -> Run` (evaluate initial points, then per
  iteration `trial_point(problem, history, rng) -> (u, surrogate, acquisition
  solution)`, the surrogate a tuple of fitted GPs, and evaluate u; the run keeps
  `settings`; a `HybridBOError` gets the partial run as `error.run`). `standard_bo` and
  `hybrid_bo` take `rng` last positionally and `settings` keyword-only and required:
  both settings require `gp`, the prior GP. `StandardBOSettings` and
  `standard_bo` (fit `u -> f`, `standard_acquisition`, `multistart` from `lhs` over
  the `u` start bounds), `HybridBOSettings` and `hybrid_bo` (artifact, then
  `saa_scenarios` or `nominal_scenario`, then the loop; fit one GP `h_inputs -> y_j`
  per unknown state, in the order of `system.y`, `hybrid_acquisition`; adds the
  scenarios to the run and to `error.run`). `_method_artifact` builds both methods'
  artifacts: the settings without the GP, and `gp`, the prior GP's repr.
- `plots.py`: static figures, no widgets and no `rng`. `convergence(study, ax)` and
  `regret(study, optimum, ax)` draw `evaluations` (median and IQR per method over
  evaluations, `_over_evaluations`). A sweep over initial fractions of one method
  has `budget_heatmap`, `regret_at_budget(study, budget)` and
  `regret_by_initial_fraction`, all from `_sweep_regret` (regret per evaluation with
  each run's `history.n_initial`; `ValueError` unless every run has the same
  `artifact["method"]` and `repr(run.settings)`; of runs with the same n_initial
  and index only the longest is kept, by the prefix property, so they count once).
  `iteration(run, problem, i, *, measure, gp,
  scenarios, acquisition) -> Figure` draws the state before trial point i over the
  `u` start bounds, with one figure legend (`_figure_legend`); it dispatches on
  `n_u` (`ValueError` above 2, and for hybrid BO's GP panels with P > 1: they draw
  `surrogates[i][0]` and `y[:, 0]`; with `gp=False` the replay works for any P).
  1D (`_iteration_1d`): stacked curves over 101 points plus the trial point. 2D (`_iteration_2d`): filled contours over a 30 x 30 grid, in
  rows of `_Map` panels (oracle | mean | std | acquisition; empty rows and columns
  dropped), oracle and mean sharing a color scale, panels of f log-scaled when
  positive over more than two decades, EI as log10 EI (`_log10_ei`: SoftSAA's
  `-value / ln 10`, never -inf; else `log10(-value)`, -inf where EI <= 0) with -inf
  in gray cells and the color scale at most `_EI_DECADES` (10) decades below the
  largest EI, a `min` arrow if values fall below (LCB linear); the acquisition
  panel notes the stored and replayed value at the trial point. The GP is
  iteration i's stored fitted GP, for the GP panel and the replay alike; the rest of the replay comes from
  `run.settings` (its type says hybrid or standard BO; None draws the points only).
  The artifact's settings text is never read back.
  `_replay`: standard BO evaluates `standard_acquisition`'s NLP; hybrid BO rebuilds
  `hybrid_acquisition` but optimizes nothing: with u fixed, each scenario's equations
  are a square system (`_scenario_equations`: M equations in x, or M + P in (x, y)
  with the unknown states as variables, the GP tie included; no equation links two
  scenarios), which `_scenario_roots` solves with CasADi's Newton `rootfinder`,
  mapped over the (point, scenario) pairs of one ring of points at a time, rings one
  median nearest-point spacing wide outwards from the trial point, each point
  started from the nearest point solved in an earlier ring, the trial point's ring
  from the stored solution. A point is NaN if a pair's residual exceeds
  `_RESIDUAL_TOLERANCE` (1e-8; mapped calls report no per-instance success, so
  Newton runs with `error_on_fail: False`) or its root is outside a state's bounds
  (solved without bounds, then rejected: IPOPT could have found another root inside
  them; and past a fold of the branch Newton stalls where IPOPT could jump to
  another root, so a gap there is by design). With M = 0 and the GP substituted
  nothing is solved. The value is `nlp.f`
  at the packed (u, x, y), mapped over the solved points, so it covers every
  approximation. `_check_trial_point` warns with `ReplayWarning` unless the trial
  point's states and value are the stored solution's (`_REPLAY_RTOL` 1e-4,
  `_REPLAY_ATOL` 1e-6); pytest turns that warning into an error (`pyproject.toml`),
  so a test that expects it uses `pytest.warns`. Then f per
  scenario in one mapped call of `_scenario_objective`, and with linearization `s`
  from `_sensitivity` (the scenarios
  panel draws nominal f ± s, `_nominal_band`). Figures are drawn in `STYLE` (`hybrid_bo.mplstyle`).
  matplotlib and Polars are imported inside the functions.
- `errors.py`: `HybridBOError` (with `.history`, the points evaluated before the
  error, and `.run`, the partial run when raised in a method), `RootFindingError`,
  `NoSolutionError`, `ReconstructionWarning`, `OracleOptimumWarning`, `ReplayWarning`
  (an iteration figure's replay left the optimizer's branch at the trial point), and
  `StudyFormatError` (a `ValueError`: a saved study this version can't load). The library
  raises these, never bare `Exception` (ruff `TRY002`/`BLE001` apply everywhere).
  It imports `History` and `Run` only for type checking, since `run.py` imports
  `gp.py`, which imports `errors.py`.

## Conventions and contracts

- **Defaults never choose the method** (ADR 0009): a default may set how hard or how
  precisely we search, never what we search for. Required, with no default:
  `acquisition` on both settings, `approximation` on `HybridBOSettings` and
  `hybrid_acquisition`, `n_scenarios` on `SAA` and `SoftSAA`, `temperature` on
  `SoftSAA`, `std_factor` on `LCB`, `noise_std` on `oracle_measure` and `GP`. Defaults
  stay for search effort, numerics and plumbing (`n_starts`, `n_starts_gp`, `tol`,
  `solver`, `jitter`, kernel priors and `ard`, `Var` bounds, `seed`, plot flags, and
  `unknown_states_as_variables=True`, a formulation with the same optimum). The
  settings and choice dataclasses (`EI`, `LCB`, `SAA`, `SoftSAA`, `Linearization`,
  both settings, `GP`, `Configuration`, `StudySettings`) are `kw_only=True`;
  `tests/test_required_arguments.py` pins both. A new setting that chooses the method
  gets no default.
- **Shapes.** Rows are points: a point is `(n,)`, a set `(n_points, n)`, `f`
  `(n_points,)`. Column vectors appear only where a traced `cas.Function` is called
  (`g(x, y, u)` takes columns) and in `predict` with symbolic X.
- **Bounds.** Each `Var` has bounds and start bounds. Bounds box the acquisition
  problem and decide whether an `fsolve` root is accepted; they may be infinite.
  Start bounds are finite and set the ranges of initial points, multistart starts and
  `fsolve` starts.
- **Runs are pure data.** A `Run` holds arrays, numbers and strings only, so it
  saves as JSON (and pickles) without user code; the artifact is text. Plots are functions of a
  finished run.
- **Runtime aliases** (e.g. `Symbolic`) are used at runtime. Keep them as plain
  assignments, never `type X = ...`.

## Behavior to know

- **Randomness is per run.** The caller passes one generator per run (a study
  spawns them from its seed). A run uses only its own generator, in DESIGN.md's
  "Order of random draws":
  1. initial points (caller): `lhs` over the `u` start bounds;
  2. method setup: hybrid BO with SAA or SoftSAA draws the scenarios
     (`rng.standard_normal((S, P))`), with linearization nothing;
     the baselines draw their whole design; standard BO draws nothing;
  3. each evaluation: oracle `fsolve` starts, then the noise (even when `noise_std`
     is 0), then root-finding starts;
  4. each iteration: GP-fit starts (hybrid BO with P > 1: each GP's own `lhs`
     starts, in the order of `system.y`), then acquisition starts, then the
     evaluation.

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
- **Progress is logged, never configured.** `study` logs each method, `repeat` each
  run, `optimization_loop` the initial points and each iteration (with the incumbent)
  at INFO; `fit_gp` and `multistart` log their times at DEBUG, and warn about a
  nearly singular training covariance and starts at the iteration limit. Times come from
  `time.perf_counter`. The library adds no handlers to the caller's process; the
  experiment CLI's task processes save each run's records as its per-run log, and the
  launcher appends them, tagged `<label>/run-003`, to the study log.
- **Oracle incumbents.** `incumbents(history)` is the running minimum of the measured
  `f`. The benchmark metric `oracle_incumbents(history)` is `f_oracle` at the
  incumbent's point, not the running minimum of `f_oracle`; they agree only
  without noise.

## Testing rules

- `pixi run -e dev test` runs in about 2 minutes (each CLI study task is a new process). Run it for every change.
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
  `test_baselines.py`, `test_standard_bo.py`, `test_hybrid_bo.py` (P = 2 unit tests
  against numpy and finite differences), `test_no_known_equations.py` (M = 0 end
  to end: every method, saving, plots, and hybrid BO matching standard BO when
  f = y), `test_two_unknown_states.py` (P = 2 end to
  end: runs, saving, plots),
  `test_studies.py`, `test_views.py`, `test_run.py`, `test_replay.py` (the replay
  against values recorded from the IPOPT replay it replaced, `tests/data/replay/`)) compare against
  numpy/scipy references or closed forms. Add one when changing math.
- Tests follow the comment rules in CLAUDE.md Conventions (docstrings, commented
  blocks, a note on the first use of each pytest feature); `test_standard_bo.py` is
  the example.
- A known bug gets a `pytest.mark.xfail(strict=True)` test and a GitHub issue
  labeled `bug`. The PR that fixes it removes the marker.
