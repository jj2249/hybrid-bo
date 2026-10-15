---
name: hybrid-bo-src
description: Use when working on src/hybrid_bo. Covers what the library computes (hybrid BO, standard BO and the baselines), the module map, data and shape conventions, behavior that surprises people, and the testing rules.
---

# Working on src/hybrid_bo

The refactor goal and workflow are in CLAUDE.md, and open bugs are in ROADMAP.md
under Known issues. This skill covers what the **current** code computes and the
contracts between its parts; DESIGN.md holds the target API and the reasons. The
code uses CONTEXT.md's terms; use them in PRs, tests and comments too.

## What the code computes ([arXiv 2603.11199](https://arxiv.org/abs/2603.11199))

    min_u f(x, y, u)   s.t.   0 = g(x, y, u),   y = h(x, u)  (unknown)

- **Evaluation** (`evaluate`): the experiment returns the measured variables for each
  trial point. The oracle simulation (`oracle_experiment`) solves `g` with
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

Everything lives at the top level of `hybrid_bo`, and `hybrid_bo/__init__.py`
re-exports the public API.

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
- `evaluation.py`: an `Experiment` is `(u (N,), rng) -> measured (P,)`, or an
  `OracleMeasurement` (measured plus oracle `x`, `y`) in test cases.
  `oracle_experiment(system, oracle, noise_std, ...)` traces the oracle like `g`.
  `evaluate(problem, experiment, u, rng)` runs the experiment per row, then always
  solves the root-finding problem (`_reconstruct`, starts over the unmeasured
  variables), and returns a `History` whose points are all initial points. Every
  fsolve goes through `_find_root` (CasADi Jacobian, accepts residual `< tol` within
  the bounds, silences fsolve's warnings). With an oracle, one extra solve from the
  oracle values (no draws) checks the reconstruction is on the same root, else
  `ReconstructionWarning`. `_start_bounds` (in `sampling.py`) and `_as_text` (in
  `problem.py`, an equation as text in the variable names) are shared helpers.
- `history.py`: `History` (`u`, `x`, `y`, `f`, oracle values or `None`, `n_initial`)
  and `incumbents`; `_concatenate` appends one history to another.
- `run.py`: `Run` (`history`, `record`, `surrogates`: one `FittedGP` per iteration,
  the one behind its trial point, empty for the baselines; `scenarios`, hybrid BO
  only, else None), pure data that pickles without user code (ADR 0003).
  `run_record(problem, experiment, rng, method, settings)` builds the text record;
  call it before the method draws. An experiment is recorded by its docstring, else
  its name (`oracle_experiment` sets the docstring).
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
  followed by its GP tie).
- `bo.py`: `optimization_loop(problem, experiment, initial_points, n_iterations,
  trial_point, rng, record) -> Run` (evaluate initial points, then per iteration
  `trial_point(problem, history, rng) -> (u, surrogate)` and evaluate u; a
  `HybridBOError` gets the partial run as `error.run`), `StandardBOSettings` and
  `standard_bo` (fit `u -> f`, `standard_acquisition`, `multistart` from `lhs` over
  the `u` start bounds), `HybridBOSettings` and `hybrid_bo` (record, then
  `saa_scenarios`, then the loop; fit `h_inputs -> y`, `hybrid_acquisition`; adds the
  scenarios to the run and to `error.run`). `_method_record` builds both methods'
  records.
- `plots.py`: `convergence(runs, ax, label)` and `iterations(run, problem, rng,
  settings, experiment, i) -> Figure`: every iteration's panels (`_Frame`, computed
  once by `_frame`) over a 101-point grid of one-dimensional `u` (start bounds),
  titled with the method, one figure legend (`_figure_legend`, outside the panels),
  and - and + buttons (kept on the figure as `_hybrid_bo_buttons`, since matplotlib
  holds widgets weakly) that redraw from the frames. The panels (`_gp_panel`, `_scenarios_panel`,
  `_acquisition_panel`) are private. The settings must match `run.record["method"]`
  (`TypeError`; None for the baselines, which show their points only). Figures are
  drawn in `STYLE` (`hybrid_bo.mplstyle`, the repo's rcParams); labels and legends
  are matplotlib's defaults. Hybrid BO's f under the scenarios solves g for each x_s with
  `_find_root` (NaN where no root); its acquisition curve is the SAA objective
  computed in numpy, standard BO's evaluates `standard_acquisition`'s NLP. matplotlib
  is imported inside the functions, so `import hybrid_bo` works without it.
- `errors.py`: `HybridBOError` (with `.history`, the points evaluated before the
  error, and `.run`, the partial run when raised in a method), `RootFindingError`,
  `NoSolutionError`, `ReconstructionWarning`. The library raises these, never bare
  `Exception` (ruff `TRY002`/`BLE001` apply everywhere but `examples/flash_unit/`).
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
  pickles without user code; the record is text. Plots are functions of a
  finished run.
- **Runtime aliases** (e.g. `Symbolic`) are used at runtime. Keep them as plain
  assignments, never `type X = ...`.

## Behavior to know

- **Randomness is per run.** The caller passes one generator per run (the example
  spawns them from one seed). A run uses only its own generator, in DESIGN.md's
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
- **Errors propagate.** Nothing in the library catches per run; a `HybridBOError`
  carries the partial run (`error.run`) and the example decides whether to save it
  and continue.
- **Oracle incumbents.** `incumbents(history)` is the running minimum of the measured
  `f`. The running minimum of `f_oracle` is not the oracle value at the measured
  incumbent (see the benchmark metric in DESIGN.md, #32).

## Testing rules

- `pixi run -e dev test` runs in about 6s. Run it for every change.
- `tests/test_regression.py` runs each method of `examples/illustrative_example.py`
  (seed 0, 2 iterations) and compares with `tests/data/regression/*.json`. The
  baselines were recorded by the pre-refactor code, so the test maps each `Run` to
  their `ResultsBO` keys (`x` holds `[x, y]`, `*_initial`/`*_bo` halves). A non-logic
  PR must not touch `tests/data/regression/`. A logic PR regenerates them with
  `--update-baselines` and explains the diff.
- The unit tests (`test_kernels.py`, `test_gp.py`, `test_scalers.py`,
  `test_sampling.py`, `test_solvers.py`, `test_system.py`, `test_evaluation.py`,
  `test_baselines.py`, `test_standard_bo.py`, `test_hybrid_bo.py`) compare against
  numpy/scipy references or closed forms. Add one when changing math.
- Tests follow the comment rules in CLAUDE.md Conventions (docstrings, commented
  blocks, a note on the first use of each pytest feature); `test_standard_bo.py` is
  the example.
- A known bug gets a `pytest.mark.xfail(strict=True)` test and a ROADMAP
  Known issues entry. The PR that fixes it removes the marker.
