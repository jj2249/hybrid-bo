---
name: hybrid-bo-src
description: Use when working on src/hybrid_bo. Covers what the library computes (hybrid BO, standard BO and the baselines), the module map, data and shape conventions, behavior that surprises people, and the testing rules.
---

# Working on src/hybrid_bo

The refactor goal and workflow are in CLAUDE.md, and open bugs are in ROADMAP.md
under Known issues. This skill covers what the **current** code computes and the
contracts between its parts. The target API is in DESIGN.md (stage 4: PRs 0 to 4, then issues #25 to #32);
each restructure PR updates this skill as it replaces code.

Use the terms in CONTEXT.md when writing about the code (PRs, tests, comments). The
current code predates them; this is the mapping:

| Current code | CONTEXT.md term |
|---|---|
| `x[indices_x_output_gp]` | unknown states `y` |
| the other entries of `x` | mechanistic states `x` |
| `h_known` | known equations `g` |
| `h_unknown` | the oracle (in residual form `h_unknown(u, x) = 0`) |
| `indices_u_input_gp`, `indices_x_input_gp` | the arguments of the unknown equations |
| GP samples, `gaussian_standard_samples`, `n_samples_gp` | scenarios, scenario draws `z`, number of scenarios |
| `*_no_noise`, "true" values | oracle values |
| `evaluate_with_fixed_x` | root-finding problem |
| routine | method |

## What the code computes ([arXiv 2603.11199](https://arxiv.org/abs/2603.11199))

    min_u f(u, x)   s.t.   h_known(u, x) = 0,   h_unknown(u, x) = 0

- **Evaluation** (`Problem.evaluate_with_noisy_simulation`): the oracle simulation
  solves `h_known` and `h_unknown` together for `x` given `u`, using multistart
  `fsolve`. Noise with std `noise_std` is drawn and added to the measured states
  (`Config.indices_x_measured`), and the root-finding problem then solves `h_known`
  for the other states. Both always happen, also when `noise_std` is 0.
- **Hybrid BO** (`bo.py`/`acquisition.py`; legacy `legacy/routines/hybrid_bo.py`
  until #30): the unknown equations are never used in the acquisition problem. A GP
  maps the `h_inputs` columns of `(u, x)` to `y` and is trained on the evaluated
  points. The acquisition problem is a sample-average approximation:
  - `n_scenarios` scenarios `mean + std * z_s`. The scenario draws `z_s` are
    drawn once per run and reused in every iteration.
  - Each scenario has its own copy `x_s` of the mechanistic states, and
    `g(x_s, y_s, u) = 0` holds for every scenario.
  - With `unknown_states_as_variables` (default True; legacy
    `use_output_gp_as_opt_var`) the `y_s` are variables with the constraint
    `y_s = scenario s`. Otherwise the GP prediction is substituted directly.
  - Acquisition function: SAA-EI is `sum_s w_s min(f_s - incumbent, 0)`, minimized.
    SAA-LCB is the weighted mean of `f_s` minus `std_factor` times the weighted std
    (`w = 1/S` gives legacy's `ddof=0`).
  - Solved with `multistart`. The trial point is the `u` part of the solution.
- **Standard BO** (`bo.py`; legacy `legacy/routines/standard_bo.py` until #30): a GP
  directly on `u -> f`, with analytic EI or LCB over `u` only.
- **Latin hypercube / uniform sampling**: the non-adaptive baselines. They draw the
  whole design at the start of a run and train no GP.

## Module map

**New code** lives at the top level of `hybrid_bo` and never imports
`hybrid_bo.legacy`; the top-level `hybrid_bo` exports only the new API. PR 4 and
issues #25 to #29 add modules here (each method ticket ports its own regression
test), and #30 switches the example over and deletes `legacy/`.

- `problem.py`: `Var` (name, bounds, start bounds), `System` (`u`, `x`, `y`, `g`,
  `h_inputs`, `measured`) and `Problem` (`system`, `f`). `g` and `f` are user
  callables over a namespace `v` of symbols (`v.x`), traced once in `_trace` into
  `cas.Function`s with column-vector inputs in the paper's order `(x, y, u)`. All
  input checks live in the constructors and raise `ValueError` (`NotImplementedError`
  for `P > 1`). `Symbolic` (default `cas.SX`) is the one place the symbol type is
  chosen.
- `sampling.py`: `lhs(lower, upper, n, rng)` and `uniform(...)` over a box, shape
  `(n, d)`; `initial_points(variables, n, rng)` is `lhs` over `Var` start bounds.
  Use them for any Latin hypercube or uniform draw; the hand-written copies in
  `legacy/` (fixed-x root-finding guesses, `create_lhs_samples`) go with it.
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
  only, else None), pure data that pickles
  without user code (ADR 0003). `run_record(problem, experiment, rng, method, settings)` builds the
  text record; call it before the method draws. An experiment is recorded by its
  docstring, else its name (`oracle_experiment` sets the docstring).
- `baselines.py`: `latin_hypercube_sampling` and `uniform_sampling` draw the whole
  design over the `u` start bounds, then `evaluate` initial points plus design in
  one call (same draw order as legacy). The regression test maps their `Run` to the
  legacy `ResultsBO` keys.
- `solvers.py`: `NLP` (`f`, bounds `(n,)`, optional `g(w) = 0`), `Ipopt` settings and
  `multistart(nlp, starts, solver) -> Solution` (`w`, `f`), the best start. It uses
  `cas.Opti` with bounds as constraints, like legacy, so IPOPT takes the same steps;
  failed starts are logged at debug level, `NoSolutionError` if all fail. The caller
  draws the starts with `lhs`.
- `kernels.py`: kernels as plain functions of CasADi matrices (`matern12/32/52`,
  `squared_exponential`, `linear`, `constant`); a lengthscale or scale is a scalar or
  a `(d, 1)` column (ARD).
- `gp.py`: `Hyper` (named, `size`, `log`), `GP` spec (hyperparameters in optimization
  order, `kernel(h, X1, X2)`, `mean(h, X)`, `jitter`), `default_gp(n)` (noise-free
  `outputscale`, unsquared as in legacy, times Matern-5/2 with ARD), `Scaler` with
  `minmax`/`standard`/`scale`/`unscale`, `FittedGP` (data only), `fit_gp`, `predict`
  (numpy X -> arrays `(n,)`, symbolic X -> `(n, 1)` columns) and `log_likelihood`. The
  hyperparameter named `noise_std`, if declared, is the noise; without it only
  `jitter` is added. Numeric data enter CasADi as `Symbolic` constants. `fit_gp`'s
  hyperparameters match the legacy GP bit for bit with the same data and generator; keep the
  `minmax` operation order, which reproduces legacy for [0, 1]. `_gp_record` gives
  the GP as text for a run's record (hyperparameters, kernel and mean expressions).
- `acquisition.py`: `EI`, `LCB(std_factor)` and `standard_acquisition` (analytic EI
  or LCB over `u`; returns the `NLP` and the `u` start bounds). `_check_acquisition`
  is the one check of an acquisition function; every method's settings and builder
  call it (`TypeError` unless an `EI` or `LCB` instance). `Scenarios` (`z (S, P)`,
  `w (S,)`, checked), `saa_scenarios(n, n_y, rng)` and `hybrid_acquisition` (variables
  `u`, then `vec` of the `(S, M)` x copies, then of the `(S, P)` y copies if variables;
  bounds and start bounds from each `Var`, repeated per scenario; `g` per scenario,
  followed by its GP tie). `_h_inputs(system, u, x)` gives the `h_inputs` columns of
  numeric or symbolic `u`, `x`, for training and for the acquisition problem.
- `bo.py`: `optimization_loop(problem, experiment, initial_points, n_iterations,
  trial_point, rng, record) -> Run` (evaluate initial points, then per iteration
  `trial_point(problem, history, rng) -> (u, surrogate)` and evaluate u; a
  `HybridBOError` gets the partial run as `error.run`), `StandardBOSettings` and
  `standard_bo` (fit `u -> f`, `standard_acquisition`, `multistart` from `lhs` over
  the `u` start bounds), `HybridBOSettings` and `hybrid_bo` (record, then
  `saa_scenarios`, then the loop; fit `h_inputs -> y`, `hybrid_acquisition`; adds the
  scenarios to the run and to `error.run`). `_method_record` builds both methods'
  records. Every regression test runs on the new API.
- `errors.py`: `HybridBOError` (with `.history`, the points evaluated before the
  error, and `.run`, the partial run when raised in a method), `RootFindingError`,
  `NoSolutionError`, `ReconstructionWarning`. New code raises these, never
  bare `Exception` (ruff `TRY002`/`BLE001` apply outside `legacy/` and `examples/`).
  It imports `History` and `Run` only for type checking, since `run.py` imports
  `gp.py`, which imports `errors.py`.

**The old code** is in `hybrid_bo/legacy/` (`from hybrid_bo.legacy import Config,
GP, Problem`). It may import new code (`legacy/problem.py` uses `lhs`), never the
reverse. Paths below are relative to `legacy/`.

- `routines/`: the four entry points, with the signature
  `(problem, config, [gp], u_initial, rngs, [optimizer], results_dir, create_plots)`.
  The sampling routines take no GP and no optimizer. They return `None` and pickle
  results into `results_dir`. Most of the loop is duplicated across the four files.
- `problem.py`: `Problem` holds `f`, `h_known`, `h_unknown` (CasADi
  functions) and the bounds, and evaluates the system. `fsolve` always uses the
  CasADi Jacobian.
- `gp.py`, `kernels.py`, `means.py`: an exact GP (Cholesky), kernels and means.
- `parameterization.py`: `Parameter` (symbol, value, bounds, transform) and
  `Parameterized`, the base of GP, Kernel and Mean.
- `optimizers.py`: CasADi/IPOPT, SciPy and multistart optimizers, which return
  `OptimizationResult`. The SciPy optimizer is unused and will be removed.
- `affine_transformers.py`: input and output scaling for the GP (`MinMax` for
  inputs, `Standard` for outputs).
- `array_operations/`: a dispatch layer, so the same code runs on numpy arrays
  and CasADi SX/MX/DM.
- `config.py`: `Config` is a frozen dataclass built from an example's
  `config.yaml`. `rng` and the derived index lists are set in `__post_init__`.
- `results.py`: the `ResultsBO` and `ResultsGP` dataclasses.

## Conventions and contracts

- **Shapes are mixed.** Single points passed to `Problem` methods and the
  optimizers are column vectors `(n, 1)`. Data sets are row-major
  `(n_points, n)`. Expect `.T` calls.
- **Bounds.** `Problem` has 8 `(n, 1)` arrays: {u, x} x {acquisition, starting
  points} x {lower, upper}.
  - Acquisition bounds are the box constraints of the acquisition problem. They
    also decide whether an `fsolve` root is accepted, and may be infinite.
  - Starting-point bounds set the ranges for multistart and `fsolve` initial
    guesses, so they must be finite.
- **`Problem` methods come in pairs.** `single_evaluate_*_original` does the
  real solve, and the plain `single_evaluate_*` delegates to it. Subclasses
  override the plain method (e.g. to add a fallback). Routines call the plain
  one; `original=True` is for plots.
- **Parameters.** `Parameter.value` is the raw value with shape (rows, cols).
  `transform_mode` ("identity"/"log") only applies in the optimizer during
  training.
- **Kernels.** `kernel.k(X1, X2)` uses current values and returns numpy for
  numpy inputs. `k_variable` is the symbolic version used to build the training
  problem.
- **GP numerics.**
  - `K = k(X, X) + (noise_std**2 + epsilon) I`, where `epsilon` is a `GP`
    constructor argument (default 1e-10) and is always added.
  - `predict` returns `(mean, full covariance)`.
  - `log_mll()` returns a (1, 1) array.
- **Runtime aliases.** `SymbolicType`, `CasadiType` and the kernel aliases
  (e.g. `RBFKernel`) are used at runtime. Keep them as plain assignments, never
  `type X = ...`.

## Behavior to know

- **Randomness is per run.** The caller passes one generator per run
  (`config.rng.spawn(n_runs_bo)`). A run uses only its own generator, in
  DESIGN.md's "Order of random draws" (set in PR 2):
  1. initial points (caller): `lhs` over the `u` starting-point bounds;
  2. hybrid BO: `z = rng.standard_normal((n_samples_gp, 1))`; LHS/uniform: the
     whole design; standard BO: nothing;
  3. each evaluation: oracle `fsolve` guesses, then
     `rng.normal(0.0, noise_std, size=(len(indices_x_measured), 1))` (even when
     `noise_std` is 0), then root-finding guesses;
  4. each iteration: GP-fit starts, then acquisition starts, then the evaluation.

  **`lhs` doesn't consume the stream.** scipy's `LatinHypercube(rng=rng)` spawns
  the next child of `rng`'s `SeedSequence`, so `lhs` calls (all `fsolve` guesses,
  multistart starts, LHS designs) are ordered only among themselves, and stream
  draws (`normal`, `uniform`, `standard_normal`) only among themselves. The
  regression baselines depend on both orders, so adding, removing or reordering a
  draw in either is a logic change. `config.rng` itself is only for spawning and
  for plots.
- Routines catch and print exceptions per run, so a failed run doesn't raise.
  Result arrays are NaN-initialized, so a crash shows up as NaN rows.
- Failed multistart starts print tracebacks too. That's noise, not failure.
- Files in `results_dir` are never overwritten (it warns and skips).
- `results_bo.pkl` is a `list[ResultsBO]`, one per run.
  - `*_initial` fields hold the initial points and `*_bo` fields the BO
    iterations.
  - `x`/`f` are the measured values and `*_no_noise` the oracle values.
  - `incumbents` is the running minimum of measured `f`. `incumbents_no_noise`
    is the running minimum of the oracle `f`, which is not the oracle value at the
    measured incumbent (see the benchmark metric in DESIGN.md).
- `results_gp.pkl` and the other pickles need `save_extended_results`.

## Testing rules

- `pixi run -e dev test` runs in about 6s. Run it for every change.
- `tests/test_regression.py` locks in the current behavior of every method on the
  illustrative example, run on the new API and mapped to the legacy `ResultsBO`
  keys. A non-logic PR must not touch
  `tests/data/regression/`. A logic PR regenerates them with
  `--update-baselines` and explains the diff.
- Unit tests (`test_kernels.py`, `test_gp.py`, `test_problem.py`,
  `test_sampling.py`, `test_system.py`, `test_evaluation.py`, `test_baselines.py`,
  `test_scalers.py`, `test_solvers.py`, `test_standard_bo.py`, `test_hybrid_bo.py`)
  compare against numpy/scipy references or the
  legacy code. Add one when changing math.
- Tests follow the comment rules in CLAUDE.md Conventions (docstrings, commented
  blocks, a note on the first use of each pytest feature); `test_standard_bo.py` is
  the example.
- A known bug gets a `pytest.mark.xfail(strict=True)` test and a ROADMAP
  Known issues entry. The PR that fixes it removes the marker.
