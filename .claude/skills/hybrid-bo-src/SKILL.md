---
name: hybrid-bo-src
description: Use when working on src/hybrid_bo. Covers what the library computes (hybrid BO, standard BO and the baselines), the module map, data and shape conventions, behavior that surprises people, and the testing rules.
---

# Working on src/hybrid_bo

The refactor goal and workflow are in CLAUDE.md, and open bugs are in ROADMAP.md
under Known issues. This skill covers what the **current** code computes and the
contracts between its parts. The target API is in DESIGN.md (stage 4, PRs 0 to 10);
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
- **Hybrid BO** (`legacy/routines/hybrid_bo.py`): `h_unknown` is never used in the
  acquisition problem. A GP maps `[u[indices_u_input_gp], x[indices_x_input_gp]]`
  to `x[indices_x_output_gp]` and is trained on the evaluated points. The
  acquisition problem is a sample-average approximation:
  - `n_samples_gp` scenarios `mean + std * z_i`. The scenario draws `z_i` are
    drawn once per run and reused in every iteration.
  - Each scenario has its own copy of the states that are not decision variables,
    and `h_known(u, x_i) = 0` holds for every scenario.
  - With `use_output_gp_as_opt_var` the unknown states are variables of the
    acquisition problem with the constraint `x_out_i = scenario i`. Otherwise the
    GP prediction is substituted directly.
  - Acquisition function: SAA-EI is `mean_i(min(f_i - incumbent, 0))`. SAA-LCB is
    `mean(f_i) - factor_lcb_std * std(f_i)` (`ddof=0`).
  - Solved with a multistart optimizer. The trial point is the `u` part of the
    solution.
- **Standard BO** (`legacy/routines/standard_bo.py`): a GP directly on `u -> f`, with
  analytic EI or LCB over `u` only.
- **Latin hypercube / uniform sampling**: the non-adaptive baselines. They draw the
  whole design at the start of a run and train no GP.

## Module map

**New code** lives at the top level of `hybrid_bo` and never imports
`hybrid_bo.legacy`; the top-level `hybrid_bo` exports only the new API. PRs 4 to 8
add modules here, and PR 9 switches the example and the regression test over and
deletes `legacy/`.

- `problem.py`: `Var` (name, bounds, start bounds), `System` (`u`, `x`, `y`, `g`,
  `h_inputs`, `measured`) and `Problem` (`system`, `f`). `g` and `f` are user
  callables over a namespace `v` of symbols (`v.x`), traced once in `_trace` into
  `cas.Function`s with column-vector inputs in the paper's order `(x, y, u)`. All
  input checks live in the constructors and raise `ValueError` (`NotImplementedError`
  for `P > 1`). `Symbolic` (default `cas.SX`) is the one place the symbol type is
  chosen.
- `sampling.py`: `lhs(lower, upper, n, rng)` and `uniform(...)` over a box, shape
  `(n, d)`; `initial_points(variables, n, rng)` is `lhs` over `Var` start bounds.
  Use them for any Latin hypercube or uniform draw; the two remaining hand-written
  copies (fixed-x root-finding guesses, `create_lhs_samples`) go in PRs 5 and 6.

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

- `pixi run -e dev test` runs in about 2s. Run it for every change.
- `tests/test_regression.py` locks in the current behavior of every routine
  on the illustrative example. A non-logic PR must not touch
  `tests/data/regression/`. A logic PR regenerates them with
  `--update-baselines` and explains the diff.
- Unit tests (`test_kernels.py`, `test_gp.py`, `test_problem.py`,
  `test_sampling.py`, `test_system.py`, `test_transformers.py`) compare against numpy/scipy references. Add one when
  changing math.
- A known bug gets a `pytest.mark.xfail(strict=True)` test and a ROADMAP
  Known issues entry. The PR that fixes it removes the marker.
