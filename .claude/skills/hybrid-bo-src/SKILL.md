---
name: hybrid-bo-src
description: Use when working on src/hybrid_bo. Covers what the library computes (hybrid BO and baselines), the module map, data and shape conventions, behaviour that surprises people, and the testing rules.
---

# Working on src/hybrid_bo

The refactor goal and workflow are in CLAUDE.md, and open bugs are in ROADMAP.md
under Known issues. This skill covers what the code computes and the contracts
between its parts.

## The method ([arXiv 2603.11199](https://arxiv.org/abs/2603.11199))

A system with decision variables `u` (n_u) and states `x` (n_x):

    min_u f(u, x)   s.t.   h_known(u, x) = 0,   h_unknown(u, x) = 0

- **Simulation ("the experiment")**: `Problem` solves `h_known` and `h_unknown`
  together for `x` given `u`, using multistart `fsolve`. In this role
  `h_unknown` is the true system. Optional noise is added to the measured states
  (`Config.indices_x_measured`), and the other states are re-solved from
  `h_known`.
- **Hybrid BO** (`routines/hybrid_bo.py`): `h_unknown` is never used in the
  acquisition problem. A GP maps `[u[indices_u_input_gp], x[indices_x_input_gp]]`
  to `x[indices_x_output_gp]` and is trained on the evaluated points. The
  acquisition problem is a sample-average approximation:
  - `n_samples_gp` GP samples `mean + std * z_i`. The standard normals `z_i` are
    drawn once per run and reused in every iteration.
  - Each sample has its own copy of the non-u states, and `h_known(u, x_i) = 0`
    holds for every sample.
  - With `use_output_gp_as_opt_var` the GP outputs are decision variables with
    the constraint `x_out_i = GP sample i`. Otherwise the sample is substituted
    directly.
  - Objective: EI is `mean_i(min(f_i - incumbent, 0))`. LCB is
    `mean(f_i) - factor_lcb_std * std(f_i)`.
  - Solved with a multistart optimiser. The next point is the `u` part of the
    solution.
- **Standard BO** (`routines/standard_bo.py`): a GP directly on `f(u)`, with
  analytic EI or LCB over `u` only. It's the baseline.
- **Latin hypercube / uniform sampling**: non-adaptive baselines. They draw the
  whole design at the start of a run and train no GP.

## Module map

This describes the current code. The restructure (DESIGN.md, PRs R0 to R9)
replaces it piece by piece, and each restructure PR updates this section.

- `routines/`: the four entry points, with the signature
  `(problem, config, [gp], u_initial, rngs, [optimizer], results_dir, create_plots)`.
  The sampling routines take no GP and no optimizer. They return `None` and pickle
  results into `results_dir`. Most of the loop is duplicated across the four files.
- `problem.py`: `Problem` holds `f`, `h_known`, `h_unknown` (CasADi
  functions) and the bounds, and evaluates the system.
- `gp.py`, `kernels.py`, `means.py`: an exact GP (Cholesky), kernels and means.
- `parameterization.py`: `Parameter` (symbol, value, bounds, transform) and
  `Parameterized`, the base of GP, Kernel and Mean.
- `optimizers.py`: CasADi/IPOPT, SciPy and multistart optimisers, which return
  `OptimizationResult`.
- `affine_transformers.py`: input and output scaling for the GP (`MinMax` for
  inputs, `Standard` for outputs).
- `array_operations/`: a dispatch layer, so the same code runs on numpy arrays
  and CasADi SX/MX/DM.
- `config.py`: `Config` is a frozen dataclass built from an example's
  `config.yaml`. `rng` and the derived index lists are set in `__post_init__`.
- `results.py`: the `ResultsBO` and `ResultsGP` dataclasses.

## Conventions and contracts

- **Shapes are mixed.** Single points passed to `Problem` methods and the
  optimisers are column vectors `(n, 1)`. Data sets are row-major
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
  `transform_mode` ("identity"/"log") only applies in the optimiser during
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

## Behaviour to know

- **Randomness is per run.** The caller passes one generator per run
  (`config.rng.spawn(n_runs_bo)`). A run uses only its own generator, in a fixed
  order: initial points (caller), noise, `z` or the sampling design, then the loop
  (DESIGN.md has the exact calls). The baselines depend on this order, so changing
  it, or adding a draw, is a logic change. `config.rng` itself is only for spawning
  and for plots.

- Routines catch and print exceptions per run, so a failed run doesn't raise.
  Result arrays are NaN-initialised, so a crash shows up as NaN rows.
- Failed multistart starts print tracebacks too. That's noise, not failure.
- Files in `results_dir` are never overwritten (it warns and skips).
- `results_bo.pkl` is a `list[ResultsBO]`, one per run.
  - `*_initial` fields hold the initial points and `*_bo` fields the BO
    iterations.
  - `x`/`f` are the measured values and `*_no_noise` the true values.
  - `incumbents*` is the running minimum of `f`.
- `results_gp.pkl` and the other pickles need `save_extended_results`.

## Testing rules

- `pixi run -e dev test` runs in about 2s. Run it for every change.
- `tests/test_regression.py` locks in the current behaviour of every routine
  on the illustrative example. A non-logic PR must not touch
  `tests/data/regression/`. A logic PR regenerates them with
  `--update-baselines` and explains the diff.
- Unit tests (`test_kernels.py`, `test_gp.py`, `test_problem.py`,
  `test_transformers.py`) compare against numpy/scipy references. Add one when
  changing maths.
- A known bug gets a `pytest.mark.xfail(strict=True)` test and a ROADMAP
  Known issues entry. The PR that fixes it removes the marker.
