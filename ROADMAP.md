# Roadmap

Each item should become one small PR into `refactor/initial`. Tick items off in
the PR that completes them. `main` stays frozen until every stage is done.

The aim throughout is code that is easier to read, change and extend, so the
library can scale (see the goal in CLAUDE.md). `examples/flash_unit` is out of scope.
The target API and the plan for each restructure PR are in [DESIGN.md](DESIGN.md).

Keep the README up to date: any PR that changes usage, the public API or
behaviour updates it in the same PR. This matters most from stage 4 onwards,
when the examples and logic start to change.

## Stage 0: workflow
- [x] CLAUDE.md, ROADMAP.md, PR template, `ai-assisted` label

## Stage 1: environment and hooks
- [x] Target Python >=3.12 everywhere (ruff `target-version`, ty `python-version`)
- [x] Add `linux-64` to pixi platforms (for CI)
- [x] Ruff passes. `TRY002`/`BLE001` are ignored until the exceptions PR (stage 4). `__init__.py`
      re-exports are exempt from F401. Aliases use plain assignment because
      they're used at runtime (`isinstance`, `.sym`, kernel instantiation).
- [x] ty resolves example imports (`quality` env includes the `examples`
      feature). The hook is manual until the remaining 60 errors are fixed
      (stage 4).
- [x] Ignore example results dirs in git

## Stage 2: tests and skills
- [x] pytest feature and task
- [x] Seeded regression tests: each routine on the illustrative example with a
      few iterations, compared against stored results
- [x] Unit tests: kernels, GP predict/log-MLL against numpy references, Problem
      evaluation
- [x] Skill: working on `src/` (what the library does, not how)
- [x] Skill: working on `examples/`

## Stage 3: non-logic cleanup
- [x] Design the target API ([DESIGN.md](DESIGN.md))
- [x] Remove local-variable annotations and the ~100 `pyright: ignore` comments
      (`self.x: T` annotations stay until the restructure turns those classes into
      dataclasses)
- [x] Fix typos (`n_colums` in parameterization.py, "Bayesion" in config.py,
      shape-check messages in problem.py)

Long names and long argument lists are not separate items any more. They go away
with the restructure, which groups the data (`Bounds`, `Evaluation`, `History`).

## Stage 4: restructure (see DESIGN.md)
Every PR keeps the regression baselines unchanged unless it is marked *logic*.
- [ ] R0 *logic*: per-run randomness; LHS/uniform stop training GPs; noise has one
      column per measured state; regenerate the baselines
- [ ] R1: tag `pre-refactor` and exclude flash_unit; `Bounds`, `lhs`/`uniform`/`stack`;
      `Problem` as a frozen dataclass
- [ ] R2: `Simulator`, `Measurement`, `Evaluation` replace the 9 evaluation methods
- [ ] R3: `NLP`, solvers, `minimize`; nothing mutates optimisers; no tracebacks for
      expected failures (`optimizers.py:295`)
- [ ] R4: `Param`, functional kernels and means (removes `k_variable`/`m_variable`)
- [ ] R5: `Scaler`, `GP`/`FittedGP`/`fit_gp`
- [ ] R6: `EI`/`LCB`, `GPStructure`, acquisition builders (replace `setup_acq_problem`)
- [ ] R7: `run_bo`, `History`, `Run`, strategies; delete the 4 routines; the example saves
- [ ] R8: remove `Config` from the library
- [ ] R9: drop the `@overload` stubs; decide on `array_operations/`
- [ ] *logic*: exception types (`HybridBOError` and subclasses); remove the ruff
      `TRY002`/`BLE001` ignores
- [ ] *logic*: fix the Known issues below, one PR each

## Stage 5: examples
- [ ] CLI for `examples/main.py` (example, strategy, config overrides)
- [ ] Only require the minimum from users; make plot callbacks optional
- [ ] Decide whether YAML is the right config format
- [ ] Skill: specify an example in markdown (equations, parameters, assumptions)
- [ ] Port flash_unit (its custom evaluation becomes a custom `Simulator`)
- [ ] A short example in the style of DESIGN.md's target example, linked from the README

## Stage 6: publishing
- [ ] Merge `refactor/initial` into `main` (`main` stays frozen until then)
- [ ] README: fix the install section (it mentions `[dependency-groups]`, which
      doesn't exist)
- [ ] CI (lint, typecheck, tests)
- [ ] Tidy pixi environments (`examples` and `dev` are currently identical)
- [ ] Packaging and PyPI

## Known issues (fix only once a test covers them)
Each entry says where it gets fixed. "Goes away" means the restructure deletes
the code, and that PR removes the entry.
- `MinMaxTransformer.fit` (`affine_transformers.py:135`): the intercept is
  `-min/(max-min) + min_t`, but it should be `min_t - min*slope`. It's only correct
  for target ranges of width 1; the routines use the default [0, 1]. Covered by an
  xfail test. The fix leaves the regression baselines unchanged. *Logic PR after R5
  (`minmax`).*
- `LinearKernel.k_variable` with ARD (`kernels.py:279-290`) uses `scale.value`
  instead of `scale.variable()`, so the ARD scale is never trained. Covered by
  an xfail test. `LinearKernel` isn't used by the examples. *Goes away in R4*: with
  a single code path the bug can't exist. R4 removes the xfail marker and says so.
- `routines/hybrid_bo.py:769`: when `use_output_gp_as_opt_var=False`, the upper
  starting-point bounds use `problem.u_upper_bounds_acquisition`; this should
  probably be `u_upper_bounds_starting_points`. The two coincide in the
  illustrative example, which hides it. *R6 keeps it; logic PR after R6.*
- `formulation_acq` is validated strictly in `hybrid_bo.py` but with
  `startswith("lcb")` in `standard_bo.py:443`. *Goes away in R6 (`EI`/`LCB`).*
- `Config.epsilon_ei_standard` is unused; `ei_gp` uses its own default. *Goes away
  in R8.*
- `hybrid_bo.py:491` requires
  `len(indices_u_input_gp) + len(indices_x_input_gp) == problem.n_u`. This looks
  like an artifact of `main.py` building the GP with `problem.n_u` inputs; the
  real requirement is that the GP's `n_inputs` equals the number of GP inputs.
  *Goes away in R5/R6* (`fit_gp` takes its input size from the data).
- `problem.py:514`: the Latin hypercube for the free states' starting points
  uses `x_fixed.size` dimensions instead of `len(indices_x_free)`. The two are
  equal in the illustrative example. *R2 keeps it; logic PR after R2.*
- `Config.indices_x_opt` is computed but never used. *Goes away in R8.*
- `MultiStartOptimizer.create_lhs_samples` (`optimizers.py:161`) builds `msg`
  when `upper_bounds_starting_points is None` but never raises it. *Goes away in R3.*
- Measurement noise is generated with shape `(n_points, 1)`, so only one
  measured state is supported. *Fixed in R0.*
