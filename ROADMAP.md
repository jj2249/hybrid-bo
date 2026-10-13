# Roadmap

Each item should become one small PR into `refactor/initial`. Tick items off in
the PR that completes them. `main` stays frozen until every stage is done.

The aim throughout is code that is easier to read, change and extend, so the
library can scale (see the goal in CLAUDE.md). `examples/flash_unit` is out of scope.

## Stage 0: workflow
- [x] CLAUDE.md, ROADMAP.md, PR template, `ai-assisted` label

## Stage 1: environment and hooks
- [x] Target Python >=3.12 everywhere (ruff `target-version`, ty `python-version`)
- [x] Add `linux-64` to pixi platforms (for CI)
- [x] Ruff passes. `TRY002`/`BLE001` are ignored until stage 5. `__init__.py`
      re-exports are exempt from F401. Aliases use plain assignment because
      they're used at runtime (`isinstance`, `.sym`, kernel instantiation).
- [x] ty resolves example imports (`quality` env includes the `examples`
      feature). The hook is manual until the remaining 64 errors are fixed
      (stage 3/5).
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
- [ ] Remove noisy type annotations and most of the ~100 `pyright: ignore` comments
- [ ] Shorten long names (e.g. `x_upper_bounds_starting_points_samples`,
      `*_complete`/`*_initial`/`*_no_noise` variants in `routines/hybrid_bo.py`)
- [ ] Condense long argument lists (`Problem.__init__` takes 11 positional
      arrays; `evaluate_with_noisy_simulation` is called with a bare `False`)
- [x] Fix typos (`n_colums` in parameterization.py, "Bayesion" in config.py,
      shape-check messages in problem.py)

## Stage 4: examples
- [ ] CLI for `examples/main.py` (example, routine, config overrides)
- [ ] Only require the minimum from users; make plot functions optional
- [ ] Decide whether YAML is the right config format
- [ ] Skill: specify an example in markdown (equations, parameters, assumptions)

## Stage 5: src logic
- [ ] Extract the shared BO loop from the 4 routines (training, evaluation,
      results, saving)
- [ ] Split `do_hybrid_bo` (~410 lines) and `setup_acq_problem` (4x repeated
      bound concatenation)
- [ ] Remove overload explosions (`kernels.py`, `parameterization.py`,
      `affine_transformers.py`); reconsider `array_operations/`
- [ ] Make shape conventions consistent (column vs row)
- [ ] Remove the duplication between `Config.n_u`/`n_x` and `Problem`
- [ ] Stop routines mutating `MultiStartOptimizer` internals
- [ ] Use proper exception types; stop swallowing errors per run
- [ ] Don't print full tracebacks for expected failures, e.g. a multistart IPOPT
      start hitting max iterations (`optimizers.py:295`)
- [ ] Review the OO design overall

## Stage 6: publishing
- [ ] Merge `refactor/initial` into `main` (`main` stays frozen until then)
- [ ] README: fix the install section (it mentions `[dependency-groups]`, which
      doesn't exist)
- [ ] CI (lint, typecheck, tests)
- [ ] Tidy pixi environments (`examples` and `dev` are currently identical)
- [ ] Packaging and PyPI

## Known issues (fix only once a test covers them)
- `MinMaxTransformer.fit` (`affine_transformers.py:135`): the intercept is
  `-min/(max-min) + min_t`, but it should be `min_t - min*slope`. It's only correct
  for target ranges of width 1; the routines use the default [0, 1]. Covered by an
  xfail test. The fix leaves the regression baselines unchanged.
- `LinearKernel.k_variable` with ARD (`kernels.py:279-290`) uses `scale.value`
  instead of `scale.variable()`, so the ARD scale is never trained. Covered by
  an xfail test. `LinearKernel` isn't used by the examples.
- `routines/hybrid_bo.py:769`: when `use_output_gp_as_opt_var=False`, the upper
  starting-point bounds use `problem.u_upper_bounds_acquisition`; this should
  probably be `u_upper_bounds_starting_points`. The two coincide in the
  illustrative example, which hides it.
- `formulation_acq` is validated strictly in `hybrid_bo.py` but with
  `startswith("lcb")` in `standard_bo.py:443`.
- `Config.epsilon_ei_standard` is unused; `ei_gp` uses its own default.
- `hybrid_bo.py:491` requires
  `len(indices_u_input_gp) + len(indices_x_input_gp) == problem.n_u`. This looks
  like an artifact of `main.py` building the GP with `problem.n_u` inputs; the
  real requirement is that the GP's `n_inputs` equals the number of GP inputs.
- `problem.py:514`: the Latin hypercube for the free states' starting points
  uses `x_fixed.size` dimensions instead of `len(indices_x_free)`. The two are
  equal in the illustrative example.
- `Config.indices_x_opt` is computed but never used.
- `MultiStartOptimizer.create_lhs_samples` (`optimizers.py:161`) builds `msg`
  when `upper_bounds_starting_points is None` but never raises it.
- Measurement noise is generated with shape `(n_points, 1)`, so only one
  measured state is supported.
