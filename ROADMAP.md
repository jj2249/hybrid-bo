# Roadmap

Each item should become one small PR into `refactor/initial`. Tick items off in
the PR that completes them. `main` stays frozen until every stage is done.

The aim throughout is code that is easier to read, change and extend, so the
library can scale (see the goal in CLAUDE.md). `examples/flash_unit` is out of scope.
The target API and the plan for each restructure PR are in [DESIGN.md](DESIGN.md).

Keep the README up to date: any PR that changes usage, the public API or
behavior updates it in the same PR. This matters most from stage 4 onwards,
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
      feature). The hook is manual until the remaining 57 errors are fixed
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
Every PR keeps the regression baselines unchanged unless it is marked *logic*. The
design was revised after R0 (CONTEXT.md, docs/adr/); the PRs below replace the
earlier R1-R9.
- [x] R0 *logic*: per-run randomness; LHS/uniform stop training GPs; noise has one
      column per measured state; regenerate the baselines
- [x] 0: design review: CONTEXT.md, ADRs 0001-0003, DESIGN.md rewritten, this list
- [x] 1: R1 trimmed (#17): exclude flash_unit; `lhs`/`uniform` on arrays; delete
      `Problem.solve`; remove `use_jacobian`; tidy `examples/main.py`
- [ ] 2 *logic*: draw order from DESIGN.md; always reconstruct after the experiment;
      regenerate the baselines
- [ ] 3: move the old code to `hybrid_bo/legacy/`
- [ ] 4: `Var`, `System`, `Problem`
- [ ] 5: `oracle_experiment`, root-finding, `evaluate`, `History`, error types (removes
      the ruff `TRY002`/`BLE001` ignores once nothing raises bare `Exception`)
- [ ] 6: `NLP`, `Ipopt`, `multistart`; no tracebacks for expected failures
- [ ] 7: `Hyper`, kernel functions, `GP`, `fit_gp`, `predict`, scalers, `default_gp`
- [ ] 8: `EI`/`LCB`, `Scenarios`, the two acquisition builders
- [ ] 9: switch: `optimization_loop`, the four methods, `Run`; port the regression
      test and the illustrative example; delete `legacy/`, `Config` and YAML
- [ ] 10: `hb.plots`; CLI for the examples
- [ ] *logic*: benchmark metric (oracle objective at the believed best point)

## Stage 5: examples
- [ ] Only require the minimum from users (defaults recorded in the run)
- [x] Decide whether YAML is the right config format: no, examples are Python
      (DESIGN.md); YAML is deleted in PR 9
- [ ] Skill: specify an example in markdown (equations, parameters, assumptions)
- [ ] Port flash_unit (its custom evaluation becomes a custom experiment)
- [ ] A short example in the style of DESIGN.md's target example, linked from the README

## Stage 6: publishing
- [ ] Merge `refactor/initial` into `main` (`main` stays frozen until then)
- [ ] README: fix the install section (it mentions `[dependency-groups]`, which
      doesn't exist)
- [ ] CI (lint, typecheck, tests)
- [ ] Tidy pixi environments (`examples` and `dev` are currently identical)
- [ ] Packaging and PyPI

## Known issues (fix only once a test covers them)
Each entry says where it gets fixed. "Goes away" means the new code does not reproduce
it (DESIGN.md: Known issues are not rebuilt), and that PR removes the entry.
- `MinMaxTransformer.fit` (`affine_transformers.py:135`): the intercept is
  `-min/(max-min) + min_t`, but it should be `min_t - min*slope`. It's only correct
  for target ranges of width 1; the routines use the default [0, 1]. Covered by an
  xfail test. The fix leaves the regression baselines unchanged. *Goes away in PR 7
  (`minmax`), which removes the xfail marker.*
- `LinearKernel.k_variable` with ARD (`kernels.py:279-290`) uses `scale.value`
  instead of `scale.variable()`, so the ARD scale is never trained. Covered by
  an xfail test. `LinearKernel` isn't used by the examples. *Goes away in PR 7*:
  kernels are functions traced once, so the bug can't exist. PR 7 removes the xfail
  marker.
- `setup_acq_problem` in `routines/hybrid_bo.py`: when `use_output_gp_as_opt_var=False`, the upper
  starting-point bounds use `problem.u_upper_bounds_acquisition`; this should
  probably be `u_upper_bounds_starting_points`. The two coincide in the
  illustrative example, which hides it. *Goes away in PR 8* (start bounds live on
  each `Var`).
- `formulation_acq` is validated strictly in `hybrid_bo.py` but with
  `startswith("lcb")` in `setup_acq_problem` (`standard_bo.py`). *Goes away in PR 8 (`EI`/`LCB`).*
- `Config.epsilon_ei_standard` is unused; `ei_gp` uses its own default. *Goes away
  in PR 9.*
- `setup_acq_problem` in `routines/hybrid_bo.py` requires
  `len(indices_u_input_gp) + len(indices_x_input_gp) == problem.n_u`. This looks
  like an artifact of `main.py` building the GP with `problem.n_u` inputs; the
  real requirement is that the GP's `n_inputs` equals the number of GP inputs.
  *Goes away in PR 8* (inputs come from `h_inputs`).
- `Problem.single_evaluate_with_fixed_x_original`: the Latin hypercube for the free states' starting points
  uses `x_fixed.size` dimensions instead of `len(indices_x_free)`. The two are
  equal in the illustrative example. *Goes away in PR 5* (root-finding starts are
  drawn over the unmeasured variables).
- `Config.indices_x_opt` is computed but never used. *Goes away in PR 9.*
- `MultiStartOptimizer.create_lhs_samples` builds `msg`
  when `upper_bounds_starting_points is None` but never raises it. *Goes away in
  PR 6.*
