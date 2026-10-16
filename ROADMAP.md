# Roadmap

Each item should become one small PR into `refactor/initial`. Tick items off in
the PR that completes them. `main` stays frozen until every stage is done.

The aim throughout is code that is easier to read, change and extend, so the
library can scale (see the goal in CONTRIBUTING.md). `examples/flash_unit` is out of scope.
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
- [x] 2 *logic*: draw order from DESIGN.md; always reconstruct after the experiment;
      regenerate the baselines
- [x] 3: move the old code to `hybrid_bo/legacy/`
- [x] 4: `Var`, `System`, `Problem`

After PR 4 the work is GitHub issues sliced by method; each method ticket reproduces
its own regression baseline on the new API. Scope is in the issues and DESIGN.md
"Restructure PRs".
- [x] #24: workflow (PR template, CLAUDE.md, this list)
- [x] #39: PR bodies in the `pr` skill format (template, CLAUDE.md)
- [x] #25: evaluation (the ruff `TRY002`/`BLE001` ignores now cover only `legacy/`
      and `examples/`; since #30 only `examples/flash_unit/`)
- [x] #26: Latin hypercube and uniform baselines
- [x] #27: solvers and GP (removes the `minmax` intercept, `LinearKernel` ARD and
      `create_lhs_samples` Known issues)
- [x] #28: standard BO
- [x] #29: hybrid BO (removes the two `setup_acq_problem` Known issues)
- [x] #30: switch the illustrative example and delete the old code (removes the
      `epsilon_ei_standard` and `indices_x_opt` Known issues)
- [x] #31: plots and CLI
- [x] #32 *logic*: benchmark metric
- [x] #41: the tests that survive #30 follow the test comment rules
- [x] #47: a walkthrough of the illustrative example in the Sphinx docs (figures
      from `hb.plots` are a follow-up)

## Stage 5: examples
- [x] Decide whether YAML is the right config format: no, examples are Python
      (DESIGN.md); YAML is deleted in #30
- [x] #52: slim `hb.*` namespace and a Getting started docs page
- [x] #53: `Study`, `StudySettings`, `hb.study`, JSON `hb.save`/`hb.load` (ADR 0004)
- [x] #54: Polars views of a study
- [x] #55: static plots: convergence, regret, iterations (ADR 0005)
- [x] #51: `pixi run example` / `pixi run plot` entry point
- [x] #71: move the maintainer workflow from CLAUDE.md to CONTRIBUTING.md
- [x] #68: rename Experiment to Measurement (`measure`, `oracle_measure`; ADR 0006)
- [ ] #69: `examples/` becomes `experiments/` (task, skill); drop flash_unit's old code
- [ ] #70: student docs: README as entry point, Writing an experiment, Reading results
- [ ] #75: default settings for `hybrid_bo` and `standard_bo` (only require the
      minimum from users)

## Stage 6: merge
- [ ] Lean pixi environments and tasks (`examples` and `dev` are currently identical)
- [ ] Merge `refactor/initial` into `main` (`main` stays frozen until then), then
      make `main` the GitHub default branch again
      (`gh repo edit --default-branch main`)

Longer-term work is tracked as GitHub issues, not here: porting flash_unit (#42) and
publishing to PyPI (#74).

## Known issues (fix only once a test covers them)
Each entry says where it gets fixed. "Goes away" means the new code does not reproduce
it (DESIGN.md: Known issues are not rebuilt), and that PR removes the entry.
- None open.
