# hybrid-bo

Bayesian optimization for partially known systems using hybrid models
([arXiv 2603.11199](https://arxiv.org/abs/2603.11199)). The objective is
`f(x, y, u)` subject to known equations `0 = g(x, y, u)` and unknown equations
`y = h(x, u)`, which a GP learns. The acquisition problem is solved over scenarios
(sample-average approximation) using CasADi. Standard BO, Latin hypercube sampling
and uniform sampling are included as baselines. The vocabulary is in
[CONTEXT.md](CONTEXT.md); use its terms. (The code before the restructure still uses
`h_known`/`h_unknown` and stores `y` inside `x`.)

## Goal of the refactor

Make the code easier to read, change and extend, so the library can scale. More
advanced constructs are fine when they make things clearer at the point of use,
e.g. grouping related arguments (`f(x, x_bounds)` instead of
`f(x, x_lower, x_upper)`). Avoid complexity that doesn't serve that, such as
abstractions with a single use or boilerplate from static languages.
The `flash_unit` example is out of scope for the refactor; leave it alone.

The roadmap is in [ROADMAP.md](ROADMAP.md). Check it before starting work, and
tick items off in the PR that completes them. The target API and the plan for
each restructure PR are in [DESIGN.md](DESIGN.md), and the decisions behind it in
[docs/adr/](docs/adr/). Read them before any restructure PR, and update DESIGN.md
when a PR changes the design.

## Commands

Environments are managed with pixi. `dev` includes everything.

```bash
pixi install -e dev
pixi run -e dev check        # lint (with --fix) + format + typecheck
pixi run -e dev test         # pytest (~2s)
pixi run -e dev test --update-baselines   # regenerate regression results
pixi run -e dev coverage     # tests + per-file line coverage of src/hybrid_bo
pixi run -e dev lint | format | typecheck
pixi run -e dev docs         # sphinx API docs -> docs/_build/html
pixi run -e dev setup        # install pre-commit hooks
pixi run -e quality pre-commit run --hook-stage manual ty   # ty hook is manual for now
pixi run -e dev python examples/main.py   # example and routine are hardcoded at the top of main.py
```

The git hook hardcodes the pixi Python path. If commits fail with "`pre-commit`
not found" after a Python upgrade, re-run `pixi run -e quality setup`.

`tests/test_regression.py` runs each routine on the illustrative example
(seed 0, 2 iterations) and compares the results with `tests/data/regression/*.json`.
Only update the baselines in a PR that is meant to change behavior, and explain
the JSON diff in that PR.
The other test files check kernels, the GP, `Problem` and the transformers
against numpy/scipy references. Known bugs are `xfail(strict=True)` tests:
remove the marker in the PR that fixes the bug.

## Skills

- `hybrid-bo-src` (`.claude/skills/`): what the library computes, the module
  map, data and shape conventions, surprising behavior and the testing rules.
  Load it before working on `src/`.
- `hybrid-bo-examples`: the files and exports an example needs, how to define
  the Problem, the config fields and common mistakes.
- Both skills describe the *current* code; DESIGN.md describes the target. When a
  PR changes behavior or contracts, update the relevant skill in the same PR.

## Workflow

- `refactor/initial` is the working branch. `main` stays frozen until the roadmap
  is complete. Never commit to or open PRs against `main`.
- One PR per issue into `refactor/initial`. Branch from it using the name
  `<stage>-<topic>` (e.g. `s1-hooks`).
- Agree scope in plan mode before editing.
- Before opening a PR, run `pixi run -e dev check` and `pixi run -e dev test`,
  then `/mattpocock-skills:code-review` since `refactor/initial`.
- Use the PR template. Add the `ai-assisted` label to PRs with AI-written code.
  Commits get a `Co-Authored-By` trailer.
- Keep non-logic and logic changes in separate PRs.
- Keep docs concise. Sphinx autoapi already covers the API, so don't duplicate it.
- When a style question comes up, ask, then record the answer under Conventions.

## Conventions

<!-- Grown incrementally: record decisions here as they are made. -->

- Data in, data out: functions return new values and don't mutate their inputs.
  No set-then-call sequences (`set_problem` → `solve()`).
- Frozen dataclasses for data and settings; behavior lives in plain functions.
  Methods are settings dataclasses plus functions, kernels are functions, and
  there are no strategy or kernel classes. Add a protocol only once a second
  implementation exists.
- American spelling everywhere, in code and prose (`optimization_loop`,
  "optimization loop").
- New code lives at the top level of `hybrid_bo` and never imports
  `hybrid_bo.legacy` (the old code, deleted in DESIGN.md PR 9). Legacy code may
  import new code (e.g. `lhs`).
- Functions take what they use, not a `Config`. Randomness is an explicit `rng`.
- Rows are points: a point is `(n,)`, a set is `(n_points, n)`, `f` is
  `(n_points,)`. Column vectors only in CasADi glue code.
- The library returns results; examples save them.
- Errors raise specific exceptions; the caller decides whether to continue.
- Annotate signatures and dataclass fields, not local variables. Type errors are
  fixed, not suppressed; if a suppression is unavoidable, use
  `# ty: ignore[rule]` (ty doesn't read `pyright:` comments).
- Comments: every function and class gets a docstring saying what it does (and what
  it raises, if anything); a short function needs nothing more. A longer function
  is split into blocks by blank lines, each led by a one-line comment saying what
  the block does or why (e.g. `# Start bounds: default to the bounds, ...`). Pull a
  long condition into a named variable rather than explaining it. Don't comment
  what a line obviously does. See `src/hybrid_bo/problem.py`.
- Defaults may be derived from other arguments when the derivation is obvious and
  recorded in the run (start bounds from finite bounds, GP size from its inputs);
  otherwise arguments are explicit. Examples should be short to write.

## Agent skills

### Issue tracker

GitHub Issues on `jj2249/hybrid-bo` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
