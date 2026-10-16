# hybrid-bo

Bayesian optimization for partially known systems using hybrid models
([arXiv 2603.11199](https://arxiv.org/abs/2603.11199)). The objective is
`f(x, y, u)` subject to known equations `0 = g(x, y, u)` and unknown equations
`y = h(x, u)`, which a GP learns. The acquisition problem is solved over scenarios
(sample-average approximation) using CasADi. Standard BO, Latin hypercube sampling
and uniform sampling are included as baselines. The vocabulary is in
[CONTEXT.md](CONTEXT.md); use its terms.

Maintainers working on the library (issues, PRs, the roadmap) follow
[CONTRIBUTING.md](CONTRIBUTING.md); read it before starting a ticket.

## Commands

Environments are managed with pixi. `dev` includes everything.

```bash
pixi install -e dev
pixi run -e dev check        # lint (with --fix) + format + typecheck
pixi run -e dev test         # pytest (~6s)
pixi run -e dev test --update-baselines   # regenerate regression results
pixi run -e dev coverage     # tests + per-file line coverage of src/hybrid_bo
pixi run -e dev lint | format | typecheck
pixi run -e dev docs         # sphinx API docs -> docs/_build/html
pixi run -e dev setup        # install pre-commit hooks
pixi run -e quality pre-commit run --hook-stage manual ty   # ty hook is manual for now
pixi run experiment illustrative_example [--n-runs 3 --plot ...]   # every method; --help lists the flags
pixi run plot illustrative_example [STUDY]   # figures of the newest (or named) study
```

The pixi tasks run an experiment as a module from `experiments/` (`python -m
illustrative_example ...`). The experiment ends with an explicit `cli.main(...)`
call that passes everything it defines; `experiments/cli.py` parses the flags and
runs or plots. The tasks set `default-environment = "experiments"`, so they need no
`-e`. `experiment` passes every flag through; `plot` takes only the study id, so plot with
other flags as `pixi run experiment illustrative_example plot [STUDY] [--results-root ...]`.

The git hook hardcodes the pixi Python path. If commits fail with "`pre-commit`
not found" after a Python upgrade, re-run `pixi run -e quality setup`.

`tests/test_regression.py` runs each method of the illustrative example
(seed 0, 2 iterations) and compares the results with `tests/data/regression/*.json`.
Only update the baselines when a change is meant to change behavior.
The other test files check kernels, the GP, `Problem` and the scalers
against numpy/scipy references. Known bugs are `xfail(strict=True)` tests:
remove the marker when fixing the bug.

## Skills

- `hybrid-bo-src` (`.claude/skills/`): what the library computes, the module
  map, data and shape conventions, surprising behavior and the testing rules.
  Load it before working on `src/`.
- `hybrid-bo-experiments`: the layout of an experiment, how to define the
  Problem and the measurement, method settings, saving runs and common mistakes.
- Both skills describe the *current* code; DESIGN.md describes the target.

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
- Library code lives at the top level of `hybrid_bo`.
- Functions take what they use. Randomness is an explicit `rng`.
- Rows are points: a point is `(n,)`, a set is `(n_points, n)`, `f` is
  `(n_points,)`. Column vectors only in CasADi glue code.
- The library never writes files on its own; saving is an explicit call (`hb.save`).
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
- Tests follow the same comment rules, so they read without pytest experience: each
  test gets a docstring saying what behavior it checks, and a longer test is split
  into commented blocks (setup, act, check). The first use in a file of a pytest
  feature (fixture, `parametrize`, `raises`, `xfail`) gets a short comment saying
  what it does.
- Defaults may be derived from other arguments when the derivation is obvious and
  recorded in the run (start bounds from finite bounds, GP size from its inputs);
  otherwise arguments are explicit. Examples should be short to write.
