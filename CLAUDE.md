# hybrid-bo

Bayesian optimization for partially known systems using hybrid models
([arXiv 2603.11199](https://arxiv.org/abs/2603.11199)). The objective is
`f(x, y, u)` subject to known equations `0 = g(x, y, u)` and unknown equations
`y = h(x, u)`, which a GP learns. The acquisition problem is solved over scenarios
(sample-average approximation) using CasADi. Standard BO, Latin hypercube sampling
and uniform sampling are included as baselines. The vocabulary is in
[GLOSSARY.md](GLOSSARY.md); use its terms.

Maintainers working on the library (issues, PRs) follow
[CONTRIBUTING.md](CONTRIBUTING.md); read it before starting a ticket.

## Commands

Environments are managed with pixi: `default` is the library, `experiments` adds
matplotlib, Polars and the docs, and `dev` includes everything.

```bash
pixi install -e dev
pixi run -e dev check        # lint (with --fix) + format + typecheck
pixi run -e dev test         # pytest (~2 min)
pixi run -e dev test --update-baselines   # regenerate regression results
pixi run -e dev coverage     # tests + per-file line coverage of src/hybrid_bo
pixi run -e dev lint | format | typecheck
pixi run -e dev docs         # sphinx API docs -> docs/_build/html
pixi run -e dev setup        # install pre-commit hooks (ruff and ty on commit)
pixi run experiment illustrative_example [--n-runs 3 --plot ...]   # every method; --help lists the flags
pixi run plot illustrative_example [STUDY]   # figures of the newest (or named) study, iterations of run 0
pixi run experiment illustrative_example plot [STUDY] --plot-runs all   # iteration figures of every run
pixi run experiment illustrative_example resume STUDY [--n-workers 4]   # finish a study that stopped
```

The pixi tasks run an experiment as a module from `experiments/` (`python -m
illustrative_example ...`). The experiment ends with an explicit `cli.main(...)`
call that passes everything it defines; `experiments/cli.py` parses the flags and
runs or plots. A study's runs are tasks, each run in its own process (`python -m
<experiment> run-task`), which rebuilds the experiment from its module and saves its
run file as it finishes; `prepare`, `run-task` and `resume` split a study up (e.g.
for a SLURM array). The `experiment`, `plot` and `docs` tasks set a
`default-environment`, so they need no `-e`. `experiment` passes every flag through;
`plot` takes only the study id, so plot with other flags as
`pixi run experiment illustrative_example plot [STUDY] [--plot-runs ...]
[--results-root ...]`.

The git hook hardcodes the pixi Python path. If commits fail with "`pre-commit`
not found" after a Python upgrade, re-run `pixi run -e dev setup`.
If `ty` fails with "Invalid `VIRTUAL_ENV`" (a shell that has `VIRTUAL_ENV` set to
`.pixi/envs/dev`, which has no `pyvenv.cfg`), run it as `env -u VIRTUAL_ENV pixi run
-e dev check`.

`tests/test_regression.py` runs each method of the illustrative example
(seed 0, 2 iterations) and compares the results with `tests/data/regression/*.json`.
Only update the baselines when a change is meant to change behavior.
The other test files check kernels, the GP, `Problem` and the scalers
against numpy/scipy references. Known bugs are `xfail(strict=True)` tests
with a GitHub issue labeled `bug`: remove the marker when fixing the bug.

## Skills

- `hybrid-bo-src` (`.claude/skills/`): what the library computes, the module
  map, data and shape conventions, surprising behavior and the testing rules.
  Load it before working on `src/`.
- `hybrid-bo-experiments`: the layout of an experiment, how to define the
  Problem and the measurement, method settings, saving runs and common mistakes.
- Both skills describe what the code does; DESIGN.md describes the design and the
  reasons for it.

## Conventions

<!-- Grown incrementally: record decisions here as they are made. -->

- Data in, data out: functions return new values and don't mutate their inputs.
  No set-then-call sequences (`set_problem` → `solve()`).
- Frozen dataclasses for data and settings; behavior lives in plain functions.
  Methods are settings dataclasses plus functions, and there are no strategy
  classes. Kernels and means are frozen dataclasses from a closed set, with their
  behavior in functions that match on the type (`covariance`; ADR 0008). An
  operator may only build data (`Param(...) * Matern52()` is a `Scale`). Add a
  protocol only once a second implementation exists.
  The exception is a value the user passes where a function is expected, which may
  be a frozen dataclass with `__call__` when it is data that the run's artifact
  describes by its repr (the oracle simulation, `OracleSimulation`; #141): users
  call and write it as a function.
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
  otherwise arguments are explicit. Experiments should be short to write.
- A default may set how hard or how precisely we search, never what we search for
  (ADR 0009): the acquisition function, the approximation, `n_scenarios`,
  `std_factor`, `temperature` and `noise_std` are required. Settings and choice
  dataclasses are keyword-only (`@dataclass(frozen=True, kw_only=True)`).
