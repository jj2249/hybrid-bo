---
name: hybrid-bo-experiments
description: Use when writing, modifying or running an experiment in experiments/, or a script that calls the library's methods. Covers the layout of an experiment, how to define the Problem and the measurement, the method settings, the Benchopt benchmark, saving runs, and common mistakes.
---

# Writing and running experiments

Read `docs/writing_an_experiment.rst` first: it is the user guide (layout, CasADi
rules, `Var`/`System`/`Problem`, oracle vs custom measurement, methods and settings,
running, common errors). `docs/reading_results.rst` covers a run, the iteration figures
and Benchopt's results. `experiments/illustrative_example.py` is the reference
experiment, and `illustrative_example_2d.py` its version with two decision variables;
`rosenbrock.py` (P = 2) and `langermann.py` (P = 5) have no known equations (M = 0).
Use GLOSSARY.md's terms. This file adds only what an agent needs on top.

## Layout contract

An experiment is one file under `experiments/`, importable as a top-level module (pytest
and ty put `experiments/` on the path). At module level, in this order: Problem,
measurement (with a module-level `def oracle(v)` and the stated `optimum`), the BO
methods' prior GP (`gp`), `methods` (each method by its label), then `n_evaluations`
and `n_initial`, every method's budget. The adapter reads `problem`, `measure`,
`optimum`, `methods`, `n_evaluations` and `n_initial` by name.

- A file in `experiments/` is an experiment unless it is `__init__.py` or
  `objective.py` (`benchmark_utils.MODULES`); `tests/test_benchmark.py` lists them.
- Importing the module runs nothing, so `tests/test_regression.py` reuses the
  illustrative example's `problem`, `measure`, `methods` and `n_initial`, and
  `tests/conftest.py`'s `illustrative_problem` fixture is its `problem`. Changing the
  illustrative experiment's problem, measurement or settings moves the regression
  baselines: that is a *logic* change.
- A custom measurement is `(u (N,), rng) -> measured (P,)`; its docstring (else its
  name) describes it in each run's artifact; a dataclass measurement is described by
  its repr.
- A BO method's settings are required, since they name its GP: bind them with
  `functools.partial(hb.hybrid_bo, settings=hb.HybridBOSettings(gp=gp, ...))`, never
  put `hb.hybrid_bo` or `hb.standard_bo` in `methods` bare. `settings` is
  keyword-only, after `rng`.
- Defaults never choose the method (ADR 0009): a default may set how hard or how
  precisely we search, never what we search for. So these are required and spelled
  out in every experiment: `acquisition` (`hb.EI()` or `hb.LCB(std_factor=...)`) on
  both settings, `approximation` on `HybridBOSettings`, `n_scenarios` on `hb.SAA` and
  `hb.SoftSAA` (with `temperature`), and `noise_std` on `hb.oracle_measure` and
  `hb.gp.GP`. Search effort and numerics keep defaults (`n_starts`, `n_starts_gp`,
  the solver, `jitter`, kernel priors, `ard`, `unknown_states_as_variables=True`). The
  settings and choice dataclasses and `GP` are keyword-only:
  `hb.SoftSAA(temperature=5e-3, n_scenarios=25, unknown_states_as_variables=False)`
  (the experiments take a temperature of about 1e-3 times the optimum's magnitude).
  Leaving one out, or passing one by position, raises `TypeError`.
- Define the GP once as a module-level `gp`, each fitted
  hyperparameter its own variable so the experiment shows them all (see
  `illustrative_example.py`): `outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))`,
  `lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))`, then
  `hb.gp.GP(mean=hb.gp.Zero(), kernel=outputscale *
  hb.gp.Matern52(lengthscale=lengthscale), noise_std=0.0)`. The shipped experiments
  share it between methods; each method fits its own GP to its own data.

## Randomness

Run i of every method gets the seed `self.get_seed(use_repetition=True)` in
`solvers/method.py`, so run i of every method starts from the same initial points. The
solver draws `hb.initial_points(system.u, n_initial, rng)` and runs the method for
`n_evaluations - n_initial` iterations, both from that generator. A BO run with a
larger budget passes through the same first evaluations as one with a smaller budget
(the prefix property); the baselines draw their whole design first: Latin hypercube
sampling lacks it, and uniform sampling has it only with a measurement that draws
nothing. In a script, do the same by hand: one generator per run, and draw nothing else
from it.

## Oracle optimum

State it as a module-level `optimum`; nothing computes it (a 10-start search found a
local minimum for the illustrative example, #62). Find it once with
`hb.oracle_optimum(problem, oracle, rng, n_starts=200)`, check it, and say how in a
comment. `hb.check_oracle_optimum(history, optimum)` warns with `OracleOptimumWarning`
if a run evaluates a point whose oracle objective is below it. The benchmark's objective
calls it and also records `below_optimum` in the rows, which the `summary` view counts,
since a cached run isn't scored again and can't warn again.

## Pitfalls

- `math.exp(v.x)` on a CasADi symbol silently returns `nan`; numpy functions mostly
  work but emit a CasADi `FutureWarning`. Always use `cas.*`.
- `System` takes keyword arguments only. Without known equations, leave out both
  `x` and `g` (M = 0); giving one without the other raises `ValueError`.
- Exactly one measured variable per unknown state. With `P > 1` unknown states,
  `plots.iteration` draws hybrid BO only with `gp=False` (else `ValueError`).
- With measurement noise, state it in the GP too, as the same number:
  `GP(..., noise_std=0.01)` for `oracle_measure(..., noise_std=0.01)`. `noise_std`
  is in measured units; kernel hyperparameters (`Param` bounds, fixed lengthscales) are in
  scaled units (inputs onto [0, 1], outputs standardized). Don't convert the noise by
  hand.
- Passing `measure` to `plots.iteration` runs it over the plot grid: only for an oracle.
- The iteration figures need one or two decision variables. They replay each run from
  what it stored (its fitted GPs and its settings), so nothing from the experiment's
  methods is passed and a saved run plots after the methods change.

## Running

    pixi run benchmark -r 3 -j 4 [-d "experiment[module=NAME]"] [-s "method[label=LABEL]"] [--seed 0] [--no-plot --no-html]

The Benchopt classes live beside the experiments, and only they use Benchopt: the
library never imports it. Each uses the base class's `**kwargs` signatures, except
`plot`, whose parameters Benchopt checks against `options`.

- A run is one call of a method (`sampling_strategy = "run_once"`), so
  `Objective.evaluate_result` returns one row per evaluation.
- `Solver.run` saves the run with `hb.save_run` to
  `outputs/runs/<module>/<code>/<label>/seed-<seed>.json`, not Benchopt's run folder,
  which holds only what wasn't cached.
- **`code`, the dataset's parameter, has to stay** (ADR 0013): Benchopt's cache only sees
  its own classes, so without the hash a changed experiment returns old results. It
  must have one value: Benchopt pairs a selected module with each value of another
  parameter, so a hash per module runs each selected module once per hash.
- A repeated command skips the finished runs, so a stopped study finishes that way.
- A `HybridBOError` keeps the partial run (`complete` is 0 in its rows); any other
  exception prints a traceback, the run has no rows, and the exit code stays 0.
- `--no-html` still draws every Benchopt PDF, which takes minutes: pass `--no-plot` too.
  `summary` is a table, so it only appears in the HTML report.
- `benchopt plot . -f <file> -k regret --no-html --no-display` draws one view again.
