---
name: hybrid-bo-experiments
description: Use when writing, modifying or running an experiment in experiments/, or a script that calls the library's methods. Covers the layout of an experiment, how to define the Problem and the measurement, the methods, saving runs, and common mistakes.
---

# Writing and running experiments

Read `docs/writing_an_experiment.rst` first: it is the user guide (layout, CasADi
rules, `Var`/`System`/`Problem`, oracle vs custom measurement, methods and settings,
running, common errors). `docs/reading_results.rst` covers a run, saving and the figures. `experiments/illustrative_example.py` is the reference
experiment, and `illustrative_example_2d.py` its version with two decision variables;
`rosenbrock.py` (P = 2) and `langermann.py` (P = 5) have no known equations (M = 0).
Use GLOSSARY.md's terms. This file adds only what an agent needs on top.

## Layout contract

An experiment is one plain script under `experiments/`, importable as a top-level module
(pytest and ty put `experiments/` on the path). At module level, in this order: Problem,
measurement (with a module-level `def oracle(v)` and the stated `optimum`), the BO
methods' prior GP (`gp`), `N_INITIAL`, `N_ITERATIONS` and `N_RUNS`, then a small function
per method, `def hybrid_bo(rng) -> hb.Run`, which calls the library. A `__main__` block
loops over seeds, then calls `hb.save_runs`, makes a figure with `plt.subplots()` and
draws on it with `hb.plots.regret(runs, optimum, ax)`.

- Importing the module runs nothing, so `tests/test_regression.py` reuses the
  illustrative example's `gp`, `problem` and `measure`, and
  `tests/conftest.py`'s `illustrative_problem` fixture is its `problem`. Changing the
  illustrative experiment's problem, measurement or settings moves the regression
  baselines: that is a *logic* change.
- A custom measurement is `(u (N,), rng) -> measured (P,)`; its docstring (else its
  name) describes it in each run's artifact; a dataclass measurement is described by
  its repr.
- Defaults never choose the method (ADR 0009): a default may set how hard or how
  precisely we search, never what we search for. So these are required and spelled
  out in every experiment: `gp` and `acquisition` (`hb.EI()` or
  `hb.LCB(std_factor=...)`) on both BO methods, `approximation` on `hb.hybrid_bo`, `n_scenarios` on `hb.SAA` and
  `hb.SoftSAA` (with `temperature`), and `noise_std` on `hb.oracle_measure` and
  `hb.gp.GP`. Search effort and numerics keep defaults (`n_starts`, `n_starts_gp`,
  the solver, `jitter`, kernel priors, `ard`, `unknown_states_as_variables=True`). The
  methods, the choice dataclasses and `GP` are keyword-only:
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

`rng` is one generator per run. With `n_initial` the method draws its initial points from
it first, so the same seed gives every method the same initial points; `initial_points=`
gives them instead. A BO run with a larger budget passes through the same first
evaluations as one with a smaller budget (the prefix property); the baselines draw their
whole design first: Latin hypercube sampling lacks it, and uniform sampling has it only
with a measurement that draws nothing. Draw nothing else from a run's generator.

## Oracle optimum

State it as a module-level `optimum`; nothing computes it (a 10-start search found a
local minimum for the illustrative example, #62). Find it once with
`hb.oracle_optimum(problem, oracle, rng, n_starts=200)`, check it, and say how in a
comment. `hb.check_oracle_optimum(history, optimum)` warns with `OracleOptimumWarning`
if a run evaluates a point whose oracle objective is below it.

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

    pixi run -e experiments python experiments/<experiment>.py <name>

Saving goes to `results/<experiment>/<name>/` (git-ignored). The script makes that folder
before its runs, so a name in use fails at once, and `hb.save_runs` takes an empty folder
and refuses any other. A failed run stops the script unless it catches `hb.HybridBOError`, whose `run`
holds the partial run. For parallel runs, `joblib` works with functions defined in a
script, with results identical to serial runs.
