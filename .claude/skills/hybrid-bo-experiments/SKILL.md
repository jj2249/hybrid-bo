---
name: hybrid-bo-experiments
description: Use when writing, modifying or running an experiment under experiments/. Covers the layout of an experiment, how to define the Problem and the measurement, the method settings, saving runs, and common mistakes.
---

# Writing and running experiments

Read `docs/writing_an_experiment.rst` first: it is the user guide (layout, CasADi
rules, `Var`/`System`/`Problem`, oracle vs custom measurement, methods and settings,
`cli.main`, running, common errors). `docs/reading_results.rst` covers the saved study,
the figures and the Polars views. `experiments/illustrative_example.py` is the
reference experiment, and `illustrative_example_2d.py` its version with two decision
variables; `rosenbrock.py` (P = 2) and `langermann.py` (P = 5) have no
known equations (M = 0). Use GLOSSARY.md's terms. This file adds only what an agent needs
on top.

## Layout contract

An experiment is one module under `experiments/`, importable as a top-level module
(pytest and ty put `experiments/` on the path). At module level, in this order:
Problem, measurement (with a module-level `def oracle(v)` and the stated `optimum` for
a benchmark), the BO methods' prior GP (`gp`), `methods`, `configurations` (each
method as an `hb.Configuration(method=..., n_evaluations=..., n_initial=...)`, the
shipped experiments giving every method the same budget), `study_settings`
(`hb.StudySettings(n_runs=..., seed=...)`), then a `__main__` block calling
`cli.main` with `configurations`.

- Everything the command line uses is passed in the `cli.main` call (directly, or
  inside `problem` and `measure`); nothing is read from the module by name.
- Importing the module runs nothing, so `tests/test_regression.py` reuses the
  illustrative example's `problem`, `measure`, `methods` and `configurations`, and
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
  settings and choice dataclasses, `GP`, `Configuration` and `StudySettings` are
  keyword-only:
  `hb.SoftSAA(temperature=1e-6, n_scenarios=25, unknown_states_as_variables=False)`.
  Leaving one out, or passing one by position, raises `TypeError`.
- Define the GP once as a module-level `gp`, each fitted
  hyperparameter its own variable so the experiment shows them all (see
  `illustrative_example.py`): `outputscale = hb.gp.Param(1.0, bounds=(0.1, 10.0))`,
  `lengthscale = hb.gp.Param(1.0, bounds=(1e-3, 1e2))`, then
  `hb.gp.GP(mean=hb.gp.Zero(), kernel=outputscale *
  hb.gp.Matern52(lengthscale=lengthscale), noise_std=0.0)`. The shipped experiments
  share it between methods; each method fits its own GP to its own data.

## Randomness

Every draw of a study comes from `settings.seed`. Run `i` of every configuration gets
the `i`th child of the seed, so run `i` of each configuration with the same `n_initial`
starts from the same initial points. Within `hb.repeat`, run `i` draws
`hb.initial_points(system.u, n_initial, rng)` and runs the method for `n_evaluations -
n_initial` iterations, both from that generator. A BO run with a larger budget passes
through the same first evaluations as one with a smaller budget (the prefix property),
so a budget sweep needs one configuration per `n_initial`, at the largest budget; the
baselines draw their whole design first: Latin hypercube sampling lacks it, and
uniform sampling has it only with a measurement that draws nothing. Outside a study, do the same by hand:
`np.random.default_rng(seed).spawn(n_runs)`, and draw nothing else from a run's
generator.

## Oracle optimum

State it as a module-level `optimum`; the study never computes it (a 10-start search
found a local minimum for the illustrative example, #62). Find it once with
`hb.oracle_optimum(problem, oracle, rng, n_starts=200)`, check it, and say how in a
comment. `hb.study` warns with `OracleOptimumWarning` if a run evaluates a point whose
oracle objective is below it.

## Pitfalls

- `math.exp(v.x)` on a CasADi symbol silently returns `nan`; numpy functions mostly
  work but emit a CasADi `FutureWarning`. Always use `cas.*`.
- `System` takes keyword arguments only. Without known equations, leave out both
  `x` and `g` (M = 0); giving one without the other raises `ValueError`.
- Exactly one measured variable per unknown state. With `P > 1` unknown states,
  `plots.iteration` draws hybrid BO only with `gp=False` (else `ValueError`), so the
  CLI draws those runs' iteration figures without the GP panels.
- With measurement noise, state it in the GP too, as the same number:
  `GP(..., noise_std=0.01)` for `oracle_measure(..., noise_std=0.01)`. `noise_std`
  is in measured units; kernel hyperparameters (`Param` bounds, fixed lengthscales) are in
  scaled units (inputs onto [0, 1], outputs standardized). Don't convert the noise by
  hand.
- `true_curves=True` runs the measurement over the plot grid: only for an oracle.
- The iteration figures need one or two decision variables; otherwise `plot` prints
  why and skips that run's figures. They are drawn for run 0 of each method unless
  `--plot-runs` names other runs (or `all`). They replay each run from what it stored (its fitted
  GPs and its settings), so nothing from the experiment's methods is passed and a
  study plots after the methods change.
- Studies saved before #119 (GPs as data) or #120 (runs hold their settings) don't
  load: `hb.load` raises `StudyFormatError`. Rerun them.

## Running

    pixi run experiment <name> [--method M ...] [--n-runs 10] [--seed 0] [--name ID] [--plot] [--plot-runs 0 3] [--require-clean]
    pixi run plot <name> [STUDY]
    pixi run experiment <name> plot [STUDY] [--plot-runs all] [--results-root DIR]

`--method` takes configuration labels. There is no `--n-iterations`: budgets belong
to configurations, so change them in the experiment (tests define small test
experiments, as `tests/test_cli.py`'s `small` does).
The tasks run `python -m <name>` from `experiments/` in the `experiments` environment.
No command lists every experiment, and a typo in the name fails with "No module named
...". A relative `--results-root` is relative to where pixi was started (`INIT_CWD`),
not `experiments/`. The study id is `--name`, else the UTC time to the second at the
start. It is reserved before anything runs by creating the study log
`<experiment>/<id>.log` exclusively, so parallel runs never share an id; an id is taken
if that log or its folder exists. A taken `--name` is a usage error; a taken UTC time
gets `-2`, `-3`, ... appended. The study log gets the library's records (DEBUG and
above: every run, iteration, GP fit and multistart with its time) and every warning;
the console shows INFO and above. Once saved, the log moves into the study folder as
`study.log`; a study that doesn't finish leaves `<id>.log` in place, which keeps the id
taken until it is deleted. `plot` loads the study given, else the one whose
artifact was written last (`artifact.json`, or `study.json` in older studies), and
saves `summary.csv` if the study has none (studies saved before summaries). Plotting
draws the study figures `cli.main`'s `figures` names (default convergence and
regret; a sweep over initial fractions passes `hb.plots.budget_heatmap` and its
siblings, file name to function of the study; one that raises `ValueError` is
skipped with a message), and the iteration figures of the runs `--plot-runs`
names (indices or `all`, default `0`, of each method); on the run command
`--plot-runs` implies `--plot`. Running
with uncommitted or untracked files warns that the study won't be reproducible;
`--require-clean` refuses to run instead.
