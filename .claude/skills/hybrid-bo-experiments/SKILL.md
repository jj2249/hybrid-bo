---
name: hybrid-bo-experiments
description: Use when writing, modifying or running an experiment under experiments/. Covers the layout of an experiment, how to define the Problem and the measurement, the method settings, saving runs, and common mistakes.
---

# Writing and running experiments

Read `docs/writing_an_experiment.rst` first: it is the user guide (layout, CasADi
rules, `Var`/`System`/`Problem`, oracle vs custom measurement, methods and settings,
`cli.main`, running, common errors). `docs/reading_results.rst` covers the saved study,
the figures and the Polars views. `experiments/illustrative_example.py` is the
reference experiment. Use GLOSSARY.md's terms. This file adds only what an agent needs
on top.

## Layout contract

An experiment is one module under `experiments/`, importable as a top-level module
(pytest and ty put `experiments/` on the path). At module level, in this order:
Problem, measurement (with a module-level `def oracle(v)` and the stated `optimum` for
a benchmark), `methods`, `study_settings`, then a `__main__` block calling `cli.main`.

- Everything the command line uses is passed in the `cli.main` call (directly, or
  inside `problem` and `measure`); nothing is read from the module by name.
- Importing the module runs nothing, so `tests/test_regression.py` reuses the
  illustrative example's `problem`, `measure`, `methods` and `study_settings`, and
  `tests/conftest.py`'s `illustrative_problem` fixture is its `problem`. Changing the
  illustrative experiment's problem, measurement or settings moves the regression
  baselines: that is a *logic* change.
- A custom measurement is `(u (N,), rng) -> measured (P,)`; its docstring (else its
  name) is its record.
- A BO method with its default settings goes into `methods` bare (`hb.standard_bo`);
  bind other settings with `functools.partial(hb.hybrid_bo, settings=...)`.
  `settings` is keyword-only, after `rng`.

## Randomness

Every draw of a study comes from `settings.seed`. Run `i` of every method gets the
`i`th child of the seed, so run `i` of each method starts from the same initial points.
Within `hb.repeat`, run `i` draws `hb.initial_points(system.u, n_initial, rng)` and
runs the method, both from that generator. Outside a study, do the same by hand:
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
- Exactly one measured variable per unknown state, and only `P == 1` is supported.
- With `noise_std > 0`, the GP needs a `noise_std` hyperparameter (`default_gp` is
  noise-free).
- `true_curves=True` runs the measurement over the plot grid: only for an oracle.
- The iteration figures need one decision variable and read the settings from the
  run's record, so a custom GP needs `settings=` in `hb.plots.iteration`; otherwise
  `plot` prints why and skips that run's figures.

## Running

    pixi run experiment <name> [--method M ...] [--n-runs 10] [--seed 0] [--n-iterations 5] [--name ID] [--plot]
    pixi run plot <name> [STUDY]
    pixi run experiment <name> plot [STUDY] [--results-root DIR]

The tasks run `python -m <name>` from `experiments/` in the `experiments` environment.
No command lists every experiment, and a typo in the name fails with "No module named
...". A relative `--results-root` is relative to where pixi was started (`INIT_CWD`),
not `experiments/`. The study id is `--name`, else the UTC time to the second at the
start, with `-2`, `-3`, ... appended if taken when saved (`hb.save` creates the folder,
so parallel runs never share one). `plot` loads the study given, else the one whose
`study.json` was written last, and saves `summary.csv` if the study has none (studies
saved before summaries).
