---
name: hybrid-bo-examples
description: Use when writing, modifying or running an example under examples/. Covers the layout of an example script, how to define the Problem and the experiment, the method settings, saving runs, and common mistakes.
---

# Writing and running examples

`examples/illustrative_example.py` is the reference: one Python module, in the style of
DESIGN.md's "Target example". `examples/flash_unit/` still targets the `pre-refactor`
tag and is out of scope until its port (#42), so leave it alone. Use CONTEXT.md's terms
(decision variables `u`, mechanistic states `x`, unknown states `y`, known equations
`g`, oracle, scenarios).

## Layout of an example script

An example is one module under `examples/`, importable as a top-level module (pytest
and ty put `examples/` on the path). At module level it defines, in this order:

1. **Problem**: `Var`s, `System`, `Problem` (below).
2. **Experiment**: for a test case, a module-level `def oracle(v)`,
   `oracle_experiment(system, oracle, noise_std)` and the stated oracle `optimum`;
   otherwise a custom `(u (N,), rng) -> measured (P,)` function.
3. **Methods**: `methods`, a dict from method label to a callable
   `(problem, experiment, initial_points, n, rng) -> Run`. Bind settings with
   `functools.partial(hb.hybrid_bo, settings=hb.HybridBOSettings(...))`; the
   baselines need no settings (`n` is their number of design points). The plots read
   a run's settings from its record.
4. **The study**, the defaults the flags override: `study_settings =
   hb.StudySettings(n_runs, n_initial, n_iterations, seed=None)` (`None` draws a
   fresh seed, which the saved study's settings keep).
5. **The command line**: a `__main__` block that passes everything above to
   `cli.main` (`import cli`, from `examples/cli.py`):

   ```python
   if __name__ == "__main__":
       cli.main(
           "illustrative_example",
           problem,
           experiment,
           methods,
           study_settings,
           optimum=optimum,
           true_curves=True,
       )
   ```

   The first argument names the study and its results folder; use the module's
   name. `optimum` is the stated oracle optimum, which only the study sees (regret);
   the methods never do. `true_curves=True` makes the iteration figures evaluate the
   experiment over the grid, so set it only for a cheap experiment (an oracle).

Everything the command line uses is passed in that call (directly, or inside
`problem` and `experiment`): nothing is read from the module by name. `cli.main` runs `hb.study(methods, problem, experiment, study_settings,
name=..., optimum=...)` and saves the study with `hb.save` to
`results/<example>/<study id>/` (`study.json`, `runs/<method>/run-000.json`;
git-ignored). The library never writes files on its own; saving is an explicit call.
Importing the module runs nothing (the call is under `__main__`), so
`tests/test_regression.py` reuses its `problem`, `experiment`, `methods` and
`study_settings`, and `tests/conftest.py`'s `illustrative_problem` fixture is its `problem`. Changing the illustrative example's
problem, experiment or settings therefore moves the regression baselines: that is a
*logic* change.

**Randomness.** Every draw of a study comes from `settings.seed`: the same settings
give the same study. Run `i` of every method gets the same generator, the `i`th
child of the seed, so run `i` of each method starts from the same initial points.
Within `hb.repeat`, run `i` draws `hb.initial_points(system.u, n_initial, rng)` and
runs the method, both from that generator. Outside a study, do the same by hand:
`np.random.default_rng(seed).spawn(n_runs)`, and draw nothing else from a run's
generator.

**Failures.** A `HybridBOError` raised during a method carries the partial run as
`error.run`; `hb.repeat` logs the error and keeps the partial run in the study.

**Oracle optimum.** A test case states it as a module-level `optimum`, which
`hb.study` records and regret is measured from. The study never computes it: a
10-start search found a local minimum for the illustrative example (#62). Find it
once with `hb.oracle_optimum(problem, oracle, rng, n_starts=200)`, check it, and say
how in a comment. `hb.study` warns with `OracleOptimumWarning` if a run evaluates a
point whose oracle objective is below it.

## Defining the problem

```python
u = hb.Var("u", lower=-2.0, upper=2.0)  # start bounds default to finite bounds
x = hb.Var("x", start=(-2.5, 1.0))  # unbounded: start bounds required
y = hb.Var("y", start=(-2.0, 2.0))
system = hb.System(
    u=[u],
    x=[x],
    y=[y],
    g=lambda v: [v.x + cas.exp(v.x) - v.y],  # 0 = g(x, y, u), one entry per x
    h_inputs=[u],  # arguments of y = h(.), from u and x
    measured=[y],  # what the sensors read, from x and y
)
problem = hb.Problem(system, f=lambda v: (6 * v.x - 2) ** 2 * cas.sin(12 * v.x - 4))
```

- `g`, `f` and the oracle are callables over a namespace `v` of CasADi symbols, one per
  variable name; use `cas` functions (`cas.exp`, `cas.sin`), not numpy.
- Bounds are the boxes of the acquisition problem and may be infinite. Start bounds
  must be finite: they are the ranges of initial points, solver starts and
  root-finding starts.
- The oracle returns one expression per unknown state: `oracle=lambda v: [cas.sin(v.u)]`.

## Common mistakes

- **Measured variables**: exactly one per unknown state (`len(measured) == P`), so the
  root-finding problem is square. Only `P == 1` is supported.
- **Noise**: with `noise_std > 0`, give the GP a `noise_std` hyperparameter
  (`hb.gp.Hyper("noise_std", ...)` in a custom `hb.gp.GP`); `default_gp` is noise-free.
- **Ambiguous reconstruction**: a `ReconstructionWarning` means the measured variables
  don't pin the states to the oracle's root. Narrow the unmeasured start bounds to the
  right branch, or measure other variables.

## Running

    pixi run example illustrative_example [--method hybrid_bo ...] [--n-runs 10] [--seed 0] [--n-iterations 5] [--name lcb-vs-ei] [--plot]
    pixi run plot illustrative_example [STUDY]
    pixi run example illustrative_example --help   # the flags and the methods
    pixi run example illustrative_example plot [STUDY] [--results-root DIR]

The pixi tasks run the example script as a module from `examples/`, in the
`examples` environment: `example` is `python -m <name>` and passes every flag
through; `plot` is `python -m <name> plot [STUDY]` and takes nothing else, so plot
with other flags through `example` (last line above). No command lists every
example, and a typo in the name fails with "No module named ...". A relative
`--results-root` is relative to where pixi was started (`INIT_CWD`), not
`examples/`. Running (the default command) runs a study from the seed (see Randomness), with
the methods given and the example's `study_settings` overridden by the flags given,
and saves it to `<study id>`: `--name`, else the UTC time to the second at the start,
with `-2`, `-3`, ... appended if taken when it is saved (`hb.save` creates the
folder, so parallel runs never share one), then prints `hb.summary(study)` and saves
it as `summary.csv` beside `study.json`. `plot` loads the study given, else the
one whose `study.json` was written last, saves `summary.csv` if the study has none
(studies saved before summaries), and, in `hb.plots.STYLE`, saves into its
`figures/` folder `convergence.png`, `regret.png` (with an oracle optimum) and
`<method>/run-000/iteration-00.png` for every iteration of every run (with the
experiment's true curves when the script passes `true_curves=True`). The iteration
figures need one decision variable and read the settings from the run's record (a
custom GP needs `settings=`); otherwise `plot` prints why and skips that run's.
