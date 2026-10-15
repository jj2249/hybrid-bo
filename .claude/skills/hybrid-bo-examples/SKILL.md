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
2. **Experiment**: `oracle_experiment(system, oracle, noise_std)` for a test case, or a
   custom `(u (N,), rng) -> measured (P,)` function.
3. **Methods**: `settings`, a dict from BO method name to its settings (the plots
   need them), and `methods`, a dict from method name to a callable
   `(problem, experiment, initial_points, n, rng) -> Run`. Bind settings with
   `functools.partial(hb.hybrid_bo, settings=settings["hybrid_bo"])`; the baselines
   need no settings (`n` is their number of design points).
4. **The study**, the CLI's defaults: `study_settings = hb.StudySettings(n_runs,
   n_initial, n_iterations, seed=None)` (`None` draws a fresh seed, which the CLI
   saves in the settings and every run records).

The module has no `__main__` block: the CLI (`examples/main.py`) runs it as
`hb.study(methods, problem, experiment, settings, np.random.default_rng(seed))` and
saves the study with `hb.save` to `examples/results/<example>/<UTC time>/`
(`study.json`, `runs/<method>/run-000.json`; git-ignored). The library never writes
files on its own; saving is an explicit call. Importing the module
runs nothing, so `tests/test_regression.py` reuses its `problem`, `experiment` and
`methods`, `tests/test_plots.py` its `settings`, and `tests/conftest.py`'s
`illustrative_problem` fixture is its `problem`. Changing the illustrative example's
problem, experiment or settings therefore moves the regression baselines: that is a
*logic* change.

**Randomness.** `hb.study` gives every method the same generators spawned from its
`rng`, so run `i` of each method starts from the same initial points. Within
`hb.repeat`, run `i` draws `hb.initial_points(system.u, n_initial, rng)` and runs the
method, both from the `i`th child of `rng`. Outside a study, do the same by hand:
`rng.spawn(n_runs)`, and draw nothing else from a run's generator.

**Failures.** A `HybridBOError` raised during a method carries the partial run as
`error.run`; `hb.repeat` logs the error and keeps the partial run in the study.

**Oracle optimum.** With an oracle experiment, `hb.study` computes the oracle optimum
(`hb.oracle_optimum`) and records it in `study.record`. If the test case knows it,
pass `optimum=` instead; it is recorded as stated.

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

    pixi run -e dev python examples/main.py run illustrative_example [--method hybrid_bo ...] [--n-runs 10] [--seed 0] [--n-iterations 5]
    pixi run -e dev python examples/main.py plot examples/results/illustrative_example/<UTC time> [--run 0] [--iteration -1] [--show]

`run` runs a study from the seed (see Randomness), with the methods given and the
`study_settings` overridden by the flags given, and saves it. `plot` loads the study
with `hb.load`, imports the example named by the results folder's parent, then, in
`hb.plots.STYLE`, saves `convergence.png` and `<method>.png`
(`hb.plots.iterations` with `settings.get(method)` and the example's experiment,
showing `--iteration`) into the folder; `--show` opens them with their - and + buttons. The figures
need one decision variable; for other problems `plot` prints why and skips them.
