---
name: hybrid-bo-examples
description: Use when writing, modifying or running an example under examples/. Covers the layout of an example script, how to define the Problem and the experiment, the method settings, saving runs, and common mistakes.
---

# Writing and running examples

`examples/illustrative_example.py` is the reference: one Python script, in the style of
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
3. **Methods**: `methods`, a dict from method name to a callable
   `(problem, experiment, initial_points, n, rng) -> Run`. Bind settings with
   `functools.partial(hb.hybrid_bo, settings=hb.HybridBOSettings(...))`; the baselines
   need no settings (`n` is their number of design points).
4. **Runs**: `seed` (`None` draws a fresh one, recorded in every run), `n_runs`,
   `n_initial`, `n_iterations`.

Under `if __name__ == "__main__":` it runs every method `n_runs` times and pickles
each method's `list[Run]` to `examples/results/<example>/<UTC time>/<method>.pkl`
(git-ignored). The library returns runs; the script saves them. Importing the module
runs nothing, so `tests/test_regression.py` reuses its `problem`, `experiment` and
`methods`, and `tests/conftest.py`'s `illustrative_problem` fixture is its `problem`.
Changing the illustrative example's problem, experiment or settings therefore moves
the regression baselines: that is a *logic* change.

**Randomness.** One generator per run, spawned from one seed that every method shares,
so run `i` of each method starts from the same initial points:
`np.random.default_rng(entropy).spawn(n_runs)`, then
`hb.initial_points(system.u, n_initial, rng)` and the method, both from that run's
`rng`. Draw nothing else from it.

**Failures.** A `HybridBOError` raised during a method carries the partial run as
`error.run`; save it with the others.

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
  (`hb.Hyper("noise_std", ...)` in a custom `hb.GP`); `default_gp` is noise-free.
- **Ambiguous reconstruction**: a `ReconstructionWarning` means the measured variables
  don't pin the states to the oracle's root. Narrow the unmeasured start bounds to the
  right branch, or measure other variables.

## Running

    pixi run -e dev python examples/illustrative_example.py

`examples/main.py` runs the example named at its top the same way (a CLI comes in #31).
There are no plots until `hb.plots` (#31); a `Run` holds everything they need.
