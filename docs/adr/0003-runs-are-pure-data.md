# Runs are pure data

A `Run` holds only arrays, numbers, strings and plain dataclasses: the history, one
fitted surrogate per iteration, the scenario set, and a text `record` of how the run was
produced (problem structure and equations as text, method settings, seeds, package and
code versions). It never holds the `Problem`, the GP spec or any user callable, because
those contain lambdas that cannot be pickled and would tie saved results to the code
that produced them.

So a fitted surrogate stores hyperparameter values, scalers and training data, and
prediction takes the spec back in: `hb.predict(gp, fitted, X)`. Plots are functions of a
finished run (`hb.plots`), not callbacks during the loop, so slow runs are never
repeated to redraw a figure.

Amended by [ADR 0008](0008-gps-are-data.md): the GP is now data (frozen dataclasses from
a closed set, no callables), so a fitted surrogate holds its whole GP and prediction
takes the fitted GP alone, `hb.gp.predict(fitted, X)`. Runs still hold only data.

## Considered Options

- **Keep the spec in the run and save with `cloudpickle`/`dill`**: saved files would
  depend on the producing code and loading them would run it.
- **Per-iteration plot callbacks** (the code before the restructure): plots needed the
  run to be repeated, and the loop carried plotting arguments.
