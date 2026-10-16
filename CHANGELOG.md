# Changelog

## 1.0.0

The first release after rewriting the original code (tag `pre-refactor`). The
reasons for the rewrite are in [DESIGN.md](DESIGN.md).

- A problem is `Var`s, a `System` of known and unknown equations and a `Problem`; a
  `Measure` (`oracle_measure` for simulated experiments) returns the measured
  variables.
- Four methods with default settings: `hybrid_bo`, `standard_bo`,
  `latin_hypercube_sampling` and `uniform_sampling`. Each returns a `Run`, which is
  plain data.
- `hb.study` runs every method several times from shared initial points; studies are
  saved and loaded as JSON (`hb.save`, `hb.load`).
- Polars views (`evaluations`, `summary`, `hyperparameters`) and static plots
  (`hb.plots`) of a study.
- Experiments are Python modules in `experiments/`, run with
  `pixi run experiment <name>` and plotted with `pixi run plot <name>`.
- User docs: Getting started, Writing an experiment and Reading results.
- Results pickled by the old code can't be loaded; regenerate them with the new API.
