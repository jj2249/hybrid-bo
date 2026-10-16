# Changelog

## 0.2.0 (2026-10-16)

The first release after rewriting the original code (tag `pre-refactor`). The
reasons for the rewrite are in [DESIGN.md](DESIGN.md).

- A problem is `hb.Var`s, an `hb.System` of known and unknown equations and an
  `hb.Problem`; a measurement (an `hb.Measure`, or `hb.oracle_measure` to simulate
  one) returns the measured variables.
- Four methods with default settings: `hb.hybrid_bo`, `hb.standard_bo`,
  `hb.latin_hypercube_sampling` and `hb.uniform_sampling`. Each returns an `hb.Run`,
  which is plain data.
- `hb.study` runs every method several times from shared initial points; studies are
  saved and loaded as JSON (`hb.save`, `hb.load`).
- Polars views (`hb.evaluations`, `hb.summary`, `hb.hyperparameters`) and static plots
  (`hb.plots`) of a study.
- Experiments are Python modules in `experiments/`, run with
  `pixi run experiment <name>` and plotted with `pixi run plot <name>`.
- User docs: Getting started, Writing an experiment and Reading results.
- Results pickled by the old code can't be loaded; regenerate them with the new API.
