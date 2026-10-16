# Changelog

## Unreleased

- **Breaking:** the prior GP and the fitted GP are explicit. `hb.gp.GP` is renamed
  `hb.gp.PriorGP` (`FittedGP` keeps its name), and `hb.gp.log_likelihood` is renamed
  `hb.gp.log_marginal_likelihood`. There are no aliases for the old names (#91).
- A study's artifact (`Study.artifact`, saved as `artifact.json`) replaces
  `Study.record` and `study.json`. `hb.load` still reads studies saved as
  `study.json`.
- The recorded code version ends in `-untracked` when files that aren't ignored were
  never added to git, as it ends in `-dirty` for uncommitted changes, so a study's
  artifact says whether it is reproducible.
- `pixi run experiment` warns when run with uncommitted or untracked files;
  `--require-clean` refuses to run instead.

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
