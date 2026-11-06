# Changelog

## Unreleased

- **Breaking:** The library is stripped down: it fits GPs and solves one iteration, and
  you write the loop (see README.md). Gone: `hb.hybrid_bo`, `hb.standard_bo`, the
  baselines, `Run`, settings, the artifact, the oracle tools, `optimize_hybrid_model`,
  `LCB`, `EI`, `Linearization`, the iteration figures, and the guides. The last of
  the old code is tagged `pre-strip`, and `v0.6.0` is the last release with it.
- **Breaking:** `hb.History` is now `hb.Dataset`, with `hb.save_dataset`,
  `hb.load_dataset` and `hb.extend`. `Param` takes its bounds alone, and `fit_gp` raises
  `NoSolutionError` when every start fails.
