# Changelog

## Unreleased

- **Breaking:** `hb.hybrid_trial_point` and `hb.standard_trial_point` are gone: build with
  `hb.hybrid_acquisition` or `hb.standard_acquisition` and call `.solve(rng)`, and pass the
  same object to `hb.plots.iteration`.
- **Breaking:** `hb.saa_scenarios` and `Scenarios` are gone: pass draws `z` and weights `w`.
  `approximation` is keyword-only, and `AcquisitionSolution` is `Solution`.
- `hb.expected_improvement` is public.
- **Breaking:** `hb.fit_gp(gp, X, Y, rng)` is `gp.fit(X, Y, rng)`, `predict(fitted, X)` is
  `fitted.predict(X)`, and `log_marginal_likelihood` is a method of `FittedGP`.
- **Breaking:** `2.0 * kernel` is gone: write `Scale(kernel, 2.0)`. `covariance` and
  `prior_mean` are the methods `kernel.covariance(X1, X2)` and `mean.at(X)`.
- **Breaking:** A fit raises `ValueError` for shapes that don't match and for NaN or
  infinity, and `predict` for a numeric X that isn't a 2-D array.
- **Breaking:** `HybridBOError` is gone; catch `NoSolutionError` or `RootFindingError`.
- **Breaking:** The scalers `hb.gp.minmax`, `standard`, `scale` and `unscale` are in
  `hybrid_bo.utils`, and `minmax` maps onto [0, 1] only. `hybrid_bo.sampling` is gone.
- New: `plt.style.use("hybrid_bo.default")` gives the library's look: a grid, so far.

## 0.7.0 (2026-11-06)

The library is stripped down: it fits GPs with CasADi and solves one iteration of hybrid
or standard BO, and you write the loop and the measurement (see README.md). Results don't
change. The last of the old code is tagged `pre-strip`, and `v0.6.0` is the last release
with it; saved runs from before this have no loader.

- **Breaking:** Gone: `hb.hybrid_bo`, `hb.standard_bo`, the baselines, `Run`, the settings,
  the run format, the artifact, `hb.oracle_measure` and the oracle tools,
  `hb.optimize_hybrid_model`, `LCB`, `Linearization`, the `EI` class and `acquisition=`,
  `n_scenarios`, the old `hb.plots.iteration` and its 2D figure, and the guides.
- **Breaking:** `hb.History` is now `hb.Dataset`, with no oracle fields and no `n_initial`.
  `hb.reconstruct` is `hb.reconstruct_states`, and `hb.initial_points` is `hb.lhs_points`.
- **Breaking:** `Param` takes its bounds alone, and `hb.fit_gp` raises `NoSolutionError`
  when every start fails.
- **Breaking:** `hb.plots.regret` and `hb.plots.convergence` take incumbent curves per
  label, plain arrays.
- New: `hb.hybrid_trial_point` and `hb.standard_trial_point` solve one iteration, and
  `hb.extend`, `hb.save_dataset`, `hb.load_dataset` and `hb.uniform_points` go with them.
- New: `hb.replay` evaluates an acquisition problem over a grid of `u`, solving the
  scenario states by Newton's method, and `hb.plots.iteration` and `hb.plots.surrogate`
  draw on the axes you give them.
- New: the library logs what it fits and how many starts of each solve succeed, at `INFO`
  on the `hybrid_bo` logger.
- `gp.py` matches on the kernel to find its hyperparameters, with the same results.
  Inputs of different widths and a lengthscale of the wrong length raise `ValueError`.
