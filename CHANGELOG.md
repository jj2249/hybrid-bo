# Changelog

## 0.8.1 (2026-11-06)

A hybrid model can now be optimized without BO, the solvers hold their own behavior, and
`solve` returns plain values. The regression results don't change.

- **Breaking:** `hb.hybrid_deterministic_equivalent(problem, fitted, z, w, incumbent,
  acquisition=hb.EI())` is `hb.HybridNLP(problem, fitted, z, w, objective=hb.EI(incumbent))`.
  It holds an `NLP` as `.nlp` rather than being one, so `.f`, `.g` and the bounds are on
  `.nlp`.
- `objective` is any function of f under the scenarios and the weights, and `hb.mean` gives
  the expected objective, or the nominal one over one scenario at `z = 0`.
- **Breaking:** `hb.EI` and `hb.SoftEI` hold the incumbent: `hb.EI(incumbent)` and
  `hb.SoftEI(incumbent, temperature=...)`. `hybrid_bo.acquisition` is `hybrid_bo.bo`.
- **Breaking:** the plots show the objective that was minimized, so EI is `-EI`, SoftEI is
  `-log SoftEI` and the trial point sits at a minimum. `hb.plots.acquisition_on_grid` is
  `hb.plots.objective_on_grid`, and `from_objective` is gone.
- **Breaking:** `solve` returns values: `u, x, y, value` from a `HybridNLP`, and
  `w, value` from an `NLP`, so the trial point of `hb.standard_acquisition` is the first of
  them. `Solution` and `ScenarioSolution` are gone, and the plots take what `solve` returned.
- **Breaking:** `find_root(problem, p, starts, tol)` is `problem.solve(p, starts, tol)`, and
  `RootProblem.newton` is private.
- `NLP.solve_from(starts)` solves from given starts, replacing the private
  `_solve_with_multistart`.
- `minmax`, `standard`, `scale` and `unscale` are `Scaler.minmax`, `Scaler.standard`,
  `Scaler.scale` and `Scaler.unscale`.

## 0.8.0 (2026-11-06)

The library is rewritten to read like the math, for a scientist reading the code: methods
that return data replace many private functions, and the names say what things are. The
regression results don't change. Nearly every call changes, so see the README for the new
loop.

- **Breaking:** `hb.hybrid_trial_point`, `hb.standard_trial_point` and `hb.hybrid_acquisition`
  are gone: build with `hb.hybrid_deterministic_equivalent` or `hb.standard_acquisition`,
  call `.solve(rng)`, and pass the same object to `hb.plots.iteration`, which no longer
  takes the scenarios.
- **Breaking:** `hb.SAA` and `hb.SoftSAA` are `hb.EI` and `hb.SoftEI`, passed as
  `acquisition=`: the sample average is what the deterministic equivalent does, and SoftEI
  reformulates the acquisition function. `unknown_states_as_variables` is a keyword of
  `hb.hybrid_deterministic_equivalent`.
- **Breaking:** `hb.saa_scenarios` and `Scenarios` are gone: pass draws `z` and weights `w`.
  `AcquisitionSolution` is `Solution` for standard BO, whose trial point is `solution.w`, and
  `ScenarioSolution` for hybrid BO, with `u`, `x` and `y`.
- **Breaking:** `hb.replay` and `Replay` are gone: `hb.plots.acquisition_on_grid` gives the
  acquisition and a `Dataset` per scenario. `hybrid_bo.evaluation` is now
  `hybrid_bo.reconstruction`.
- **Breaking:** `reconstruct_states` solves with CasADi's Newton method instead of scipy's
  `fsolve`, which is no longer a dependency. With several roots in the bounds, it may find
  a different one. Known equations that can't determine the unmeasured states now raise
  `RootFindingError` at every row, where `fsolve` could return an arbitrary value.
- **Breaking:** Ctrl-C during IPOPT skips one start rather than stopping the solve.
- **Breaking:** `solve_with_multistart` is private: call `.solve(rng)` on an `NLP`.
- **Breaking:** `hb.fit_gp(gp, X, Y, rng)` is `gp.fit(X, Y, rng)`, `predict(fitted, X)` is
  `fitted.predict(X)`, and `log_marginal_likelihood` is a method of `FittedGP`.
- **Breaking:** `2.0 * kernel` is gone: write `Scale(kernel, 2.0)`. `covariance` and
  `prior_mean` are the methods `kernel.covariance(X1, X2)` and `mean.at(X)`.
- **Breaking:** A fit raises `ValueError` for shapes that don't match and for NaN or
  infinity, and `predict` for a numeric X that isn't a finite 2-D array with the columns of
  the training inputs.
- **Breaking:** `HybridBOError` is gone; catch `NoSolutionError` or `RootFindingError`.
- **Breaking:** The scalers `hb.gp.minmax`, `standard`, `scale` and `unscale` are in
  `hybrid_bo.utils`, and `minmax` maps onto [0, 1] only. `hybrid_bo.sampling` is gone.
- New: `hb.expected_improvement` is public.
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
