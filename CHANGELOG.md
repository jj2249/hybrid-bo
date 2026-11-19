# Changelog

## Unreleased

- `hb.plots.solution` and `hb.plots.iteration` draw f, and an unknown state whose GP
  takes more than u, as the weighted mean over the scenarios with a band of ±2
  standard deviations, in place of one line per scenario.

## 0.2.0 (2026-11-19)

Measure any P of the states, give each experiment a fixed configuration c, and fit
GPs with the defaults of Hvarfner et al. (2024). Two changes break v0.1.0 code: the
`measured` argument of `from_measurements`, and the new field `c` of `Dataset`.

- **Breaking:** `hb.Dataset.from_measurements` takes `measured`, a dict of any P of
  the states x and y and their values (n,), in place of `y` (n, P). g is solved for
  the other states, so a system can measure an x and infer a y. Replace `y=values`
  with `measured={y: values[:, 0]}`.
- A fixed configuration c (K,): known values that change between experiments but
  aren't chosen, such as a feed's composition. `hb.HybridNLP(c=[...])` lets g and f
  read them as `v.<name>`, `from_measurements(c=(n, K))` and `solve(c=(K,))` take
  them, and `Dataset` and `Solution` store them. Unless the problem has a
  configuration, nothing changes.
- **Breaking:** `hb.Dataset` has a field `c` (n, K) after `u`, so a dataset built
  directly takes it too, and `Dataset.load` needs a file saved with it.
- `hb.Var` needs no `start` when its bounds are infinite; its `start` is then None.
  `HybridNLP` raises for a u, x or y without one, since those are sampled.
- `hb.gp.Matern52(lengthscale_prior=True)` fits a learned lengthscale by MAP, with
  each log lengthscale ~ N(sqrt(2) + log(d) / 2, 3) over d inputs (Hvarfner et al.,
  2024), instead of by maximum likelihood. Off by default.
- `hb.gp.Matern52()` fixes the output scale at 1, the variance of the standardized
  outputs, instead of learning it (Hvarfner et al., 2024). Pass `outputscale=None` to
  learn it as before. This changes default fits.

## 0.1.0 (2026-11-10)

First release. Minimize `f(x, y, u)` subject to known equations `0 = g(x, y, u)` and
unknown equations `y = h(x, u)`, each learned by a GP.

- `hb.HybridNLP` states the problem from `hb.Var`s and the callables `g` and `f`,
  traced once by CasADi for exact derivatives.
- `nlp.solve(method, rng, surrogate=..., objective=...)` solves it once, by `hb.SAA`
  (a sample average over scenarios) or `hb.Exact` (closed form, for standard BO), for
  an objective from `hb.obj` (`Mean`, `EI`, `SoftEI`), and returns a `hb.Solution`.
- `hb.gp.GP` fits a `FittedGP` per unknown state by maximum likelihood, with a
  Matern 5/2 kernel and a zero mean.
- `hb.Dataset` holds the measured points, builds them from measurements, extends them
  and saves them.
- `hb.plots.iteration` and `hb.plots.solution` draw a solved iteration from the stored
  `Solution`.
