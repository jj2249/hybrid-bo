# Changelog

## Unreleased

- `hb.gp.Matern52(lengthscale_prior=True)` fits a learned lengthscale by MAP, with
  each log lengthscale ~ N(sqrt(2) + log(d) / 2, 3) over d inputs (Hvarfner et al.,
  2024), instead of by maximum likelihood. Off by default.

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
