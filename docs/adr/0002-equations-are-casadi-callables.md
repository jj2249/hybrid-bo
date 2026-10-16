# Equations are callables traced by CasADi

Users declare variables as plain `Var` dataclasses and write the known equations, the
objective and the oracle as callables over the variables by name
(`g=lambda v: [v.x + cas.exp(v.x) - v.y]`). The library calls each callable once with
CasADi symbols to build its `cas.Function`s, so every solve gets exact derivatives from
CasADi's automatic differentiation, and the cost after construction is the same as
hand-written `cas.Function`s.

## Considered Options

- **`Var` subclasses `cas.SX`**, so equations use the variables directly (`v.x` becomes
  `x`). Works, but depends on subclassing a generated wrapper class, cannot be a frozen
  dataclass, and `var in list` raises because CasADi's `==` is symbolic.
- **A `Var` that wraps a symbol** and forwards arithmetic: `cas.exp(var)` rejects it.
- **Arbitrary Python callables** (e.g. an external thermodynamics package as the
  oracle), evaluated numerically with finite-difference Jacobians: rejected because
  finite differences perform worse in root-finding and optimization. Such an oracle can
  still be wrapped in a custom measurement.

## Consequences

The `use_jacobian` option of the evaluation (`fsolve` without the CasADi Jacobian) was
removed: with every equation traceable it had no use, and both examples set it to
`True`. If a case turns up that needs it, it can come back as a root-finding setting.
