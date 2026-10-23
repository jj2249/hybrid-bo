# Defaults never choose the method

**A default may set how hard or how precisely we search, never what we search for.**
An argument that changes which problem a method solves, or which method it is, has no
default: the acquisition function (`acquisition` of both BO settings), hybrid BO's
approximation (`approximation`), the number of scenarios (`n_scenarios` of `SAA` and
`SoftSAA`), LCB's `std_factor`, SoftSAA's `temperature` and the noise (`noise_std` of
`oracle_measure` and of `GP`). Leaving one out is a `TypeError` at construction, so an
experiment shows which method it runs.

Defaults stay for search effort, numerics and plumbing: multistart starts
(`n_starts`, `n_starts_gp`), tolerances, the solver, the GP's `jitter`, kernel
hyperparameter priors and `ard` (they act in scaled units, so they don't depend on the
problem), `Var` bounds and start bounds, the study `seed` and plot flags. A value
derived from other arguments (start bounds from finite bounds) is a default too, when
the derivation is obvious and recorded in the run (DESIGN.md, principle 8).
`unknown_states_as_variables` keeps its default of `True`: it is a formulation of the
same acquisition problem, with the same optimum either way.

The settings and choice dataclasses (`EI`, `LCB`, `SAA`, `SoftSAA`, `Linearization`,
`HybridBOSettings`, `StandardBOSettings`, `GP`, `StudySettings`) are keyword-only, so a
call names what it sets (`SoftSAA(temperature=1e-6, n_scenarios=25, ...)`) rather
than relying on argument position.

## Worked examples

- **`temperature`.** SoftSAA's temperature is in units of the objective. A default
  would be wrong by orders of magnitude on most problems, and it sets how close
  SoftSAA-EI is to SAA-EI, so it says what we search for. It never had one (DESIGN.md,
  SoftSAA).
- **`noise_std`.** A default of 0 reads as "the measurement is noise-free", which is
  a claim about the system, not a search setting. A GP whose noise is left at 0 on
  noisy data interpolates the noise, and an oracle measurement left at 0 benchmarks a
  different problem. The measurement and the GP each state it, so an experiment that
  forgets one fails instead of silently disagreeing with the other.

## Considered Options

- **Defaults for the common case** (EI, SAA with 25 scenarios, no noise; what this
  replaces): shorter experiments, but someone reading one couldn't see which method it
  ran, and a change of a default would change every experiment that relied on it.
- **Keep the defaults, since the run records them**: a saved study's artifact holds
  the settings in full, so its reader sees the method. The experiment's reader
  doesn't, and the shipped experiments serve as examples (ADR 0006).

## Consequences

Experiments, tests and docs spell out each of these arguments, which makes them a
little longer. Saved studies are unaffected: they hold every field, so they load as
before. A new setting is sorted by the rule when it is added: one that chooses the
method gets no default.
